from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env import TSNDefragEnv
from tsn_scheduling_problem import TSNScheduling


@dataclass
class StageConfig:
    name: str
    incoming_stream_count: int
    frame_size: int
    timesteps: int


class StageTraceCallback(BaseCallback):
    def __init__(self, sink: List[Dict[str, float]], stage_name: str, instance_name: str):
        super().__init__()
        self.sink = sink
        self.stage_name = stage_name
        self.instance_name = instance_name

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        infos = self.locals.get("infos", [{}])
        info0 = infos[0] if isinstance(infos, list) and infos else {}

        self.sink.append(
            {
                "stage": self.stage_name,
                "instance": self.instance_name,
                "global_step": int(self.num_timesteps),
                "reward": float(np.mean(rewards)),
                "admission_rate": float(info0.get("new_admission_rate", np.nan)),
                "attempted_violation_sum": float(info0.get("attempted_violation_sum", np.nan)),
                "final_violation_sum": float(info0.get("final_violation_sum", info0.get("new_violation_sum", np.nan))),
                "rollback_applied": float(bool(info0.get("rollback_applied", False))),
            }
        )
        return True


def parse_list(text: str, cast_type):
    return [cast_type(x.strip()) for x in text.split(",") if x.strip()]


def create_numbered_run_dir(base_out_dir: Path) -> Path:
    base_out_dir.mkdir(parents=True, exist_ok=True)
    run_numbers: List[int] = []
    for p in base_out_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        suffix = p.name[4:]
        if suffix.isdigit():
            run_numbers.append(int(suffix))

    next_run = (max(run_numbers) + 1) if run_numbers else 1
    run_dir = base_out_dir / f"run_{next_run:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_env(problem: TSNScheduling, instance_path: str, seed: int, env_kwargs: Dict[str, object]):
    def _build():
        usecase = problem.import_scheduling_usecase(instance_path)
        env = TSNDefragEnv(problem=problem, usecase=usecase, **env_kwargs)
        env.reset(seed=seed)
        return env

    return _build


def build_stage_env_kwargs(base_env_kwargs: Dict[str, object], stage_config: StageConfig) -> Dict[str, object]:
    env_kwargs = dict(base_env_kwargs)
    env_kwargs["reward_frame_sizes"] = [int(stage_config.frame_size)]
    env_kwargs["reward_mix_percentages"] = [100.0]
    env_kwargs["reward_incoming_stream_count"] = int(stage_config.incoming_stream_count)
    return env_kwargs


def evaluate_model(model: PPO, vec_env: DummyVecEnv, eval_steps: int) -> Dict[str, float]:
    obs = vec_env.reset()
    rewards: List[float] = []
    admissions: List[float] = []
    attempted_viols: List[float] = []
    final_viols: List[float] = []

    for _ in range(int(eval_steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        info0 = infos[0] if infos else {}

        rewards.append(float(reward[0]))
        admissions.append(float(info0.get("new_admission_rate", np.nan)))
        attempted_viols.append(float(info0.get("attempted_violation_sum", np.nan)))
        final_viols.append(float(info0.get("final_violation_sum", info0.get("new_violation_sum", np.nan))))

        if bool(dones[0]):
            obs = vec_env.reset()

    return {
        "eval_reward_mean": float(np.nanmean(rewards)) if rewards else np.nan,
        "eval_admission_mean": float(np.nanmean(admissions)) if admissions else np.nan,
        "eval_attempted_violation_mean": float(np.nanmean(attempted_viols)) if attempted_viols else np.nan,
        "eval_final_violation_mean": float(np.nanmean(final_viols)) if final_viols else np.nan,
    }


def plot_stage_metrics(df: pd.DataFrame, stage_dir: Path, ma_window: int) -> None:
    if df.empty:
        return

    work = df.copy()
    work["step_idx"] = np.arange(1, len(work) + 1)
    work["reward_ma"] = work["reward"].rolling(window=max(1, int(ma_window)), min_periods=1).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(work["step_idx"], work["reward"], alpha=0.35, label="reward_raw")
    plt.plot(work["step_idx"], work["reward_ma"], linewidth=2.0, label=f"reward_ma_{int(ma_window)}")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward (Raw + Moving Average)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(stage_dir / "reward_raw_and_ma.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(work["step_idx"], work["admission_rate"], label="admission_rate")
    plt.xlabel("Step")
    plt.ylabel("Admission Rate")
    plt.title("Admission Rate Over Steps")
    plt.tight_layout()
    plt.savefig(stage_dir / "admission_rate_over_steps.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(work["step_idx"], work["attempted_violation_sum"], label="attempted_violation_sum")
    plt.plot(work["step_idx"], work["final_violation_sum"], label="final_violation_sum")
    plt.xlabel("Step")
    plt.ylabel("Violations")
    plt.title("Violations Over Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(stage_dir / "violations_over_steps.png")
    plt.close()


def train_stage(
    stage_config: StageConfig,
    model_path: Optional[Path] = None,
    *,
    problem: TSNScheduling,
    train_instances: Sequence[str],
    base_env_kwargs: Dict[str, object],
    ppo_kwargs: Dict[str, object],
    seed: int,
    eval_steps: int,
    run_dir: Path,
    reward_ma_window: int,
    auto_advance_metric: str,
    auto_advance_threshold: float,
) -> Tuple[Path, Dict[str, float]]:
    """Train one curriculum stage; optionally continue from a previously saved PPO model."""
    stage_name = stage_config.name.lower()
    stage_dir = run_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    stage_env_kwargs = build_stage_env_kwargs(base_env_kwargs, stage_config)

    trace_rows: List[Dict[str, float]] = []
    model: Optional[PPO] = None
    early_advance_triggered = False

    for idx, instance_path in enumerate(train_instances):
        env_fn = make_env(problem, instance_path, seed + idx, stage_env_kwargs)
        vec_env = DummyVecEnv([env_fn])

        if model is None and model_path is None:
            # EASY stage starts from scratch exactly once.
            model = PPO("MultiInputPolicy", vec_env, **ppo_kwargs)
        elif model is None and model_path is not None:
            # MEDIUM/HARD stage must load previous stage checkpoint.
            model = PPO.load(str(model_path), env=vec_env)
        else:
            model.set_env(vec_env)

        cb = StageTraceCallback(trace_rows, stage_name=stage_name, instance_name=instance_path)
        model.learn(total_timesteps=int(stage_config.timesteps), reset_num_timesteps=False, callback=cb)

        eval_summary = evaluate_model(model, vec_env, eval_steps=int(eval_steps))
        eval_summary["instance"] = instance_path
        eval_summary["stage"] = stage_name

        if auto_advance_metric != "none":
            metric_value = float(eval_summary.get(auto_advance_metric, np.nan))
            if np.isfinite(metric_value) and metric_value >= float(auto_advance_threshold):
                early_advance_triggered = True
                break

    if model is None:
        raise RuntimeError(f"Stage '{stage_name}' did not create or load a PPO model.")

    stage_model_path = stage_dir / f"model_{stage_name}.zip"
    model.save(str(stage_model_path))

    trace_df = pd.DataFrame(trace_rows)
    trace_csv = stage_dir / "stage_trace.csv"
    trace_df.to_csv(trace_csv, index=False)
    plot_stage_metrics(trace_df, stage_dir=stage_dir, ma_window=int(reward_ma_window))

    stage_summary = {
        "stage": stage_name,
        "incoming_stream_count": int(stage_config.incoming_stream_count),
        "frame_size": int(stage_config.frame_size),
        "timesteps": int(stage_config.timesteps),
        "model_path": str(stage_model_path),
        "n_trace_rows": int(len(trace_rows)),
        "early_advance_triggered": bool(early_advance_triggered),
        "auto_advance_metric": auto_advance_metric,
        "auto_advance_threshold": float(auto_advance_threshold),
    }
    (stage_dir / "stage_summary.json").write_text(json.dumps(stage_summary, indent=2), encoding="utf-8")

    print(f"[{stage_name}] saved model: {stage_model_path}")
    print(f"[{stage_name}] saved trace: {trace_csv}")

    return stage_model_path, stage_summary


def _instance_shape(problem: TSNScheduling, instance_path: str) -> Tuple[int, int]:
    usecase = problem.import_scheduling_usecase(instance_path)
    with problem.use_instance(usecase):
        return int(problem.n_streams), int(problem.n_ports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum learning pipeline for PPO: EASY -> MEDIUM -> HARD")
    parser.add_argument("--config", default="uc_config.cfg")
    parser.add_argument("--train-instances", default="instance1.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs")

    parser.add_argument("--reward-mode", default="weighted_fragmentation", choices=["weighted_fragmentation", "gap_improvement_binary", "hole_improvement_binary"])
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--w-gap", type=float, default=1.0)
    parser.add_argument("--w-hole", type=float, default=0.5)
    parser.add_argument("--w-move", type=float, default=0.01)
    parser.add_argument("--w-invalid", type=float, default=1.0)
    parser.add_argument("--force-current-start-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-candidate-validation", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--easy-incoming", type=int, default=50)
    parser.add_argument("--easy-frame-size", type=int, default=512)
    parser.add_argument("--easy-timesteps", type=int, default=10_000)

    parser.add_argument("--medium-incoming", type=int, default=180)
    parser.add_argument("--medium-frame-size", type=int, default=1280)
    parser.add_argument("--medium-timesteps", type=int, default=10_000)

    parser.add_argument("--hard-incoming", type=int, default=400)
    parser.add_argument("--hard-frame-size", type=int, default=2300)
    parser.add_argument("--hard-timesteps", type=int, default=10_000)

    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--reward-ma-window", type=int, default=100)

    parser.add_argument(
        "--auto-advance-metric",
        default="none",
        choices=["none", "eval_reward_mean", "eval_admission_mean"],
        help="Bonus option: stage can advance early when threshold is met.",
    )
    parser.add_argument("--auto-advance-threshold", type=float, default=0.0)

    args = parser.parse_args()

    np.random.seed(int(args.seed))

    run_dir = create_numbered_run_dir(Path(args.out_dir))
    train_instances = parse_list(args.train_instances, str)

    problem = TSNScheduling(args.config)

    shape_ref = _instance_shape(problem, train_instances[0])
    for path in train_instances[1:]:
        shp = _instance_shape(problem, path)
        if shp != shape_ref:
            raise ValueError(
                "All train instances must share shape (n_streams, n_ports) for one PPO policy. "
                f"Reference={shape_ref}, got {shp} for instance '{path}'."
            )

    base_env_kwargs: Dict[str, object] = {
        "single_move_per_step": True,
        "init_from_usecase_schedule": True,
        "reward_mode": str(args.reward_mode),
        "reward_weights": {
            "w_gap": float(args.w_gap),
            "w_hole": float(args.w_hole),
            "w_move": float(args.w_move),
            "w_invalid": float(args.w_invalid),
        },
        "max_steps": int(args.max_steps),
        "max_candidates": int(args.max_candidates),
        "force_current_start_candidate": bool(args.force_current_start_candidate),
        "debug_candidate_validation": bool(args.debug_candidate_validation),
    }

    ppo_kwargs: Dict[str, object] = {
        "n_steps": 256,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "verbose": 1,
        "seed": int(args.seed),
    }

    stages = [
        StageConfig(name="easy", incoming_stream_count=int(args.easy_incoming), frame_size=int(args.easy_frame_size), timesteps=int(args.easy_timesteps)),
        StageConfig(name="medium", incoming_stream_count=int(args.medium_incoming), frame_size=int(args.medium_frame_size), timesteps=int(args.medium_timesteps)),
        StageConfig(name="hard", incoming_stream_count=int(args.hard_incoming), frame_size=int(args.hard_frame_size), timesteps=int(args.hard_timesteps)),
    ]

    plan = {
        "seed": int(args.seed),
        "train_instances": train_instances,
        "stages": [asdict(s) for s in stages],
        "base_env_kwargs": base_env_kwargs,
        "ppo_kwargs": ppo_kwargs,
        "auto_advance_metric": str(args.auto_advance_metric),
        "auto_advance_threshold": float(args.auto_advance_threshold),
    }
    (run_dir / "curriculum_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    easy_model_path, easy_summary = train_stage(
        stages[0],
        model_path=None,
        problem=problem,
        train_instances=train_instances,
        base_env_kwargs=base_env_kwargs,
        ppo_kwargs=ppo_kwargs,
        seed=int(args.seed),
        eval_steps=int(args.eval_steps),
        run_dir=run_dir,
        reward_ma_window=int(args.reward_ma_window),
        auto_advance_metric=str(args.auto_advance_metric),
        auto_advance_threshold=float(args.auto_advance_threshold),
    )

    medium_model_path, medium_summary = train_stage(
        stages[1],
        model_path=easy_model_path,
        problem=problem,
        train_instances=train_instances,
        base_env_kwargs=base_env_kwargs,
        ppo_kwargs=ppo_kwargs,
        seed=int(args.seed) + 1000,
        eval_steps=int(args.eval_steps),
        run_dir=run_dir,
        reward_ma_window=int(args.reward_ma_window),
        auto_advance_metric=str(args.auto_advance_metric),
        auto_advance_threshold=float(args.auto_advance_threshold),
    )

    hard_model_path, hard_summary = train_stage(
        stages[2],
        model_path=medium_model_path,
        problem=problem,
        train_instances=train_instances,
        base_env_kwargs=base_env_kwargs,
        ppo_kwargs=ppo_kwargs,
        seed=int(args.seed) + 2000,
        eval_steps=int(args.eval_steps),
        run_dir=run_dir,
        reward_ma_window=int(args.reward_ma_window),
        auto_advance_metric=str(args.auto_advance_metric),
        auto_advance_threshold=float(args.auto_advance_threshold),
    )

    final_summary = {
        "run_dir": str(run_dir),
        "easy": easy_summary,
        "medium": medium_summary,
        "hard": hard_summary,
        "final_model": str(hard_model_path),
    }
    (run_dir / "curriculum_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print(f"Saved run directory: {run_dir}")
    print(f"Final model: {hard_model_path}")


if __name__ == "__main__":
    main()
