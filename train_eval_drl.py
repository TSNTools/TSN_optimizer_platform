from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env import TSNDefragEnv
from heuristic import fragmentation_metrics
from tsn_scheduling_problem import TSNScheduling


class RewardTraceCallback(BaseCallback):
    def __init__(self, sink: List[Dict[str, float]], phase: str, instance_name: str):
        super().__init__()
        self.sink = sink
        self.phase = phase
        self.instance_name = instance_name

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0])
        mean_reward = float(np.mean(rewards))
        self.sink.append(
            {
                "phase": self.phase,
                "instance": self.instance_name,
                "global_step": int(self.num_timesteps),
                "reward": mean_reward,
            }
        )
        return True


def make_env(problem: TSNScheduling, instance_path: str, reward_weights: Dict[str, float], seed: int):
    def _build():
        usecase = problem.import_scheduling_usecase(instance_path)
        env = TSNDefragEnv(
            problem=problem,
            usecase=usecase,
            single_move_per_step=False,
            init_from_usecase_schedule=True,
            reward_weights=reward_weights,
        )
        env.reset(seed=seed)
        return env

    return _build


def metrics_snapshot(
    env: TSNDefragEnv,
    frame_sizes: Sequence[int],
    mix_percentages: Sequence[float],
    incoming_stream_count: int,
) -> Dict[str, object]:
    compact = env.get_current_compact()
    return fragmentation_metrics(
        compact=compact,
        wind_lengths=np.asarray(env.sched_eval.windLengths),
        cycle_ns=int(env.cycle_ns),
        thresholds=frame_sizes,
        reference_speed_mbps=float(np.mean(env.sched_eval.port_speeds)) if env.sched_eval.port_speeds.size else 1000.0,
        incoming_stream_count=int(incoming_stream_count),
        frame_mix_percentages=mix_percentages,
    )


def flatten_metric_row(
    instance_name: str,
    phase: str,
    before: Dict[str, object],
    after: Dict[str, object],
    frame_sizes: Sequence[int],
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "instance": instance_name,
        "phase": phase,
    }

    scalar_keys = [
        "largest_hole_mean",
        "largest_hole_max",
        "gap_count_total",
        "usable_space_total_ns",
        "accommodated_streams_mix_total",
        "service_block_ratio",
    ]
    for key in scalar_keys:
        b = float(before.get(key, np.nan))
        a = float(after.get(key, np.nan))
        row[f"before_{key}"] = b
        row[f"after_{key}"] = a
        row[f"delta_{key}"] = a - b

    b_usable = before.get("usable_holes_total", {})
    a_usable = after.get("usable_holes_total", {})
    b_acc = before.get("accommodated_streams_total", {})
    a_acc = after.get("accommodated_streams_total", {})

    for fs in frame_sizes:
        fs_key = int(fs)
        bu = float(b_usable.get(fs_key, np.nan))
        au = float(a_usable.get(fs_key, np.nan))
        ba = float(b_acc.get(fs_key, np.nan))
        aa = float(a_acc.get(fs_key, np.nan))

        row[f"before_usable_holes_{fs_key}B"] = bu
        row[f"after_usable_holes_{fs_key}B"] = au
        row[f"delta_usable_holes_{fs_key}B"] = au - bu

        row[f"before_accommodated_streams_{fs_key}B"] = ba
        row[f"after_accommodated_streams_{fs_key}B"] = aa
        row[f"delta_accommodated_streams_{fs_key}B"] = aa - ba

    return row


def rollout_and_collect(
    model: PPO,
    vec_env: DummyVecEnv,
    eval_steps: int,
    frame_sizes: Sequence[int],
    mix_percentages: Sequence[float],
    incoming_stream_count: int,
    phase: str,
    instance_name: str,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    base_env: TSNDefragEnv = vec_env.envs[0]
    before = metrics_snapshot(base_env, frame_sizes, mix_percentages, incoming_stream_count)

    obs = vec_env.reset()
    rewards: List[Dict[str, float]] = []
    for step in range(int(eval_steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, _, info = vec_env.step(action)
        rewards.append(
            {
                "phase": phase,
                "instance": instance_name,
                "global_step": int(step + 1),
                "reward": float(reward[0]),
                "is_feasible": float(bool(info[0].get("is_feasible", False))),
                "new_violation_sum": float(info[0].get("new_violation_sum", 0)),
            }
        )

    after = metrics_snapshot(base_env, frame_sizes, mix_percentages, incoming_stream_count)
    row = flatten_metric_row(
        instance_name=instance_name,
        phase=phase,
        before=before,
        after=after,
        frame_sizes=frame_sizes,
    )
    return row, rewards


def plot_reward_curves(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 4))
    for (phase, instance), grp in df.groupby(["phase", "instance"]):
        x = np.arange(len(grp))
        y = grp["reward"].to_numpy(dtype=float)
        plt.plot(x, y, label=f"{phase}:{instance}", alpha=0.8)
    plt.title("Reward Over Steps")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_comparison(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    metrics = [
        "largest_hole_mean",
        "largest_hole_max",
        "gap_count_total",
        "usable_space_total_ns",
        "accommodated_streams_mix_total",
        "service_block_ratio",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        b_col = f"before_{metric}"
        a_col = f"after_{metric}"
        if b_col not in df.columns or a_col not in df.columns:
            continue

        by_phase = df.groupby("phase")[[b_col, a_col]].mean(numeric_only=True)
        labels = list(by_phase.index)
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, by_phase[b_col].to_numpy(), width=width, label="before")
        ax.bar(x + width / 2, by_phase[a_col].to_numpy(), width=width, label="after")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(metric)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_list(text: str, cast_type):
    return [cast_type(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on instances 1-4, test on instance 5 with heuristic-aligned metrics.")
    parser.add_argument("--config", default="uc_config.cfg")
    parser.add_argument("--train-instances", default="instance1.json,instance2.json,instance3.json,instance4.json")
    parser.add_argument("--test-instance", default="instance5.json")
    parser.add_argument("--timesteps-per-train-instance", type=int, default=10_000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-sizes", default="64,128,256")
    parser.add_argument("--mix-percentages", default="50,25,25")
    parser.add_argument("--incoming-stream-count", type=int, default=100)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_instances = parse_list(args.train_instances, str)
    test_instance = args.test_instance
    frame_sizes = parse_list(args.frame_sizes, int)
    mix_percentages = parse_list(args.mix_percentages, float)

    if len(frame_sizes) != len(mix_percentages):
        raise ValueError("frame_sizes and mix_percentages must have the same length")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reward profile: balanced default.
    reward_weights = {
        "w_admission": 1.0,
        "w_violation": 0.50,
    }

    problem = TSNScheduling(args.config)

    reward_trace: List[Dict[str, float]] = []
    metric_rows: List[Dict[str, float]] = []

    # Sequential train on instance1..instance4
    model: PPO | None = None
    for idx, train_path in enumerate(train_instances):
        env_fn = make_env(problem, train_path, reward_weights, seed=args.seed + idx)
        vec_env = DummyVecEnv([env_fn])

        if model is None:
            model = PPO("MultiInputPolicy", vec_env, verbose=1, seed=args.seed)
        else:
            model.set_env(vec_env)

        cb = RewardTraceCallback(reward_trace, phase="train", instance_name=train_path)
        model.learn(total_timesteps=int(args.timesteps_per_train_instance), reset_num_timesteps=False, callback=cb)

        train_metric_row, train_eval_rewards = rollout_and_collect(
            model=model,
            vec_env=vec_env,
            eval_steps=int(args.eval_steps),
            frame_sizes=frame_sizes,
            mix_percentages=mix_percentages,
            incoming_stream_count=int(args.incoming_stream_count),
            phase="train",
            instance_name=train_path,
        )
        metric_rows.append(train_metric_row)
        reward_trace.extend(train_eval_rewards)

    if model is None:
        raise RuntimeError("Training did not start.")

    # Evaluate on instance5
    test_env = DummyVecEnv([make_env(problem, test_instance, reward_weights, seed=args.seed + 999)])
    eval_metric_row, eval_rewards = rollout_and_collect(
        model=model,
        vec_env=test_env,
        eval_steps=int(args.eval_steps),
        frame_sizes=frame_sizes,
        mix_percentages=mix_percentages,
        incoming_stream_count=int(args.incoming_stream_count),
        phase="eval",
        instance_name=test_instance,
    )
    metric_rows.append(eval_metric_row)
    reward_trace.extend(eval_rewards)

    reward_df = pd.DataFrame(reward_trace)
    metrics_df = pd.DataFrame(metric_rows)

    reward_csv = out_dir / "drl_reward_trace.csv"
    metrics_csv = out_dir / "drl_metrics_summary.csv"
    reward_png = out_dir / "drl_reward_over_steps.png"
    metric_png = out_dir / "drl_metric_before_after.png"
    model_path = out_dir / "ppo_tsn_defrag_model.zip"

    reward_df.to_csv(reward_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    plot_reward_curves(reward_df, reward_png)
    plot_metric_comparison(metrics_df, metric_png)
    model.save(model_path)

    print(f"Saved reward trace: {reward_csv}")
    print(f"Saved metrics summary: {metrics_csv}")
    print(f"Saved reward plot: {reward_png}")
    print(f"Saved metric plot: {metric_png}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
