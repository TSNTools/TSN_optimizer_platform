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
        infos = self.locals.get("infos", [{}])
        mean_reward = float(np.mean(rewards))
        info0 = infos[0] if isinstance(infos, list) and infos else {}

        self.sink.append(
            {
                "phase": self.phase,
                "instance": self.instance_name,
                "global_step": int(self.num_timesteps),
                "reward": mean_reward,
                # Attempted violations are the primary safety/debug signal.
                # Final/post-rollback violations can remain near zero.
                "attempted_violation_sum": float(info0.get("attempted_violation_sum", np.nan)),
                "final_violation_sum": float(info0.get("final_violation_sum", info0.get("new_violation_sum", np.nan))),
                "admission_rate": float(info0.get("new_admission_rate", np.nan)),
                "rollback_applied": float(bool(info0.get("rollback_applied", False))),
                "acted_stream_id": float(info0.get("acted_stream_id", np.nan)),
                "next_stream_id": float(info0.get("next_stream_id", np.nan)),
                "selected_candidate_index": float(info0.get("selected_candidate_index", np.nan)),
                "selected_first_offset_ns": float(info0.get("selected_first_offset_ns", np.nan))
                if info0.get("selected_first_offset_ns", None) is not None
                else np.nan,
                "delta_gap": float(info0.get("delta_gap", np.nan)),
                "delta_hole": float(info0.get("delta_hole", np.nan)),
                "delta_attempted_violation": float(info0.get("delta_attempted_violation", np.nan)),
                "delta_final_violation": float(info0.get("delta_final_violation", np.nan)),
                "invalid_selection": float(bool(info0.get("invalid_selection", False))),
                "valid_candidate_count": float(info0.get("valid_candidate_count", np.nan)),
                "attempted_violation_dict": info0.get("attempted_violation_dict", {}),
                "selected_candidate_was_current_start": float(bool(info0.get("selected_candidate_was_current_start", False))),
                "selected_candidate_was_fallback_unsafe": float(bool(info0.get("selected_candidate_was_fallback_unsafe", False))),
                "selected_candidate_passed_full_evaluator": float(bool(info0.get("selected_candidate_passed_full_evaluator", False))),
                "initial_violation_sum": float(info0.get("initial_violation_sum", np.nan)),
                "moved_stream": float(info0.get("moved_stream", np.nan))
                if info0.get("moved_stream", None) is not None
                else np.nan,
                "step_count": float(info0.get("step_count", np.nan)),
            }
        )
        return True


def make_env(
    problem: TSNScheduling,
    instance_path: str,
    reward_weights: Dict[str, float],
    seed: int,
    reward_frame_sizes: Sequence[int],
    reward_mix_percentages: Sequence[float],
    reward_incoming_stream_count: int,
    max_steps: int,
    max_candidates: int,
):
    def _build():
        usecase = problem.import_scheduling_usecase(instance_path)
        env = TSNDefragEnv(
            problem=problem,
            usecase=usecase,
            single_move_per_step=True,
            init_from_usecase_schedule=True,
            reward_weights=reward_weights,
            reward_frame_sizes=list(reward_frame_sizes),
            reward_mix_percentages=list(reward_mix_percentages),
            reward_incoming_stream_count=int(reward_incoming_stream_count),
            max_steps=int(max_steps),
            max_candidates=int(max_candidates),
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


def _violation_category_labels(env: TSNDefragEnv, compact: np.ndarray) -> List[str]:
    labels: List[str] = []
    for check_fn in env.sched_eval.active_checks:
        arr = check_fn(compact)
        row_count = int(arr.shape[0])
        name = check_fn.__name__

        if name == "_per_stream_window_sequence_violations" and row_count == 2:
            labels.extend(["stream_window_sequence", "interleaving"])
        elif name == "_per_stream_macrocycle_overlap_violations" and row_count == 1:
            labels.append("macrocycle_overlap")
        elif name == "_per_stream_talker_offset_violations" and row_count == 1:
            labels.append("talker_offset")
        elif name == "_per_stream_latency_violations_start_time" and row_count == 1:
            labels.append("maxlatency_violations")
        else:
            clean = name.replace("_per_stream_", "")
            for idx in range(row_count):
                labels.append(f"{clean}[{idx}]")
    return labels


def _violation_category_sums(env: TSNDefragEnv, compact: np.ndarray) -> Dict[str, int]:
    per_stream = env.sched_eval.per_stream_all_violations(compact)
    if per_stream.size == 0:
        return {}
    labels = _violation_category_labels(env, compact)
    row_sums = np.asarray(per_stream.sum(axis=1), dtype=int)
    return {label: int(row_sums[idx]) for idx, label in enumerate(labels)}


def flatten_metric_row(
    instance_name: str,
    phase: str,
    before: Dict[str, object],
    after: Dict[str, object],
    frame_sizes: Sequence[int],
    violation_before: Dict[str, int],
    violation_after: Dict[str, int],
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

    all_violation_keys = sorted(set(violation_before.keys()) | set(violation_after.keys()))
    for key in all_violation_keys:
        b = int(violation_before.get(key, 0))
        a = int(violation_after.get(key, 0))
        row[f"before_violation_{key}"] = float(b)
        row[f"after_violation_{key}"] = float(a)
        row[f"delta_violation_{key}"] = float(a - b)

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
    obs = vec_env.reset()
    base_env: TSNDefragEnv = vec_env.envs[0].unwrapped

    before_compact = base_env.get_current_compact()
    before = metrics_snapshot(base_env, frame_sizes, mix_percentages, incoming_stream_count)
    violation_before = _violation_category_sums(base_env, before_compact)

    rewards: List[Dict[str, float]] = []
    collected = 0
    while collected < int(eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        info0 = infos[0] if infos else {}

        rewards.append(
            {
                "phase": phase,
                "instance": instance_name,
                "global_step": int(collected + 1),
                "reward": float(reward[0]),
                "is_feasible": float(bool(info0.get("is_feasible", False))),
                "attempted_violation_sum": float(info0.get("attempted_violation_sum", np.nan)),
                "final_violation_sum": float(info0.get("final_violation_sum", info0.get("new_violation_sum", np.nan))),
                "admission_rate": float(info0.get("new_admission_rate", np.nan)),
                "rollback_applied": float(bool(info0.get("rollback_applied", False))),
                "acted_stream_id": float(info0.get("acted_stream_id", np.nan)),
                "next_stream_id": float(info0.get("next_stream_id", np.nan)),
                "selected_candidate_index": float(info0.get("selected_candidate_index", np.nan)),
                "selected_first_offset_ns": float(info0.get("selected_first_offset_ns", np.nan))
                if info0.get("selected_first_offset_ns", None) is not None
                else np.nan,
                "delta_gap": float(info0.get("delta_gap", np.nan)),
                "delta_hole": float(info0.get("delta_hole", np.nan)),
                "delta_attempted_violation": float(info0.get("delta_attempted_violation", np.nan)),
                "delta_final_violation": float(info0.get("delta_final_violation", np.nan)),
                "invalid_selection": float(bool(info0.get("invalid_selection", False))),
                "valid_candidate_count": float(info0.get("valid_candidate_count", np.nan)),
                "attempted_violation_dict": info0.get("attempted_violation_dict", {}),
                "selected_candidate_was_current_start": float(bool(info0.get("selected_candidate_was_current_start", False))),
                "selected_candidate_was_fallback_unsafe": float(bool(info0.get("selected_candidate_was_fallback_unsafe", False))),
                "selected_candidate_passed_full_evaluator": float(bool(info0.get("selected_candidate_passed_full_evaluator", False))),
                "initial_violation_sum": float(info0.get("initial_violation_sum", np.nan)),
                "moved_stream": float(info0.get("moved_stream", np.nan))
                if info0.get("moved_stream", None) is not None
                else np.nan,
                "step_count": float(info0.get("step_count", np.nan)),
            }
        )

        collected += 1
        if bool(dones[0]) and collected < int(eval_steps):
            obs = vec_env.reset()

    after_compact = base_env.get_current_compact()
    after = metrics_snapshot(base_env, frame_sizes, mix_percentages, incoming_stream_count)
    violation_after = _violation_category_sums(base_env, after_compact)

    row = flatten_metric_row(
        instance_name=instance_name,
        phase=phase,
        before=before,
        after=after,
        frame_sizes=frame_sizes,
        violation_before=violation_before,
        violation_after=violation_after,
    )
    return row, rewards


def plot_metric_per_instance_and_avg(
    df: pd.DataFrame,
    phase: str,
    metric_col: str,
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    work = df[df["phase"] == phase].copy()
    if work.empty or metric_col not in work.columns:
        return

    work = work.dropna(subset=[metric_col])
    if work.empty:
        return

    work["step_idx"] = work.groupby("instance").cumcount() + 1
    instances = sorted(work["instance"].dropna().unique().tolist())
    if not instances:
        return

    fig, axes = plt.subplots(len(instances) + 1, 1, figsize=(10, 2.6 * (len(instances) + 1)), sharex=False)
    if len(instances) + 1 == 1:
        axes = [axes]

    for i, instance_name in enumerate(instances):
        grp = work[work["instance"] == instance_name]
        x = grp["step_idx"].to_numpy(dtype=int)
        y = grp[metric_col].to_numpy(dtype=float)
        axes[i].plot(x, y, alpha=0.85)
        axes[i].set_title(str(instance_name))
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel(y_label)

    pivot = work.pivot_table(index="step_idx", columns="instance", values=metric_col, aggfunc="mean")
    avg = pivot.mean(axis=1)
    axes[-1].plot(avg.index.to_numpy(dtype=int), avg.to_numpy(dtype=float), color="black", linewidth=2.0)
    axes[-1].set_title(f"Average across {phase} instances")
    axes[-1].set_xlabel("Step")
    axes[-1].set_ylabel(y_label)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_moving_average_metric_per_instance_and_avg(
    df: pd.DataFrame,
    phase: str,
    metric_col: str,
    y_label: str,
    title: str,
    out_path: Path,
    window: int,
) -> None:
    """Moving-average per-instance and overall trend plot."""
    # Moving average smooths noisy per-step reward for clearer learning progress.
    work = df[df["phase"] == phase].copy()
    if work.empty or metric_col not in work.columns:
        return

    work = work.dropna(subset=[metric_col])
    if work.empty:
        return

    work["step_idx"] = work.groupby("instance").cumcount() + 1
    work["metric_ma"] = work.groupby("instance")[metric_col].transform(
        lambda s: s.rolling(window=max(1, int(window)), min_periods=1).mean()
    )

    instances = sorted(work["instance"].dropna().unique().tolist())
    if not instances:
        return

    fig, axes = plt.subplots(len(instances) + 1, 1, figsize=(10, 2.6 * (len(instances) + 1)), sharex=False)
    if len(instances) + 1 == 1:
        axes = [axes]

    for i, instance_name in enumerate(instances):
        grp = work[work["instance"] == instance_name]
        x = grp["step_idx"].to_numpy(dtype=int)
        y = grp["metric_ma"].to_numpy(dtype=float)
        axes[i].plot(x, y, alpha=0.9)
        axes[i].set_title(str(instance_name))
        axes[i].set_xlabel("Step")
        axes[i].set_ylabel(y_label)

    pivot = work.pivot_table(index="step_idx", columns="instance", values="metric_ma", aggfunc="mean")
    avg = pivot.mean(axis=1)
    axes[-1].plot(avg.index.to_numpy(dtype=int), avg.to_numpy(dtype=float), color="black", linewidth=2.0)
    axes[-1].set_title(f"Average across {phase} instances")
    axes[-1].set_xlabel("Step")
    axes[-1].set_ylabel(y_label)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_attempted_violation_category_means(df: pd.DataFrame, phase: str, out_path: Path) -> None:
    """Mean attempted violation counts per category over steps."""
    work = df[df["phase"] == phase].copy()
    if work.empty or "attempted_violation_dict" not in work.columns:
        return

    rows = [v for v in work["attempted_violation_dict"].tolist() if isinstance(v, dict) and v]
    if not rows:
        return

    categories = sorted({k for d in rows for k in d.keys()})
    if not categories:
        return

    means = [float(np.mean([float(d.get(cat, 0.0)) for d in rows])) for cat in categories]

    x = np.arange(len(categories))
    plt.figure(figsize=(max(10, 1.2 * len(categories)), 4.8))
    plt.bar(x, means)
    plt.xticks(x, categories, rotation=25, ha="right")
    plt.ylabel("Mean Attempted Violation Count")
    # Attempted categories reflect what the policy proposed pre-rollback.
    plt.title(f"Attempted Violation Categories ({phase})")
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


def plot_violation_categories_before_after(df: pd.DataFrame, phase: str, out_path: Path) -> None:
    work = df[df["phase"] == phase].copy()
    if work.empty:
        return

    before_cols = sorted([c for c in work.columns if c.startswith("before_violation_")])
    after_cols = sorted([c for c in work.columns if c.startswith("after_violation_")])
    if not before_cols or not after_cols:
        return

    categories = [c[len("before_violation_") :] for c in before_cols]
    before_vals = []
    after_vals = []
    for cat in categories:
        b_col = f"before_violation_{cat}"
        a_col = f"after_violation_{cat}"
        before_vals.append(float(work[b_col].mean()) if b_col in work.columns else 0.0)
        after_vals.append(float(work[a_col].mean()) if a_col in work.columns else 0.0)

    x = np.arange(len(categories))
    width = 0.38

    plt.figure(figsize=(max(10, 1.2 * len(categories)), 4.8))
    plt.bar(x - width / 2, before_vals, width=width, label="before")
    plt.bar(x + width / 2, after_vals, width=width, label="after")
    plt.xticks(x, categories, rotation=25, ha="right")
    plt.ylabel("Violation Count")
    plt.title(f"Scheduling Violations Per Category ({phase})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _instance_shape(problem: TSNScheduling, instance_path: str) -> Tuple[int, int]:
    usecase = problem.import_scheduling_usecase(instance_path)
    with problem.use_instance(usecase):
        return int(problem.n_streams), int(problem.n_ports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on instances and evaluate with V1.1 candidate-index TSN defragmentation env.")
    parser.add_argument("--config", default="uc_config.cfg")
    parser.add_argument("--train-instances", default="instance1.json")
    parser.add_argument("--test-instance", default="instance1.json")
    parser.add_argument("--timesteps-per-train-instance", type=int, default=10_000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-sizes", default="10,64,2500")
    parser.add_argument("--mix-percentages", default="0,0,100")
    parser.add_argument("--incoming-stream-count", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--w-gap", type=float, default=1.0)
    parser.add_argument("--w-hole", type=float, default=0.5)
    parser.add_argument("--w-move", type=float, default=0.01)
    parser.add_argument("--w-invalid", type=float, default=1.0)
    parser.add_argument("--reward-ma-window", type=int, default=100)
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

    reward_weights = {
        "w_gap": float(args.w_gap),
        "w_hole": float(args.w_hole),
        "w_move": float(args.w_move),
        "w_invalid": float(args.w_invalid),
    }

    problem = TSNScheduling(args.config)

    shape_ref = _instance_shape(problem, train_instances[0])
    all_instances = list(train_instances) + [test_instance]
    for path in all_instances[1:]:
        shp = _instance_shape(problem, path)
        if shp != shape_ref:
            raise ValueError(
                "All train/eval instances must have matching (n_streams, n_ports) for one PPO policy. "
                f"Reference={shape_ref}, got {shp} for instance '{path}'."
            )

    train_trace: List[Dict[str, float]] = []
    eval_trace: List[Dict[str, float]] = []
    metric_rows: List[Dict[str, float]] = []

    model: PPO | None = None
    for idx, train_path in enumerate(train_instances):
        env_fn = make_env(
            problem,
            train_path,
            reward_weights,
            seed=args.seed + idx,
            reward_frame_sizes=frame_sizes,
            reward_mix_percentages=mix_percentages,
            reward_incoming_stream_count=int(args.incoming_stream_count),
            max_steps=int(args.max_steps),
            max_candidates=int(args.max_candidates),
        )
        vec_env = DummyVecEnv([env_fn])

        if model is None:
            # Standard PPO starting values for small custom environments.
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                n_steps=256,
                batch_size=64,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                clip_range=0.2,
                verbose=1,
                seed=args.seed,
            )
        else:
            model.set_env(vec_env)

        cb = RewardTraceCallback(train_trace, phase="train", instance_name=train_path)
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
        train_trace.extend(train_eval_rewards)

    if model is None:
        raise RuntimeError("Training did not start.")

    test_env = DummyVecEnv(
        [
            make_env(
                problem,
                test_instance,
                reward_weights,
                seed=args.seed + 999,
                reward_frame_sizes=frame_sizes,
                reward_mix_percentages=mix_percentages,
                reward_incoming_stream_count=int(args.incoming_stream_count),
                max_steps=int(args.max_steps),
                max_candidates=int(args.max_candidates),
            )
        ]
    )
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
    eval_trace.extend(eval_rewards)

    reward_df = pd.concat([pd.DataFrame(train_trace), pd.DataFrame(eval_trace)], ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)

    reward_csv = out_dir / "drl_reward_trace.csv"
    metrics_csv = out_dir / "drl_metrics_summary.csv"
    train_reward_png = out_dir / "drl_train_reward_over_steps.png"
    train_reward_ma_png = out_dir / "drl_train_reward_ma_over_steps.png"
    train_final_viol_png = out_dir / "drl_train_final_violations_over_steps.png"
    train_attempted_viol_png = out_dir / "drl_train_attempted_violations_over_steps.png"
    train_rollback_png = out_dir / "drl_train_rollback_over_steps.png"
    train_delta_gap_png = out_dir / "drl_train_delta_gap_over_steps.png"
    train_delta_hole_png = out_dir / "drl_train_delta_hole_over_steps.png"
    train_admission_png = out_dir / "drl_train_admission_over_steps.png"
    eval_reward_png = out_dir / "drl_eval_reward_over_steps.png"
    eval_reward_ma_png = out_dir / "drl_eval_reward_ma_over_steps.png"
    eval_final_viol_png = out_dir / "drl_eval_final_violations_over_steps.png"
    eval_attempted_viol_png = out_dir / "drl_eval_attempted_violations_over_steps.png"
    eval_rollback_png = out_dir / "drl_eval_rollback_over_steps.png"
    eval_delta_gap_png = out_dir / "drl_eval_delta_gap_over_steps.png"
    eval_delta_hole_png = out_dir / "drl_eval_delta_hole_over_steps.png"
    eval_admission_png = out_dir / "drl_eval_admission_over_steps.png"
    train_violation_category_png = out_dir / "drl_train_violation_categories_before_after.png"
    eval_violation_category_png = out_dir / "drl_eval_violation_categories_before_after.png"
    train_attempted_violation_category_png = out_dir / "drl_train_attempted_violation_categories.png"
    eval_attempted_violation_category_png = out_dir / "drl_eval_attempted_violation_categories.png"
    metric_png = out_dir / "drl_metric_before_after.png"
    model_path = out_dir / "ppo_tsn_defrag_model.zip"

    reward_df.to_csv(reward_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    plot_metric_per_instance_and_avg(
        reward_df,
        phase="train",
        metric_col="reward",
        y_label="Reward",
        title="Train Reward Over Steps (Per Instance + Average)",
        out_path=train_reward_png,
    )
    plot_moving_average_metric_per_instance_and_avg(
        reward_df,
        phase="train",
        metric_col="reward",
        y_label="Reward Moving Average",
        title="Train Reward Moving Average Over Steps (Per Instance + Average)",
        out_path=train_reward_ma_png,
        window=int(args.reward_ma_window),
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="train",
        metric_col="final_violation_sum",
        y_label="Final/Post-Rollback Violation Sum",
        title="Train Final/Post-Rollback Violations Over Steps (Per Instance + Average)",
        out_path=train_final_viol_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="train",
        metric_col="attempted_violation_sum",
        y_label="Attempted Violation Sum",
        title="Train Attempted Violations Over Steps (Per Instance + Average)",
        out_path=train_attempted_viol_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="train",
        metric_col="rollback_applied",
        y_label="Rollback Applied (0/1)",
        title="Train Rollback Events Over Steps (Per Instance + Average)",
        out_path=train_rollback_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="train",
        metric_col="delta_gap",
        y_label="Delta Gap",
        title="Train Delta Gap Over Steps (Per Instance + Average)",
        out_path=train_delta_gap_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="train",
        metric_col="delta_hole",
        y_label="Delta Hole",
        title="Train Delta Hole Over Steps (Per Instance + Average)",
        out_path=train_delta_hole_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="train",
        metric_col="admission_rate",
        y_label="Admission Rate",
        title="Train Admission Rate Over Steps (Per Instance + Average)",
        out_path=train_admission_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="eval",
        metric_col="reward",
        y_label="Reward",
        title="Eval Reward Over Steps (Per Instance + Average)",
        out_path=eval_reward_png,
    )
    plot_moving_average_metric_per_instance_and_avg(
        reward_df,
        phase="eval",
        metric_col="reward",
        y_label="Reward Moving Average",
        title="Eval Reward Moving Average Over Steps (Per Instance + Average)",
        out_path=eval_reward_ma_png,
        window=int(args.reward_ma_window),
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="eval",
        metric_col="final_violation_sum",
        y_label="Final/Post-Rollback Violation Sum",
        title="Eval Final/Post-Rollback Violations Over Steps (Per Instance + Average)",
        out_path=eval_final_viol_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="eval",
        metric_col="attempted_violation_sum",
        y_label="Attempted Violation Sum",
        title="Eval Attempted Violations Over Steps (Per Instance + Average)",
        out_path=eval_attempted_viol_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="eval",
        metric_col="rollback_applied",
        y_label="Rollback Applied (0/1)",
        title="Eval Rollback Events Over Steps (Per Instance + Average)",
        out_path=eval_rollback_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="eval",
        metric_col="delta_gap",
        y_label="Delta Gap",
        title="Eval Delta Gap Over Steps (Per Instance + Average)",
        out_path=eval_delta_gap_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="eval",
        metric_col="delta_hole",
        y_label="Delta Hole",
        title="Eval Delta Hole Over Steps (Per Instance + Average)",
        out_path=eval_delta_hole_png,
    )
    plot_metric_per_instance_and_avg(
        reward_df,
        phase="eval",
        metric_col="admission_rate",
        y_label="Admission Rate",
        title="Eval Admission Rate Over Steps (Per Instance + Average)",
        out_path=eval_admission_png,
    )
    plot_violation_categories_before_after(
        metrics_df,
        phase="train",
        out_path=train_violation_category_png,
    )
    plot_violation_categories_before_after(
        metrics_df,
        phase="eval",
        out_path=eval_violation_category_png,
    )
    plot_attempted_violation_category_means(
        reward_df,
        phase="train",
        out_path=train_attempted_violation_category_png,
    )
    plot_attempted_violation_category_means(
        reward_df,
        phase="eval",
        out_path=eval_attempted_violation_category_png,
    )
    plot_metric_comparison(metrics_df, metric_png)
    model.save(model_path)

    print(f"Saved reward trace: {reward_csv}")
    print(f"Saved metrics summary: {metrics_csv}")
    print(f"Saved train reward plot: {train_reward_png}")
    print(f"Saved train reward moving-average plot: {train_reward_ma_png}")
    print(f"Saved train final-violations plot: {train_final_viol_png}")
    print(f"Saved train attempted-violations plot: {train_attempted_viol_png}")
    print(f"Saved train rollback plot: {train_rollback_png}")
    print(f"Saved train delta-gap plot: {train_delta_gap_png}")
    print(f"Saved train delta-hole plot: {train_delta_hole_png}")
    print(f"Saved train admission plot: {train_admission_png}")
    print(f"Saved train attempted-violation-category plot: {train_attempted_violation_category_png}")
    print(f"Saved eval reward plot: {eval_reward_png}")
    print(f"Saved eval reward moving-average plot: {eval_reward_ma_png}")
    print(f"Saved eval final-violations plot: {eval_final_viol_png}")
    print(f"Saved eval attempted-violations plot: {eval_attempted_viol_png}")
    print(f"Saved eval rollback plot: {eval_rollback_png}")
    print(f"Saved eval delta-gap plot: {eval_delta_gap_png}")
    print(f"Saved eval delta-hole plot: {eval_delta_hole_png}")
    print(f"Saved eval admission plot: {eval_admission_png}")
    print(f"Saved eval attempted-violation-category plot: {eval_attempted_violation_category_png}")
    print(f"Saved train violation-category plot: {train_violation_category_png}")
    print(f"Saved eval violation-category plot: {eval_violation_category_png}")
    print(f"Saved metric plot: {metric_png}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
