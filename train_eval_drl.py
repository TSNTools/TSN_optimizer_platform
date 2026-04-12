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
                "reward_mode": info0.get("reward_mode", "unknown"),
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
    seed: int,
    env_kwargs: Dict[str, object],
):
    def _build():
        usecase = problem.import_scheduling_usecase(instance_path)
        env = TSNDefragEnv(
            problem=problem,
            usecase=usecase,
            **env_kwargs,
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
                "reward_mode": info0.get("reward_mode", "unknown"),
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


def plot_train_online_reward_trend_with_envelope(
    df: pd.DataFrame,
    out_path: Path,
    window: int,
) -> None:
    """Primary training plot: rolling mean with min/max envelope by global_step."""
    if df.empty:
        return

    work = df[df["phase"] == "train_online"].copy()
    if work.empty or "reward" not in work.columns or "global_step" not in work.columns:
        return

    work = work.dropna(subset=["instance", "reward", "global_step"])
    if work.empty:
        return

    work["global_step"] = work["global_step"].astype(int)
    work = work.sort_values(["instance", "global_step"])

    w = max(1, int(window))
    work["reward_roll_mean"] = work.groupby("instance")["reward"].transform(
        lambda s: s.rolling(window=w, min_periods=1).mean()
    )
    work["reward_roll_min"] = work.groupby("instance")["reward"].transform(
        lambda s: s.rolling(window=w, min_periods=1).min()
    )
    work["reward_roll_max"] = work.groupby("instance")["reward"].transform(
        lambda s: s.rolling(window=w, min_periods=1).max()
    )

    instances = sorted(work["instance"].dropna().unique().tolist())
    if not instances:
        return

    fig, axes = plt.subplots(len(instances) + 1, 1, figsize=(10, 2.8 * (len(instances) + 1)), sharex=False)
    if len(instances) + 1 == 1:
        axes = [axes]

    for i, instance_name in enumerate(instances):
        grp = work[work["instance"] == instance_name]
        x = grp["global_step"].to_numpy(dtype=int)
        y_mean = grp["reward_roll_mean"].to_numpy(dtype=float)
        y_min = grp["reward_roll_min"].to_numpy(dtype=float)
        y_max = grp["reward_roll_max"].to_numpy(dtype=float)

        axes[i].plot(x, y_mean, linewidth=1.8, label="rolling mean")
        axes[i].fill_between(x, y_min, y_max, alpha=0.22, label="rolling min/max")
        axes[i].set_title(str(instance_name))
        axes[i].set_xlabel("Global Step")
        axes[i].set_ylabel("Reward")
        axes[i].legend(loc="best")

    by_step = work.groupby("global_step")["reward"].mean().sort_index()
    overall = pd.DataFrame({"reward": by_step})
    overall["reward_roll_mean"] = overall["reward"].rolling(window=w, min_periods=1).mean()
    overall["reward_roll_min"] = overall["reward"].rolling(window=w, min_periods=1).min()
    overall["reward_roll_max"] = overall["reward"].rolling(window=w, min_periods=1).max()

    x_o = overall.index.to_numpy(dtype=int)
    y_o_mean = overall["reward_roll_mean"].to_numpy(dtype=float)
    y_o_min = overall["reward_roll_min"].to_numpy(dtype=float)
    y_o_max = overall["reward_roll_max"].to_numpy(dtype=float)

    axes[-1].plot(x_o, y_o_mean, color="black", linewidth=2.0, label="overall rolling mean")
    axes[-1].fill_between(x_o, y_o_min, y_o_max, color="gray", alpha=0.25, label="overall rolling min/max")
    axes[-1].set_title("Overall Average Across Instances")
    axes[-1].set_xlabel("Global Step")
    axes[-1].set_ylabel("Reward")
    axes[-1].legend(loc="best")

    fig.suptitle("Train Online Reward Trend (Rolling Mean + Min/Max Envelope)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def reward_sign_summary(df: pd.DataFrame, phase: str) -> Dict[str, float]:
    """Return fraction of positive/zero/negative rewards for one phase."""
    work = df[df["phase"] == phase].copy()
    if work.empty or "reward" not in work.columns:
        return {
            "phase": phase,
            "n": 0,
            "positive_fraction": np.nan,
            "zero_fraction": np.nan,
            "negative_fraction": np.nan,
        }

    rewards = work["reward"].dropna().to_numpy(dtype=float)
    n = int(rewards.size)
    if n == 0:
        return {
            "phase": phase,
            "n": 0,
            "positive_fraction": np.nan,
            "zero_fraction": np.nan,
            "negative_fraction": np.nan,
        }

    pos = float(np.mean(rewards > 0))
    zer = float(np.mean(np.isclose(rewards, 0.0)))
    neg = float(np.mean(rewards < 0))
    return {
        "phase": phase,
        "n": n,
        "positive_fraction": pos,
        "zero_fraction": zer,
        "negative_fraction": neg,
    }


def plot_reward_sign_summary(summary: Dict[str, float], out_path: Path) -> None:
    labels = ["positive", "zero", "negative"]
    values = [
        float(summary.get("positive_fraction", 0.0)),
        float(summary.get("zero_fraction", 0.0)),
        float(summary.get("negative_fraction", 0.0)),
    ]
    x = np.arange(len(labels))
    plt.figure(figsize=(5.8, 4.0))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Fraction")
    plt.title(f"Reward Sign Summary ({summary.get('phase', 'unknown')})")
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


def create_numbered_run_dir(base_out_dir: Path) -> Path:
    """Create outputs/run_XXX folder and return it."""
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


def _instance_shape(problem: TSNScheduling, instance_path: str) -> Tuple[int, int]:
    usecase = problem.import_scheduling_usecase(instance_path)
    with problem.use_instance(usecase):
        return int(problem.n_streams), int(problem.n_ports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on instances and evaluate with V1.1 candidate-index TSN defragmentation env.")
    parser.add_argument("--config", default="uc_config.cfg")
    parser.add_argument("--train-instances", default="instance2.json")
    parser.add_argument("--test-instance", default="instance2.json")
    parser.add_argument("--timesteps-per-train-instance", type=int, default=20000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-sizes", default="10,64,1500")
    parser.add_argument("--mix-percentages", default="0,0,100")
    parser.add_argument("--incoming-stream-count", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument(
        "--reward-mode",
        default="hole_improvement_binary",
        choices=["weighted_fragmentation", "gap_improvement_binary", "hole_improvement_binary"],
    )
    parser.add_argument("--w-gap", type=float, default=1.0)
    parser.add_argument("--w-hole", type=float, default=0.5)
    parser.add_argument("--w-move", type=float, default=0.01)
    parser.add_argument("--w-invalid", type=float, default=1.0)
    parser.add_argument("--force-current-start-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-candidate-validation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reward-ma-window", type=int, default=100)
    parser.add_argument("--single-frame-size", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    np.random.seed(args.seed)

    train_instances = parse_list(args.train_instances, str)
    test_instance = args.test_instance
    frame_sizes = parse_list(args.frame_sizes, int)
    mix_percentages = parse_list(args.mix_percentages, float)

    if args.single_frame_size is not None:
        frame_sizes = [int(args.single_frame_size)]
        mix_percentages = [100.0]

    if len(frame_sizes) != len(mix_percentages):
        raise ValueError("frame_sizes and mix_percentages must have the same length")

    out_dir = create_numbered_run_dir(Path(args.out_dir))

    reward_weights = {
        "w_gap": float(args.w_gap),
        "w_hole": float(args.w_hole),
        "w_move": float(args.w_move),
        "w_invalid": float(args.w_invalid),
    }

    env_kwargs: Dict[str, object] = {
        "single_move_per_step": True,
        "init_from_usecase_schedule": True,
        "reward_mode": str(args.reward_mode),
        "reward_weights": reward_weights,
        "reward_frame_sizes": list(frame_sizes),
        "reward_mix_percentages": list(mix_percentages),
        "reward_incoming_stream_count": int(args.incoming_stream_count),
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

    incoming_stream_count = int(args.incoming_stream_count)
    reward_mode = str(args.reward_mode)

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
    train_rollout_trace: List[Dict[str, float]] = []
    eval_rollout_trace: List[Dict[str, float]] = []
    metric_rows: List[Dict[str, float]] = []

    model: PPO | None = None
    for idx, train_path in enumerate(train_instances):
        env_fn = make_env(
            problem,
            train_path,
            seed=args.seed + idx,
            env_kwargs=env_kwargs,
        )
        vec_env = DummyVecEnv([env_fn])

        if model is None:
            model = PPO("MultiInputPolicy", vec_env, **ppo_kwargs)
        else:
            model.set_env(vec_env)

        cb = RewardTraceCallback(train_trace, phase="train_online", instance_name=train_path)
        model.learn(total_timesteps=int(args.timesteps_per_train_instance), reset_num_timesteps=False, callback=cb)

        train_metric_row, train_eval_rewards = rollout_and_collect(
            model=model,
            vec_env=vec_env,
            eval_steps=int(args.eval_steps),
            frame_sizes=frame_sizes,
            mix_percentages=mix_percentages,
            incoming_stream_count=int(incoming_stream_count),
            phase="train_rollout",
            instance_name=train_path,
        )
        metric_rows.append(train_metric_row)
        train_rollout_trace.extend(train_eval_rewards)

    if model is None:
        raise RuntimeError("Training did not start.")

    test_env = DummyVecEnv(
        [
            make_env(
                problem,
                test_instance,
                seed=args.seed + 999,
                env_kwargs=env_kwargs,
            )
        ]
    )
    eval_metric_row, eval_rewards = rollout_and_collect(
        model=model,
        vec_env=test_env,
        eval_steps=int(args.eval_steps),
        frame_sizes=frame_sizes,
        mix_percentages=mix_percentages,
        incoming_stream_count=int(incoming_stream_count),
        phase="eval_rollout",
        instance_name=test_instance,
    )
    metric_rows.append(eval_metric_row)
    eval_rollout_trace.extend(eval_rewards)

    train_online_df = pd.DataFrame(train_trace)
    rollout_df = pd.concat([pd.DataFrame(train_rollout_trace), pd.DataFrame(eval_rollout_trace)], ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)

    reward_csv = out_dir / "drl_reward_trace.csv"
    rollout_reward_csv = out_dir / "drl_rollout_trace_debug.csv"
    metrics_csv = out_dir / "drl_metrics_summary.csv"
    train_online_reward_trend_envelope_png = out_dir / "drl_train_online_reward_trend_envelope.png"
    debug_train_online_reward_raw_png = out_dir / "debug_train_online_reward_raw.png"
    train_delta_gap_png = out_dir / "drl_train_delta_gap_over_steps.png"
    train_delta_hole_png = out_dir / "drl_train_delta_hole_over_steps.png"
    train_admission_png = out_dir / "drl_train_admission_over_steps.png"
    eval_delta_gap_png = out_dir / "drl_eval_delta_gap_over_steps.png"
    eval_delta_hole_png = out_dir / "drl_eval_delta_hole_over_steps.png"
    eval_admission_png = out_dir / "drl_eval_admission_over_steps.png"
    reward_sign_summary_csv = out_dir / "drl_train_online_reward_sign_summary.csv"
    train_reward_sign_png = out_dir / "drl_train_online_reward_sign_summary.png"
    metric_png = out_dir / "drl_metric_before_after.png"
    model_path = out_dir / "ppo_tsn_defrag_model.zip"

    train_online_df.to_csv(reward_csv, index=False)
    rollout_df.to_csv(rollout_reward_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    # Primary training visualization: online callback reward only.
    plot_train_online_reward_trend_with_envelope(
        train_online_df,
        out_path=train_online_reward_trend_envelope_png,
        window=int(args.reward_ma_window),
    )

    # Optional debug-only raw online reward plot.
    plot_metric_per_instance_and_avg(
        train_online_df,
        phase="train_online",
        metric_col="reward",
        y_label="Reward",
        title="Debug Train Online Reward Over Steps (Per Instance + Average)",
        out_path=debug_train_online_reward_raw_png,
    )

    plot_metric_per_instance_and_avg(
        rollout_df,
        phase="train_rollout",
        metric_col="delta_gap",
        y_label="Delta Gap",
        title="Train Delta Gap Over Steps (Per Instance + Average)",
        out_path=train_delta_gap_png,
    )
    plot_metric_per_instance_and_avg(
        rollout_df,
        phase="train_rollout",
        metric_col="delta_hole",
        y_label="Delta Hole",
        title="Train Delta Hole Over Steps (Per Instance + Average)",
        out_path=train_delta_hole_png,
    )
    plot_metric_per_instance_and_avg(
        rollout_df,
        phase="train_rollout",
        metric_col="admission_rate",
        y_label="Admission Rate",
        title="Train Admission Rate Over Steps (Per Instance + Average)",
        out_path=train_admission_png,
    )

    plot_metric_per_instance_and_avg(
        rollout_df,
        phase="eval_rollout",
        metric_col="delta_gap",
        y_label="Delta Gap",
        title="Eval Delta Gap Over Steps (Per Instance + Average)",
        out_path=eval_delta_gap_png,
    )
    plot_metric_per_instance_and_avg(
        rollout_df,
        phase="eval_rollout",
        metric_col="delta_hole",
        y_label="Delta Hole",
        title="Eval Delta Hole Over Steps (Per Instance + Average)",
        out_path=eval_delta_hole_png,
    )
    plot_metric_per_instance_and_avg(
        rollout_df,
        phase="eval_rollout",
        metric_col="admission_rate",
        y_label="Admission Rate",
        title="Eval Admission Rate Over Steps (Per Instance + Average)",
        out_path=eval_admission_png,
    )

    train_sign = reward_sign_summary(train_online_df, phase="train_online")
    sign_df = pd.DataFrame([train_sign])
    sign_df.to_csv(reward_sign_summary_csv, index=False)
    plot_reward_sign_summary(train_sign, train_reward_sign_png)

    plot_metric_comparison(metrics_df, metric_png)
    model.save(model_path)

    print(f"Saved reward trace: {reward_csv}")
    print(f"Saved rollout debug trace: {rollout_reward_csv}")
    print(f"Saved metrics summary: {metrics_csv}")
    print(f"Saved run directory: {out_dir}")
    print(f"Reward mode: {reward_mode}")
    print(f"Saved train-online reward trend plot: {train_online_reward_trend_envelope_png}")
    print(f"Saved debug train-online raw reward plot: {debug_train_online_reward_raw_png}")
    print(f"Saved train delta-gap plot: {train_delta_gap_png}")
    print(f"Saved train delta-hole plot: {train_delta_hole_png}")
    print(f"Saved train admission plot: {train_admission_png}")
    print(f"Saved eval delta-gap plot: {eval_delta_gap_png}")
    print(f"Saved eval delta-hole plot: {eval_delta_hole_png}")
    print(f"Saved eval admission plot: {eval_admission_png}")
    print(f"Saved train-online reward sign summary CSV: {reward_sign_summary_csv}")
    print(f"Saved train-online reward sign plot: {train_reward_sign_png}")
    print(f"Saved metric plot: {metric_png}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
