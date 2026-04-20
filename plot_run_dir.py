from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_monitor_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"monitor.csv not found: {path}")
    df = pd.read_csv(path, comment="#")
    if len(df) == 0:
        raise ValueError(f"monitor.csv is empty: {path}")
    return df.reset_index(drop=True)


def plot_reward(df: pd.DataFrame, out_dir: Path, window: int) -> None:
    if "r" not in df.columns:
        return

    tmp = df.copy()
    tmp["episode"] = np.arange(1, len(tmp) + 1)
    tmp["rolling_mean"] = tmp["r"].rolling(window, min_periods=1).mean()
    tmp["rolling_std"] = tmp["r"].rolling(window, min_periods=1).std().fillna(0.0)
    tmp["upper"] = tmp["rolling_mean"] + tmp["rolling_std"]
    tmp["lower"] = tmp["rolling_mean"] - tmp["rolling_std"]

    plt.figure(figsize=(10, 5))
    plt.plot(tmp["episode"], tmp["rolling_mean"], label="rolling mean reward")
    plt.fill_between(tmp["episode"], tmp["lower"], tmp["upper"], alpha=0.25, label="rolling std band")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_band.png", dpi=180)
    plt.close()

def plot_realloc_and_cycles(df: pd.DataFrame, out_dir: Path, window: int) -> None:
    if "episode_reallocations" in df.columns:
        roll = df["episode_reallocations"].rolling(window, min_periods=1).mean()

        plt.figure(figsize=(10, 4.5))
        plt.plot(df.index + 1, df["episode_reallocations"], alpha=0.25, label="episode reallocations")
        plt.plot(df.index + 1, roll, label=f"rolling mean ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Reallocations")
        plt.title("Reallocations per episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "reallocations.png", dpi=180)
        plt.close()

    if "episode_defrag_cycles" in df.columns:
        roll = df["episode_defrag_cycles"].rolling(window, min_periods=1).mean()

        plt.figure(figsize=(10, 4.5))
        plt.plot(df.index + 1, df["episode_defrag_cycles"], alpha=0.25, label="episode defrag cycles")
        plt.plot(df.index + 1, roll, label=f"rolling mean ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Defrag cycles")
        plt.title("Defrag cycles per episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "defrag_cycles.png", dpi=180)
        plt.close()

def plot_episode_sbr(df: pd.DataFrame, out_dir: Path, window: int) -> None:
    if "episode_sbr" not in df.columns:
        return

    tmp = df.copy()
    tmp["episode"] = np.arange(1, len(tmp) + 1)
    tmp["rolling_mean"] = tmp["episode_sbr"].rolling(window, min_periods=1).mean()

    plt.figure(figsize=(10, 4.5))
    plt.plot(tmp["episode"], tmp["episode_sbr"], alpha=0.25, label="episode SBR")
    plt.plot(tmp["episode"], tmp["rolling_mean"], label=f"rolling mean ({window})")
    plt.xlabel("Episode")
    plt.ylabel("SBR")
    plt.title("Service Blocking Ratio per episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "episode_sbr.png", dpi=180)
    plt.close()


def plot_invalid_and_violations(df: pd.DataFrame, out_dir: Path, window: int) -> None:
    if "episode_invalid_moves" in df.columns:
        roll = df["episode_invalid_moves"].rolling(window, min_periods=1).mean()

        plt.figure(figsize=(10, 4.5))
        plt.plot(df.index + 1, df["episode_invalid_moves"], alpha=0.25, label="episode invalid moves")
        plt.plot(df.index + 1, roll, label=f"rolling mean ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Invalid moves")
        plt.title("Invalid moves per episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "invalid_moves.png", dpi=180)
        plt.close()

    if "episode_violation_sum" in df.columns:
        roll = df["episode_violation_sum"].rolling(window, min_periods=1).mean()

        plt.figure(figsize=(10, 4.5))
        plt.plot(df.index + 1, df["episode_violation_sum"], alpha=0.25, label="episode violation sum")
        plt.plot(df.index + 1, roll, label=f"rolling mean ({window})")
        plt.xlabel("Episode")
        plt.ylabel("Violation sum")
        plt.title("Violation sum per episode")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "violation_sum.png", dpi=180)
        plt.close()


def plot_episode_length(df: pd.DataFrame, out_dir: Path, window: int) -> None:
    if "l" not in df.columns:
        return

    tmp = df.copy()
    tmp["episode"] = np.arange(1, len(tmp) + 1)
    tmp["rolling_mean"] = tmp["l"].rolling(window, min_periods=1).mean()

    plt.figure(figsize=(10, 4.5))
    plt.plot(tmp["episode"], tmp["l"], alpha=0.25, label="episode length")
    plt.plot(tmp["episode"], tmp["rolling_mean"], label=f"rolling mean ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Length")
    plt.title("Episode length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "episode_length.png", dpi=180)
    plt.close()


def plot_before_after(summary: dict, out_dir: Path) -> None:
    ep = summary["episode_results"][0]
    before = ep["before"]
    after = ep["after"]
    delta = ep["delta"]

    metric_names = [
        "largest_hole_ns",
        "total_useless_gaps",
        "total_gap_count",
        "mean_utilization",
        "violation_sum",
    ]
    pretty = {
        "largest_hole_ns": "Largest hole (ns)",
        "total_useless_gaps": "Useless gaps",
        "total_gap_count": "Total gaps",
        "mean_utilization": "Mean utilization",
        "violation_sum": "Violations",
    }

    before_vals = [before[k] for k in metric_names]
    after_vals = [after[k] for k in metric_names]
    x = np.arange(len(metric_names))
    w = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, before_vals, width=w, label="Before")
    plt.bar(x + w / 2, after_vals, width=w, label="After")
    plt.xticks(x, [pretty[k] for k in metric_names], rotation=20, ha="right")
    plt.title("Before vs after metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "before_after_metrics.png", dpi=180)
    plt.close()

    delta_vals = [delta[k] for k in metric_names]
    plt.figure(figsize=(9, 4.5))
    plt.bar(np.arange(len(metric_names)), delta_vals)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(np.arange(len(metric_names)), [pretty[k] for k in metric_names], rotation=20, ha="right")
    plt.title("Metric deltas (after - before)")
    plt.tight_layout()
    plt.savefig(out_dir / "metric_deltas.png", dpi=180)
    plt.close()

    reward_trace = ep.get("reward_trace_first_20", [])
    hole_trace = ep.get("largest_hole_trace_first_20", [])
    useless_trace = ep.get("useless_gap_trace_first_20", [])

    if reward_trace:
        plt.figure(figsize=(9, 4))
        plt.plot(reward_trace)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("Reward trace (first 20 steps)")
        plt.tight_layout()
        plt.savefig(out_dir / "reward_trace_first20.png", dpi=180)
        plt.close()

    if hole_trace:
        plt.figure(figsize=(9, 4))
        plt.plot(hole_trace)
        plt.xlabel("Step")
        plt.ylabel("Largest hole (ns)")
        plt.title("Largest hole trace (first 20 steps)")
        plt.tight_layout()
        plt.savefig(out_dir / "largest_hole_trace_first20.png", dpi=180)
        plt.close()

    if useless_trace:
        plt.figure(figsize=(9, 4))
        plt.plot(useless_trace)
        plt.xlabel("Step")
        plt.ylabel("Useless gap count")
        plt.title("Useless gap trace (first 20 steps)")
        plt.tight_layout()
        plt.savefig(out_dir / "useless_gap_trace_first20.png", dpi=180)
        plt.close()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_bytes(src.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all training/evaluation plots for one run directory.")
    parser.add_argument("--run-dir", required=True, help="Training run directory, e.g. test_run_1m")
    parser.add_argument("--window", type=int, default=100, help="Rolling window for training curves")
    parser.add_argument("--eval-subdir", default=None, help="Override evaluation folder name, e.g. eval_best")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor_path = run_dir / "monitor.csv"
    if monitor_path.exists():
        df = load_monitor_csv(monitor_path)
        plot_reward(df, out_dir, args.window)
        plot_episode_sbr(df, out_dir, args.window)
        plot_invalid_and_violations(df, out_dir, args.window)
        plot_episode_length(df, out_dir, args.window)
        plot_realloc_and_cycles(df, out_dir, args.window)

    eval_dir = None
    if args.eval_subdir:
        cand = run_dir / args.eval_subdir
        if cand.exists():
            eval_dir = cand
    else:
        for name in ("eval_best", "eval_final"):
            cand = run_dir / name
            if cand.exists():
                eval_dir = cand
                break

    if eval_dir is not None:
        summary_path = eval_dir / "evaluation_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            plot_before_after(summary, out_dir)

        copy_if_exists(eval_dir / "before_schedule.png", out_dir / "before_schedule.png")
        copy_if_exists(eval_dir / "after_schedule.png", out_dir / "after_schedule.png")

    print(f"Saved plots in: {out_dir}")


if __name__ == "__main__":
    main()
