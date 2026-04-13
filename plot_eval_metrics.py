import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to evaluation_summary.json")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ep = data["episode_results"][0]
    before = ep["before"]
    after = ep["after"]
    delta = ep["delta"]

    out_dir = summary_path.parent

    # -----------------------------
    # 1) before/after grouped bars
    # -----------------------------
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
    plt.title("Before vs After metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "before_after_metrics.png", dpi=160)
    plt.close()

    # -----------------------------
    # 2) delta bars
    # -----------------------------
    delta_names = [
        "largest_hole_ns",
        "total_useless_gaps",
        "total_gap_count",
        "mean_utilization",
        "violation_sum",
    ]
    delta_vals = [delta[k] for k in delta_names]

    plt.figure(figsize=(9, 4.5))
    bars = plt.bar(np.arange(len(delta_names)), delta_vals)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(np.arange(len(delta_names)), [pretty[k] for k in delta_names], rotation=20, ha="right")
    plt.title("Metric deltas (After - Before)")
    plt.tight_layout()
    plt.savefig(out_dir / "metric_deltas.png", dpi=160)
    plt.close()

    # -----------------------------
    # 3) traces from evaluated episode
    # -----------------------------
    reward_trace = ep.get("reward_trace_first_20", [])
    hole_trace = ep.get("largest_hole_trace_first_20", [])
    useless_trace = ep.get("useless_gap_trace_first_20", [])

    if reward_trace:
        plt.figure(figsize=(9, 4))
        plt.plot(reward_trace)
        plt.title("Reward trace (first 20 steps)")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(out_dir / "reward_trace_first20.png", dpi=160)
        plt.close()

    if hole_trace:
        plt.figure(figsize=(9, 4))
        plt.plot(hole_trace)
        plt.title("Largest hole trace (first 20 steps)")
        plt.xlabel("Step")
        plt.ylabel("Largest hole (ns)")
        plt.tight_layout()
        plt.savefig(out_dir / "largest_hole_trace_first20.png", dpi=160)
        plt.close()

    if useless_trace:
        plt.figure(figsize=(9, 4))
        plt.plot(useless_trace)
        plt.title("Useless gap trace (first 20 steps)")
        plt.xlabel("Step")
        plt.ylabel("Useless gap count")
        plt.tight_layout()
        plt.savefig(out_dir / "useless_gap_trace_first20.png", dpi=160)
        plt.close()

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()