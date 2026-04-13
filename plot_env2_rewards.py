from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-csv", required=True)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.monitor_csv, comment="#")
    if "r" not in df.columns:
        raise ValueError("Could not find reward column 'r' in monitor csv.")

    df = df.reset_index(drop=True)
    df["episode"] = df.index + 1
    df["rolling_mean"] = df["r"].rolling(args.window, min_periods=1).mean()
    df["rolling_std"] = df["r"].rolling(args.window, min_periods=1).std().fillna(0.0)
    df["upper"] = df["rolling_mean"] + df["rolling_std"]
    df["lower"] = df["rolling_mean"] - df["rolling_std"]

    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["rolling_mean"], label="rolling mean reward")
    plt.fill_between(df["episode"], df["lower"], df["upper"], alpha=0.25, label="rolling std band")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode reward with rolling mean and band")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
