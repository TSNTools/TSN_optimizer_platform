import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-csv", required=True)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--out-prefix", default="training")
    args = parser.parse_args()

    df = pd.read_csv(args.monitor_csv, comment="#")
    if "episode_invalid_moves" not in df.columns:
        raise ValueError("episode_invalid_moves not found in monitor.csv")

    # Invalid moves rolling mean
    roll_invalid = df["episode_invalid_moves"].rolling(args.window, min_periods=1).mean()

    plt.figure(figsize=(10, 4.5))
    plt.plot(df.index, df["episode_invalid_moves"], alpha=0.25, label="episode invalid moves")
    plt.plot(df.index, roll_invalid, label=f"rolling mean ({args.window})")
    plt.title("Invalid moves per episode")
    plt.xlabel("Episode")
    plt.ylabel("Invalid moves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_invalid_moves.png", dpi=160)
    plt.close()

    # Violation sum rolling mean, if available
    if "episode_violation_sum" in df.columns:
        roll_viol = df["episode_violation_sum"].rolling(args.window, min_periods=1).mean()

        plt.figure(figsize=(10, 4.5))
        plt.plot(df.index, df["episode_violation_sum"], alpha=0.25, label="episode violation sum")
        plt.plot(df.index, roll_viol, label=f"rolling mean ({args.window})")
        plt.title("Violation sum per episode")
        plt.xlabel("Episode")
        plt.ylabel("Violation sum")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_violation_sum.png", dpi=160)
        plt.close()

    print("Saved violation plots.")


if __name__ == "__main__":
    main()