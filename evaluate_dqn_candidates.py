from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from env_candidates import TSNCandidateHopEnv
from tsn_scheduling_problem import TSNScheduling


def build_env(config: str, instance_path: str, run_args: Dict[str, Any]) -> TSNCandidateHopEnv:
    problem = TSNScheduling(config)
    instance = problem.import_scheduling_usecase(instance_path)

    env = TSNCandidateHopEnv(
        problem=problem,
        usecase=instance,
        useless_gap_threshold_bytes=int(run_args.get("useless_gap_threshold_bytes", 1024)),
        invalid_penalty=float(run_args.get("invalid_penalty", 10000.0)),
        largest_hole_weight=float(run_args.get("largest_hole_weight", 1.0)),
        useless_gap_weight=float(run_args.get("useless_gap_weight", 1.0)),
        acceptance_weight=float(run_args.get("acceptance_weight", 1.0)),
        expected_incoming_bytes=int(run_args.get("expected_incoming_bytes", 1024)),
        max_candidates=int(run_args.get("max_candidates", 10)),
        max_steps=int(run_args.get("max_steps", 60)),
        init_from_usecase_schedule=not bool(run_args.get("no_usecase_schedule", False)),
        include_current_start_candidate=not bool(run_args.get("no_current_start_candidate", False)),
    )
    return env


def schedule_metrics(env: TSNCandidateHopEnv) -> Dict[str, Any]:
    g = env._global_metrics()
    violation_sum = int(env._violation_sum())
    compact = env.get_combined_schedule().to_compact_encoding(env.n_ports, env.n_streams)
    per_stream = env.sched_eval.per_stream_all_violations(compact)
    row_sums = per_stream.sum(axis=1).astype(int).tolist()

    return {
        "violation_sum": violation_sum,
        "largest_hole_ns": float(g["largest_hole_ns"]),
        "total_useless_gaps": float(g["total_useless_gaps"]),
        "total_gap_count": float(g["total_gap_count"]),
        "mean_utilization": float(g["mean_utilization"]),
        "acceptance_count": float(g.get("acceptance_count", 0.0)),
        "violation_row_sums": row_sums,
    }


def save_schedule_plot_from_schedule(
    schedule,
    out_path: str,
    cycles: int = 1,
    label: bool = True,
    by_cycle: bool = False,
) -> None:
    fig, _ = schedule.plot(cycles=cycles, label=label, by_cycle=by_cycle)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_eval_reward_plot(results: List[Dict[str, Any]], out_path: str) -> None:
    if not results:
        return

    rewards = [float(r["episode_reward"]) for r in results]
    xs = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(9, 4.5))
    plt.plot(xs, rewards, marker="o")
    plt.xlabel("Evaluation episode")
    plt.ylabel("Episode reward")
    plt.title("Evaluation episode rewards")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def rollout_episode(
    model: DQN,
    env: TSNCandidateHopEnv,
    deterministic: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    obs, _ = env.reset()
    before_metrics = schedule_metrics(env)

    total_reward = 0.0
    steps = 0
    invalid_moves = 0

    rewards: List[float] = []
    largest_hole_trace: List[float] = [before_metrics["largest_hole_ns"]]
    useless_gap_trace: List[float] = [before_metrics["total_useless_gaps"]]
    acceptance_trace: List[float] = [before_metrics["acceptance_count"]]
    candidate_count_trace: List[int] = [len(env.current_candidates)]

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        rewards.append(float(reward))
        steps += 1

        if bool(info.get("invalid_move", False)):
            invalid_moves += 1

        largest_hole_trace.append(float(info.get("largest_hole_after_ns", largest_hole_trace[-1])))
        useless_gap_trace.append(float(info.get("useless_gaps_after", useless_gap_trace[-1])))
        acceptance_trace.append(float(info.get("acceptance_after", acceptance_trace[-1])))
        candidate_count_trace.append(int(info.get("candidate_count", candidate_count_trace[-1])))

    final_violation_sum = int(env._violation_sum())
    if final_violation_sum != 0:
        raise RuntimeError(
            f"Evaluation ended in an invalid schedule. violation_sum={final_violation_sum}"
        )

    after_metrics = schedule_metrics(env)
    final_schedule = env.get_combined_schedule()

    result = {
        "before": before_metrics,
        "after": after_metrics,
        "episode_reward": float(total_reward),
        "steps": int(steps),
        "invalid_moves": int(invalid_moves),
        "reward_trace_first_20": rewards[:20],
        "largest_hole_trace_first_20": largest_hole_trace[:20],
        "useless_gap_trace_first_20": useless_gap_trace[:20],
        "acceptance_trace_first_20": acceptance_trace[:20],
        "candidate_count_trace_first_20": candidate_count_trace[:20],
        "delta": {
            "largest_hole_ns": float(after_metrics["largest_hole_ns"] - before_metrics["largest_hole_ns"]),
            "total_useless_gaps": float(after_metrics["total_useless_gaps"] - before_metrics["total_useless_gaps"]),
            "total_gap_count": float(after_metrics["total_gap_count"] - before_metrics["total_gap_count"]),
            "mean_utilization": float(after_metrics["mean_utilization"] - before_metrics["mean_utilization"]),
            "acceptance_count": float(after_metrics["acceptance_count"] - before_metrics["acceptance_count"]),
            "violation_sum": int(after_metrics["violation_sum"] - before_metrics["violation_sum"]),
            "acceptance_count": float(after_metrics["acceptance_count"] - before_metrics["acceptance_count"]),
        },
    }
    return result, final_schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained candidate-based TSN DRL model and save before/after schedules + metrics."
    )
    parser.add_argument("--run-dir", required=True, help="Training output folder containing run_args.json and model files.")
    parser.add_argument("--model", choices=["best", "final"], default="best", help="Which model to load from run-dir.")
    parser.add_argument("--config", default=None, help="Override config path. Defaults to run_args.json value.")
    parser.add_argument("--instance", default=None, help="Override instance path. Defaults to run_args.json value.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of deterministic evaluation episodes.")
    parser.add_argument("--label", action="store_true", help="Annotate plots with stream ids.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles to render in schedule plots.")
    parser.add_argument("--by-cycle", action="store_true", help="Wrap plots by cycle.")
    args = parser.parse_args()

    run_args_path = os.path.join(args.run_dir, "run_args.json")
    if not os.path.exists(run_args_path):
        raise FileNotFoundError(f"Could not find {run_args_path}")

    with open(run_args_path, "r", encoding="utf-8") as f:
        run_args = json.load(f)

    config = args.config or run_args["config"]
    instance_path = args.instance or run_args["instance"]

    if args.model == "best":
        candidates = [
            os.path.join(args.run_dir, "best_model.zip"),
            os.path.join(args.run_dir, "best_model"),
        ]
    else:
        candidates = [
            os.path.join(args.run_dir, "final_model.zip"),
            os.path.join(args.run_dir, "final_model"),
        ]

    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(f"Could not find requested model in {args.run_dir}. Tried: {candidates}")

    eval_dir = os.path.join(args.run_dir, f"eval_{args.model}")
    os.makedirs(eval_dir, exist_ok=True)

    env = build_env(config=config, instance_path=instance_path, run_args=run_args)
    model = DQN.load(model_path)

    # Force greedy action selection
    if hasattr(model, "exploration_rate"):
        model.exploration_rate = 0.0

    # Save before schedule once from reset state
    obs, _ = env.reset()
    save_schedule_plot_from_schedule(
        env.get_combined_schedule(),
        os.path.join(eval_dir, "before_schedule.png"),
        cycles=args.cycles,
        label=args.label,
        by_cycle=args.by_cycle,
    )

    results: List[Dict[str, Any]] = []
    best_schedule = None
    best_result = None
    best_reward = -float("inf")

    for ep in range(args.episodes):
        ep_result, final_schedule = rollout_episode(model, env, deterministic=True)
        results.append(ep_result)

        if ep_result["episode_reward"] > best_reward:
            best_reward = ep_result["episode_reward"]
            best_result = ep_result
            best_schedule = final_schedule

    # Save AFTER schedule for the best eval episode, not the first one
    if best_schedule is not None:
        save_schedule_plot_from_schedule(
            best_schedule,
            os.path.join(eval_dir, "after_schedule.png"),
            cycles=args.cycles,
            label=args.label,
            by_cycle=args.by_cycle,
        )

    save_eval_reward_plot(results, os.path.join(eval_dir, "eval_episode_rewards.png"))

    summary = {
        "run_dir": args.run_dir,
        "model_path": model_path,
        "episodes": args.episodes,
        "config": config,
        "instance": instance_path,
        "best_episode_reward": float(best_reward) if best_result is not None else None,
        "best_episode_index_1based": (
            int(max(range(len(results)), key=lambda i: results[i]["episode_reward"]) + 1) if results else None
        ),
        "episode_results": results,
        "mean_episode_reward": float(sum(r["episode_reward"] for r in results) / max(1, len(results))),
        "mean_invalid_moves": float(sum(r["invalid_moves"] for r in results) / max(1, len(results))),
        "mean_delta_largest_hole_ns": float(sum(r["delta"]["largest_hole_ns"] for r in results) / max(1, len(results))),
        "mean_delta_total_useless_gaps": float(sum(r["delta"]["total_useless_gaps"] for r in results) / max(1, len(results))),
        "mean_delta_total_gap_count": float(sum(r["delta"]["total_gap_count"] for r in results) / max(1, len(results))),
        "mean_delta_acceptance_count": float(sum(r["delta"]["acceptance_count"] for r in results) / max(1, len(results))),
    }

    out_json = os.path.join(eval_dir, "evaluation_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\\nSaved evaluation artifacts in: {eval_dir}")


if __name__ == "__main__":
    main()