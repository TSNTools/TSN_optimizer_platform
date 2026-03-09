"""
Visual debug for one defragmentation step.

Shows:
1) schedule before action,
2) schedule after action,
3) port fragmentation metrics before/after.
"""

import matplotlib.pyplot as plt
import numpy as np

from scenario_tools import build_env, build_problem


def _build_move_action(env: object, shift_ratio: float = 0.35) -> np.ndarray:
    """
    Build a deterministic action that pushes each stream to a non-zero offset.
    """
    action = np.zeros((env.n_streams, env.n_ports), dtype=np.float32)
    for s_idx, sinfo in enumerate(env.stream_info):
        path = sinfo["path"]
        if not path:
            continue

        # spread streams over the cycle to make movement visible
        offset_ratio = (shift_ratio + (s_idx / max(1, env.n_streams)) * 0.5) % 0.95
        val = 1.0 + offset_ratio
        for pid in path:
            action[s_idx, pid - 1] = val
    return action


def _collect_port_metrics(env: object):
    met = env._get_port_metrics()
    return {
        "utilization": met[:, 0],
        "useless_gap_ratio": met[:, 1],
        "avg_useful_gap": met[:, 2],
    }


def main():
    problem = build_problem("uc_config.cfg")
    usecase = problem.instantiate()
    env = build_env(problem, usecase, fragmentation_metric="hybrid")

    obs0, _ = env.reset()
    sched0 = env.get_combined_schedule()
    metrics0 = _collect_port_metrics(env)
    info0 = {
        "frag": env._fragmentation_value(),
        "all_gaps": env._count_gaps(),
        "useless_gaps": env._count_useless_gaps(),
    }

    action = _build_move_action(env)
    obs1, reward, _, _, info1 = env.step(action)
    sched1 = env.get_combined_schedule()
    metrics1 = _collect_port_metrics(env)

    print("Before:", info0)
    print("After:", {
        "frag": info1["new_fragmentation"],
        "all_gaps": info1["all_gaps"],
        "useless_gaps": info1["useless_gaps"],
    })
    print("Reward:", reward, "Delta:", info1["delta_frag"], "Moved:", info1["n_moved"])

    fig1 = sched0.plot(label=True)
    fig1.suptitle("Before defrag step", fontsize=14)

    fig2 = sched1.plot(label=True)
    fig2.suptitle("After defrag step", fontsize=14)

    ports = np.arange(env.n_ports)
    fig3, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(ports, metrics0["utilization"], "o-", label="before")
    axes[0].plot(ports, metrics1["utilization"], "o-", label="after")
    axes[0].set_ylabel("Utilization")
    axes[0].legend()

    axes[1].plot(ports, metrics0["useless_gap_ratio"], "o-", label="before")
    axes[1].plot(ports, metrics1["useless_gap_ratio"], "o-", label="after")
    axes[1].set_ylabel("Useless gap ratio")
    axes[1].legend()

    axes[2].plot(ports, metrics0["avg_useful_gap"], "o-", label="before")
    axes[2].plot(ports, metrics1["avg_useful_gap"], "o-", label="after")
    axes[2].set_ylabel("Avg useful gap")
    axes[2].set_xlabel("Port index")
    axes[2].legend()

    fig3.suptitle("Per-port fragmentation metrics", fontsize=14)
    fig3.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
