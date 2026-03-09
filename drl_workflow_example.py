"""
Minimal end-to-end workflow for TSN defragmentation DRL experiments.

Flow:
1) Build problem from config.
2) Generate train/test usecases.
3) Train loop placeholder (random actions for now).
4) Build a final Schedule (slot logic) from chosen offsets.
"""

import numpy as np

from scenario_tools import (
    build_env,
    build_problem,
    generate_scenario_split,
    make_action_from_first_offsets,
    schedule_from_first_offsets,
)


def random_policy_action(env):
    """
    Random action with explicit stream moves.

    Pure Box sampling is centered around action~=1, which decodes near
    first-offset 0 after projection; this makes schedule updates weak.
    Here we force per-stream path-port actions in [1.05, 1.95].
    """
    action = np.zeros((env.n_streams, env.n_ports), dtype=np.float32)
    for s_idx, sinfo in enumerate(env.stream_info):
        if not sinfo["path"]:
            continue
        val = np.random.uniform(1.05, 1.95)
        for pid in sinfo["path"]:
            action[s_idx, pid - 1] = val
    return action


def main():
    config_file = "uc_config.cfg"
    problem = build_problem(config_file)

    # Create train/test scenarios from the same config distribution.
    split = generate_scenario_split(problem, n_train=5, n_test=2)

    # --- "Training" loop placeholder ---
    for episode_idx, usecase in enumerate(split.train):
        env = build_env(problem, usecase, fragmentation_metric="hybrid")
        obs, _ = env.reset()

        episode_reward = 0.0
        for _ in range(10):  # replace with your RL horizon
            action = random_policy_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        print(
            f"[train] episode={episode_idx} reward={episode_reward:.4f} "
            f"frag={info['new_fragmentation']:.4f} "
            f"all_gaps={info['all_gaps']} useless_gaps={info['useless_gaps']}"
        )

    # --- Testing + schedule export example ---
    test_usecase = split.test[0]
    test_env = build_env(problem, test_usecase, fragmentation_metric="hybrid")
    obs, _ = test_env.reset()

    # Example: keep all streams at offset 0 on first hop.
    chosen_offsets = np.zeros(test_env.n_streams, dtype=float)
    action = make_action_from_first_offsets(test_env, chosen_offsets)
    obs2, reward, _, _, info = test_env.step(action)
    print(
        f"[test] one-step reward={reward:.4f}, delta={info['delta_frag']:.4f}, "
        f"frag={info['new_fragmentation']:.4f}"
    )

    # Build a standalone Schedule object using the same offset logic.
    final_schedule = schedule_from_first_offsets(
        problem=problem,
        usecase=test_usecase,
        first_offsets_ns=chosen_offsets.tolist(),
    )
    print(f"Final schedule slots: {len(final_schedule)}")

    # Optional: visualize
    # fig = final_schedule.plot()
    # fig.show()


if __name__ == "__main__":
    main()
