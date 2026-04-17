from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

from gymnasium.wrappers import RecordEpisodeStatistics
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from env2 import TSNDefragStepEnv
from tsn_scheduling_problem import TSNScheduling
from usecase_generator import Usecase


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.invalid_moves = 0

    def reset(self, **kwargs):
        self.invalid_moves = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("invalid_move", False):
            self.invalid_moves += 1

        if terminated or truncated:
            info = dict(info)
            info["episode_invalid_moves"] = self.invalid_moves
            info["episode_violation_sum"] = self.env._violation_sum()

        return obs, reward, terminated, truncated, info

class SaveBestMeanRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = int(check_freq)
        self.save_path = save_path
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        ep_info = self.model.ep_info_buffer
        if len(ep_info) == 0:
            return True

        mean_reward = sum(x["r"] for x in ep_info) / len(ep_info)
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.model.save(self.save_path)
            if self.verbose:
                print(f"[callback] new best mean reward={mean_reward:.4f}")
        return True


def make_env(args):
    problem = TSNScheduling(args.config)
    instance = problem.import_scheduling_usecase(args.instance)

    env = TSNDefragStepEnv(
        problem=problem,
        usecase=instance,
        step_ns=args.step_ns,
        useless_gap_threshold_bytes=args.useless_gap_threshold_bytes,
        invalid_penalty=args.invalid_penalty,
        largest_hole_weight=args.largest_hole_weight,
        useless_gap_weight=args.useless_gap_weight,
        max_steps=args.max_steps,
        init_from_usecase_schedule=not args.no_usecase_schedule,
        reset_mode=args.reset_mode,
        freeze_reset_after_episode=args.freeze_reset_after_episode,
    )

    env = EpisodeStatsWrapper(env)

    monitor_path = os.path.join(args.out_dir, "monitor.csv")
    env = Monitor(
        env,
        monitor_path,
        info_keywords=("episode_invalid_moves", "episode_violation_sum"),
    )
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-ns", type=int, default=1000)
    parser.add_argument("--useless-gap-threshold-bytes", type=int, default=1024)
    parser.add_argument("--invalid-penalty", type=float, default=10000.0)
    parser.add_argument("--largest-hole-weight", type=float, default=1.0)
    parser.add_argument("--acceptance-weight", type=float, default=1.0)
    parser.add_argument("--useless-gap-weight", type=float, default=1.0)
    parser.add_argument("--expected-incoming-bytes", type=int, default=1024)

    parser.add_argument("--max-steps", type=int, default=2_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--target-update-interval", type=int, default=2000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--no-usecase-schedule", action="store_true")
    parser.add_argument("--reset-mode", choices=["always", "never", "freeze_after"], default="never")
    parser.add_argument("--freeze-reset-after-episode", type=int, default=100000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = make_env(args)

    model = DQN(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=1.0,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=os.path.join(args.out_dir, "tb"),
        seed=args.seed,
        verbose=1,
        device="cpu",
    )

    callback = SaveBestMeanRewardCallback(
        check_freq=5000,
        save_path=os.path.join(args.out_dir, "best_model"),
    )

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    model.save(os.path.join(args.out_dir, "final_model"))

    with open(os.path.join(args.out_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Training finished.")
    print(f"Artifacts saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
