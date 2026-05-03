from __future__ import annotations

import argparse
import json
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from env import TSNCandidatePathEnv
from tsn_scheduling_problem import TSNScheduling


class SaveBestMeanRewardCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        if not any(bool(x) for x in dones):
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


class StopAfterNEpisodesCallback(BaseCallback):
    def __init__(self, max_episodes: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_episodes = int(max_episodes)
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        finished = sum(bool(x) for x in dones)
        if finished == 0:
            return True

        for done, info in zip(dones, infos):
            if not bool(done):
                continue

            self.episode_count += 1

            ep = info.get("episode", {})
            ep_reward = ep.get("r", None)
            ep_len = ep.get("l", None)

            ep_sbr = info.get("episode_sbr", None)
            ep_invalid = info.get("episode_invalid_moves", None)
            ep_dp = info.get("episode_decision_points", None)

            if self.verbose:
                msg = f"[callback] episode {self.episode_count}/{self.max_episodes}"
                if ep_reward is not None:
                    msg += f" | reward={ep_reward:.4f}"
                if ep_len is not None:
                    msg += f" | len={ep_len}"
                if ep_sbr is not None:
                    msg += f" | sbr={ep_sbr:.4f}"
                if ep_invalid is not None:
                    msg += f" | invalid={ep_invalid}"
                if ep_dp is not None:
                    msg += f" | dp={ep_dp}"
                print(msg)

            if self.episode_count >= self.max_episodes:
                return False

        return True


def make_env(args):
    problem = TSNScheduling(args.config)
    instance = problem.import_scheduling_usecase(args.instance)

    env = TSNCandidatePathEnv(
        problem=problem,
        usecase=instance,
        useless_gap_threshold_bytes=args.useless_gap_threshold_bytes,
        expected_incoming_bytes=args.expected_incoming_bytes,
        invalid_penalty=args.invalid_penalty,
        max_candidates=args.max_candidates,
        max_steps=args.max_steps,
        max_hops=args.max_hops,
        init_from_usecase_schedule=not args.no_usecase_schedule,
        include_current_offset_candidate=not args.no_current_offset_candidate,
        per_hop_candidate_limit=args.per_hop_candidate_limit,
        beam_width=args.beam_width,
        defrag_cycle_penalty=args.defrag_cycle_penalty,
        reallocation_penalty=args.reallocation_penalty,
        lambda_arrival=args.lambda_arrival,
        mu_departure=args.mu_departure,
        dt=args.dt,
        decision_points_per_episode=args.decision_points_per_episode,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    monitor_path = os.path.join(args.out_dir, "monitor.csv")
    env = Monitor(
        env,
        monitor_path,
        info_keywords=(
            "episode_sbr",
            "episode_acceptance",
            "episode_arrivals",
            "episode_rejections",
            "episode_decision_points",
            "episode_invalid_moves",
            "episode_violation_sum",
            "episode_reallocations",
            "episode_defrag_cycles",
        ),
    )
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--useless-gap-threshold-bytes", type=int, default=1024)
    parser.add_argument("--expected-incoming-bytes", type=int, default=1024)

    parser.add_argument("--invalid-penalty", type=float, default=50.0)
    parser.add_argument("--defrag-cycle-penalty", type=float, default=0.05)
    parser.add_argument("--reallocation-penalty", type=float, default=0.01)

    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-hops", type=int, default=None)
    parser.add_argument("--per-hop-candidate-limit", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=None)

    parser.add_argument("--lambda-arrival", type=float, default=1.0)
    parser.add_argument("--mu-departure", type=float, default=0.2)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--decision-points-per-episode", type=int, default=100)

    parser.add_argument("--no-usecase-schedule", action="store_true")
    parser.add_argument("--no-current-offset-candidate", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--exploration-fraction", type=float, default=0.2)
    parser.add_argument("--exploration-initial-eps", type=float, default=1.0)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = make_env(args)

    if args.resume_from is not None:
        print(f"Resuming training from: {args.resume_from}")
        model = DQN.load(args.resume_from, env=env, device="cpu")
    else:
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

    callback = CallbackList(
        [
            SaveBestMeanRewardCallback(
                save_path=os.path.join(args.out_dir, "best_model"),
            ),
            StopAfterNEpisodesCallback(max_episodes=args.train_episodes),
        ]
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
        reset_num_timesteps=False,
    )
    model.save(os.path.join(args.out_dir, "final_model"))

    with open(os.path.join(args.out_dir, "run_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("Training finished.")
    print(f"Artifacts saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
