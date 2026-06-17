from __future__ import annotations

import argparse
import json
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from env_clean_simple import TSNCandidatePathEnv
from tsn_scheduling_problem import TSNScheduling

DEFAULT_ED_ED_SETTINGS = {
    "packet_sizes": [64, 128, 256, 512],
    "periods": [500_000, 1000_000],
    "latency_factors": [0.7, 0.8, 0.9, 1.0],
    "priorities": [1],
    "gammas": [1.0],
    "jitter_factors": [0.0],
}


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
            ep_acc = info.get("episode_acceptance", None)
            ep_arr = info.get("episode_arrivals", None)
            ep_rej = info.get("episode_rejections", None)
            ep_invalid = info.get("episode_invalid_moves", None)
            ep_viol = info.get("episode_violation_sum", None)
            ep_realloc = info.get("episode_reallocations", None)
            ep_cycles = info.get("episode_defrag_cycles", None)
            ep_dp = info.get("episode_decision_points", None)

            if self.verbose:
                msg = f"[callback] episode {self.episode_count}/{self.max_episodes}"
                if ep_reward is not None:
                    msg += f" | reward={ep_reward:.4f}"
                if ep_len is not None:
                    msg += f" | len={ep_len}"
                if ep_sbr is not None:
                    msg += f" | sbr={ep_sbr:.4f}"
                if ep_acc is not None:
                    msg += f" | acc={ep_acc:.4f}"
                if ep_arr is not None:
                    msg += f" | arr={ep_arr}"
                if ep_rej is not None:
                    msg += f" | rej={ep_rej}"
                if ep_invalid is not None:
                    msg += f" | invalid={ep_invalid}"
                if ep_viol is not None:
                    msg += f" | viol={ep_viol}"
                if ep_realloc is not None:
                    msg += f" | realloc={ep_realloc}"
                if ep_cycles is not None:
                    msg += f" | cycles={ep_cycles}"
                if ep_dp is not None:
                    msg += f" | dp={ep_dp}"
                reward_diag = (
                    f" | moves={info.get('move_actions', 0)}"
                    f" stops={info.get('stop_actions', 0)}"
                    f" forced={info.get('forced_stops', 0)}"
                    f" | sumER={info.get('sum_E_reward', 0.0):+.2f}"
                    f" sumFR={info.get('sum_frss_reward', 0.0):+.2f}"
                    f" rawM={info.get('sum_raw_move_reward', 0.0):+.2f}"
                    f" stopR={info.get('sum_stop_reward', 0.0):+.2f}"
                    f" penR={info.get('sum_realloc_penalty', 0.0):+.2f}"
                    f" penC={info.get('sum_cycle_penalty', 0.0):+.2f}"
                    f" dEavg={info.get('avg_delta_E', 0.0):+.5f}"
                    f" dFavg={info.get('avg_delta_frss', 0.0):+.5f}"
                    f" moveR={info.get('sum_scaled_move_reward', 0.0):+.2f}"
                    f" termSBR={info.get('terminal_sbr_reward', 0.0):+.2f}"
                )
                msg += reward_diag
                print(msg, flush=True)

            if self.episode_count >= self.max_episodes:
                return False

        return True


class RewardDebugCallback(BaseCallback):
    def __init__(self, max_steps: int = 200, verbose: int = 1):
        super().__init__(verbose)
        self.max_steps = int(max_steps)
        self.printed = 0

    def _on_step(self) -> bool:
        if self.printed >= self.max_steps:
            return True

        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", [])

        for i, info in enumerate(infos):
            if self.printed >= self.max_steps:
                break

            action = actions[i] if len(actions) > i else None

            if info.get("stop_action", False):
                print(
                    "[reward-debug] "
                    f"step={self.num_timesteps} action={action} STOP "
                    f"reward={info.get('reward', 0.0):+.4f} "
                    f"sbr={info.get('sbr', 0.0):.4f} "
                    f"cands={info.get('candidate_count', -1)} "
                    f"forced={info.get('forced_to_stop', False)}",
                    flush=True,
                )
            else:
                print(
                    "[reward-debug] "
                    f"step={self.num_timesteps} action={action} MOVE "
                    f"reward={info.get('reward', 0.0):+.4f} "
                    f"raw={info.get('raw_move_reward', 0.0):+.4f} "
                    f"ER={info.get('E_reward', 0.0):+.4f} "
                    f"frssR={info.get('frss_reward', 0.0):+.4f} "
                    f"dE={info.get('delta_E', 0.0):+.5f} "
                    f"E={info.get('E_before', 0.0):.4f}->{info.get('E_after', 0.0):.4f} "
                    f"dFRSS={info.get('delta_frss', 0.0):+.5f} "
                    f"penR={info.get('realloc_penalty', 0.0):.3f} "
                    f"penC={info.get('cycle_penalty', 0.0):.3f} "
                    f"invalid={info.get('invalid_move', False)}",
                    flush=True,
                )

            self.printed += 1

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
        ed_ed_settings=DEFAULT_ED_ED_SETTINGS,
        top_k_oldest_random=args.top_k_oldest_random,
        seed=args.seed,
        reward_alpha=args.reward_alpha,
        min_reallocations_per_cycle=args.min_reallocations_per_cycle,
        early_stop_penalty=args.early_stop_penalty,
        stream_cooldown_decisions=args.stream_cooldown_decisions,
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
            "episode_requested_departures",
            "episode_executed_departures",
            "episode_skipped_departures",
            "episode_trace_steps",
            "episode_invalid_moves",
            "episode_violation_sum",
            "episode_reallocations",
            "episode_defrag_cycles",
            "sum_E_reward",
            "sum_frss_reward",
            "sum_raw_move_reward",
            "sum_stop_reward",
            "sum_realloc_penalty",
            "sum_cycle_penalty",
            "avg_delta_E",
            "avg_delta_frss",
            "move_actions",
            "stop_actions",
            "forced_stops",
            "terminal_sbr_reward",
        ),
    )
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-k-oldest-random", type=int, default=10)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--reward-alpha", type=float, default=0.7)

    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--train-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--useless-gap-threshold-bytes", type=int, default=1024)
    parser.add_argument("--expected-incoming-bytes", type=int, default=1024)

    parser.add_argument("--invalid-penalty", type=float, default=50.0)
    parser.add_argument("--defrag-cycle-penalty", type=float, default=0.05)
    parser.add_argument("--reallocation-penalty", type=float, default=0.01)

    parser.add_argument("--debug-reward-steps", type=int, default=0)

    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-hops", type=int, default=None)
    parser.add_argument("--per-hop-candidate-limit", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=None)

    parser.add_argument("--min-reallocations-per-cycle", type=int, default=5)
    parser.add_argument("--early-stop-penalty", type=float, default=0.10)
    parser.add_argument("--stream-cooldown-decisions", type=int, default=3)

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

    callbacks = [
        SaveBestMeanRewardCallback(
            save_path=os.path.join(args.out_dir, "best_model"),
        ),
        StopAfterNEpisodesCallback(max_episodes=args.train_episodes),
    ]

    if args.debug_reward_steps > 0:
        callbacks.insert(0, RewardDebugCallback(max_steps=args.debug_reward_steps))

    callback = CallbackList(callbacks)

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
