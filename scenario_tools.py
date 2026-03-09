"""
Utilities to generate TSN DRL scenarios and convert them to schedules.

This module is intentionally lightweight so you can use it for:
- train/test usecase generation from a config file,
- environment construction,
- deterministic schedule creation from per-stream first-hop offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from env import TSNDefragEnv
from schedule import Schedule
from tsn_scheduling_problem import TSNScheduling
from usecase_generator import Usecase


@dataclass
class ScenarioSplit:
    """Container for generated scenario splits."""

    train: List[Usecase]
    test: List[Usecase]


def build_problem(config_file: str) -> TSNScheduling:
    """Create a TSN scheduling problem object from a config file path."""
    return TSNScheduling(problem_definition=config_file)


def generate_scenario_split(
    problem: TSNScheduling,
    n_train: int,
    n_test: int,
) -> ScenarioSplit:
    """
    Generate random usecases and return them as train/test splits.

    Uses repeated `instantiate()`, so each sample is a fresh random scenario
    from the same config distribution.
    """
    if n_train < 0 or n_test < 0:
        raise ValueError("n_train and n_test must be non-negative.")

    train = [problem.instantiate() for _ in range(n_train)]
    test = [problem.instantiate() for _ in range(n_test)]
    return ScenarioSplit(train=train, test=test)


def build_env(
    problem: TSNScheduling,
    usecase: Usecase,
    gap_threshold_ratio: float = 0.01,
    move_penalty: float = 0.01,
    max_edges: int = 200,
    fragmentation_metric: str = "hybrid",
    single_move_per_step: bool = False,
) -> TSNDefragEnv:
    """Construct a defragmentation environment for one usecase."""
    return TSNDefragEnv(
        problem=problem,
        usecase=usecase,
        gap_threshold_ratio=gap_threshold_ratio,
        move_penalty=move_penalty,
        max_edges=max_edges,
        fragmentation_metric=fragmentation_metric,
        single_move_per_step=single_move_per_step,
    )


def schedule_from_first_offsets(
    problem: TSNScheduling,
    usecase: Usecase,
    first_offsets_ns: Optional[Sequence[float]] = None,
    gap_threshold_ratio: float = 0.01,
    move_penalty: float = 0.0,
    max_edges: int = 200,
) -> Schedule:
    """
    Build a `Schedule` from stream first-hop offsets.

    Parameters
    ----------
    first_offsets_ns:
        Sequence of length N (N streams), where item i is the desired start
        offset for stream i at its first path port, in nanoseconds.
        If None, all offsets default to 0.
    """
    env = build_env(
        problem=problem,
        usecase=usecase,
        gap_threshold_ratio=gap_threshold_ratio,
        move_penalty=move_penalty,
        max_edges=max_edges,
    )
    env.reset()

    n_streams = env.n_streams
    if first_offsets_ns is None:
        first_offsets_ns = [0.0] * n_streams
    if len(first_offsets_ns) != n_streams:
        raise ValueError(
            f"Expected {n_streams} first offsets, got {len(first_offsets_ns)}."
        )

    for s_idx, off in enumerate(first_offsets_ns):
        env._place_stream(s_idx, float(off))

    return env.get_combined_schedule()


def make_action_from_first_offsets(
    env: TSNDefragEnv, first_offsets_ns: Sequence[float]
) -> List[List[float]]:
    """
    Convert first-hop offsets (ns) into an env action matrix (N x P).

    The same action value is written to each path port for a stream so the
    projection in `TSNDefragEnv.step()` decodes exactly this first offset.
    """
    if len(first_offsets_ns) != env.n_streams:
        raise ValueError(
            f"Expected {env.n_streams} first offsets, got {len(first_offsets_ns)}."
        )

    action = [[0.0 for _ in range(env.n_ports)] for _ in range(env.n_streams)]
    for s_idx, off in enumerate(first_offsets_ns):
        val = float(off) / env.cycle_ns + 1.0
        for pid in env.stream_info[s_idx]["path"]:
            action[s_idx][pid - 1] = val
    return action
