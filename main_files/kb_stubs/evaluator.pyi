from enum import Enum
from typing import Any, Tuple
import numpy as np

class ScheduleEncodingType(Enum):
    PRIORITY = 1
    START_TIME = 2

class TSNConstraintEvaluator:
    # Built FROM a TSNScheduling that already has an active use case. scheduling_type
    # defaults to PRIORITY, which is why problem.evaluator is never START_TIME.
    def __init__(self, problem: Any, weights: Any = ..., check_routing: bool = ...,
                 check_task_placement: bool = ...,
                 scheduling_type: ScheduleEncodingType = ...) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def evaluate(self, chromosome: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...
    def feasible_streams(self, chromosome: np.ndarray) -> np.ndarray: ...
