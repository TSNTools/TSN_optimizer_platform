# TSN OF-FF Heuristic Handoff

This file summarizes the current state of the defragmentation heuristic implementation so another assistant/session can continue immediately.

## Scope
- Repository: TSN optimizer platform
- Main implementation file: `heuristic.py`
- Input schedule source: `instance.schedule` converted to compact encoding
- Validation backend: `TSNConstraintEvaluator` from `evaluator.py` (START_TIME scheduling checks only)

## Core Goal
Implement an OF-FF(p,n) baseline heuristic for GCL defragmentation in TSN, with:
1. Departure phase (`p` streams removed)
2. Older-first selection (`n` streams considered)
3. First-possible-fit movement per port/path
4. Acceptance only when scheduling violations do not worsen

## Current Implementation (`heuristic.py`)

### Main public APIs
- `of_ff(problem, instance, p, n, stream_ages=None, departed_stream_ids=None, frame_size_thresholds=(500,1000,2000,5000))`
- `of_ff_with_current_info(problem, instance, p, n, stream_ages=None, departed_stream_ids=None, frame_size_thresholds=(500,1000,2000,5000))`
- `infer_stream_ages_from_compact(compact)`
- `fragmentation_metrics(compact, wind_lengths, cycle_ns, thresholds=None)`

### Result object
`HeuristicRunResult` contains:
- `before_compact`, `after_compact`
- `before_schedule`, `after_schedule`
- `departed_stream_ids`, `considered_stream_ids`, `moved_stream_ids`
- `scheduler_violation_sum_before`, `scheduler_violation_sum_after`
- `metrics_before`, `metrics_after`, `metrics_delta`
- `debug_reports`

### Movement logic
- Per-stream ports are ordered in a topology-aware order (from evaluator upstream/downstream maps, with fallback).
- Lower bounds are causal and hop-aware:
  - downstream earliest start >= upstream selected start + hop_cost
- Candidate starts are scanned earliest-first.
- For each port, heuristic picks the first candidate that passes scheduling gate (non-worsening violation sum) at that step.
- Final stream candidate is accepted only if total scheduling violation sum <= baseline.

### Debug reports per attempted stream
For each attempted stream, report includes:
- `port_attempts` list:
  - `port`, `current_start`, `lower_bound`, `first_fit_start`, `had_earlier_hole`, `checked_candidates`
- `baseline_violation_sum`, `candidate_violation_sum`
- `accepted`, `reason`
- `per_check_violation_deltas` for:
  - `stream_window_sequence`, `interleaving`, `macrocycle_overlap`, `talker_offset`, `maxlatency_violations`

## Metrics now included in heuristic

### Fragmentation metrics (before/after)
- `largest_hole_per_port`
- `largest_hole_mean`
- `largest_hole_max`
- `gap_count_total`

### Threshold-based accommodation metrics
For each threshold `c` in `frame_size_thresholds`:
- `holes_ge_c_per_port[c]`
- `holes_ge_c_total[c]`
- `c_capacity_total[c]` (sum of floor(hole/c) over all holes)

### Delta metrics
`metrics_delta` stores after-before for numeric/array/dict-numeric fields.

## Stream age policy
If `stream_ages` is not passed:
- Ages are inferred directly from compact encoding.
- For each stream: earliest non-zero start over all ports.
- Streams with lower earliest start are considered older.
- Returned age convention: higher age means older.

## Data-model note
- Compact encoding convention: value > 0 means `start_ns + 1`; 0 means inactive.

## Instance update performed
`instance2.json` switch processing delay values were normalized:
- Changed all switch `"procDelay": 40000` to `"procDelay": 40`.

## Quick notebook usage
```python
result, current_info = of_ff_with_current_info(
    problem=problem,
    instance=instance,
    p=10,
    n=5,
    stream_ages=None,
    departed_stream_ids=None,
    frame_size_thresholds=(500, 1000, 2000, 5000),
)

print(result.metrics_before["largest_hole_mean"], result.metrics_after["largest_hole_mean"])
print(result.metrics_after["holes_ge_c_total"])  # dict keyed by c
```

## Recommended next steps
1. Add an optional lower-bound mode switch in heuristic:
   - hop-cost based (current)
   - duration-based (upstream start + upstream duration)
2. Add CSV exporter for per-run metrics and debug summary.
3. Build a benchmark harness (multiple instances/seeds) for heuristic vs DRL comparison.
4. Add unit tests around:
   - age inference
   - threshold metrics
   - acceptance/rejection logic
   - debug report consistency.
