# TSN Optimizer Platform - AI Handoff Context

## 1. Project Overview

### Purpose
`TSN_optimizer_platform` is a development platform for building and testing scheduling/optimization algorithms for Time-Sensitive Networking (TSN), with a current focus on:
- generating TSN problem instances,
- representing and validating schedules,
- enabling DRL-based schedule defragmentation,
- integrating external optimization outputs (ILP) back into platform schedule objects.

### Problem It Solves
TSN requires deterministic per-port transmission windows that satisfy constraints (latency, jitter, path ordering, periodicity). This project provides:
- problem/instance generation (`TSNScheduling` + `Usecase`),
- schedule representation (`Schedule` + `Slot`),
- validation/plotting tools,
- an RL environment for defragmentation (`env.py`),
- conversion from external ILP solver outputs (`ilp_to_platform.py`).

### High-Level System Description
1. Build a **problem definition** from config (`uc_config.cfg`) via `TSNScheduling`.
2. Create one or many **instances** (`Usecase`) with concrete topology + streams.
3. Build/attach a **Schedule** for a specific instance.
4. Validate conflicts, inspect utilization/fragmentation, and visualize schedules.
5. Optionally run DRL defragmentation or external ILP and convert results into platform schedule format.

---

## 2. Architecture

### Main Components
- `usecase_generator.py`
  - `UsecaseGenerator`: random topology/stream generation.
  - `Usecase`: concrete instance object (topology graph + streams + optional attached schedule).
- `tsn_scheduling_problem.py`
  - `TSNScheduling`: problem wrapper around usecases; metadata, port/stream mappings, and helper calculations.
- `schedule.py`
  - `Slot`: periodic transmission slot.
  - `Schedule`: collection of slots with compile/conflict checks, compact encoding conversion, plotting, and integration hooks into `Usecase`.
- `env.py`
  - `TSNDefragEnv`: Gymnasium environment for schedule defragmentation.
- `rctsn/`
  - separate robust ILP scheduling pipeline (batch heuristic + Gurobi).
- `ilp_to_platform.py`
  - adapter from ILP pickle outputs to platform `Schedule`.

### Interaction Model
- `TSNScheduling.instantiate()` -> `Usecase`.
- `Usecase` can receive schedule via `addSchedule(schedule, port_id_to_name)`.
- `Schedule.update_graph()` and `Schedule.update_streams()` inject schedule output back into graph/stream data.
- `env.py` uses `problem + usecase` to produce state/action/reward and mutate schedules.
- `ilp_to_platform.py` bridges external ILP output into platform-native `Schedule`.

### Important Design Decisions
- Instance context is explicit: `with problem.use_instance(instance): ...`.
- Stream/port indexing is mostly 0-based for streams and 1-based for port IDs in problem mappings.
- Time fields in platform schedule are `*_ns`; external sources must be converted carefully.
- `Usecase.schedule` exists and can be attached/removed (not serialized by default in `to_json`).

---

## 3. Repository Structure

## Root
- `README.md`: minimal project title.
- `uc_config.cfg`: default generation config.
- `test.json`: imported test usecase.
- `test.ipynb`: exploratory notebook.
- `schedule.py`: core schedule data model and plotting.
- `usecase_generator.py`: topology/stream generator + `Usecase` class.
- `tsn_scheduling_problem.py`: problem wrapper and metadata helpers.
- `env.py`: RL defragmentation environment.
- `ilp_to_platform.py`: ILP output converter into `Schedule`.
- `test_conversion.ipynb`: ILP conversion test notebook.

## `rctsn/` (external/companion optimizer)
- `pipeline.py`: full ILP run pipeline.
- `models/data_factory.py`: stream/path preprocessing for ILP.
- `models/batches/optimization_model.py`: ILP model (variables x/z/a, constraints, batch fixing).
- `models/output_generator.py`: charts/metrics generation (wrapped/unwrapped visualizations).
- `results_to_json.py`: helper to convert outputs to JSON-like structure.
- `data/output/run_*/`: run artifacts (`variables_a.pkl`, `variables_z.pkl`, `variables_x.pkl`, `stream_dict.pkl`, charts).
- `2509.15930v1.pdf`: paper describing robust scheduling model and variables.

---

## 4. Core Logic

### 4.1 Problem/Instance Lifecycle
- `TSNScheduling(problem_definition)` loads config.
- `instantiate()` creates a new random `Usecase` and stores it in `problem.instances`.
- Each `Usecase` has:
  - graph (`networkx.DiGraph`),
  - stream list,
  - config,
  - optional attached `schedule`.

### 4.2 Schedule Model
- `Slot(start_ns, end_ns, period_ns, stream_id, port_id, queue_id)`.
- `Slot.all_occurrences(span_ns)` expands periodic instances.
- `Schedule.validate_conflicts()` checks overlaps per `port_id`.
- `Schedule.plot()` draws per-port stream windows.
- `Schedule.from_compact_encoding(...)` / `to_compact_encoding(...)` convert matrix format.
- `Schedule.update_graph(...)` and `update_streams(...)` write schedule output into graph/stream metadata.

### 4.3 Usecase Schedule Binding
- `Usecase.addSchedule(schedule, port_id_to_name)`:
  - stores schedule in `self.schedule`,
  - updates graph port schedules and stream offsets/path info.
- `Usecase.removeSchedule()` removes attached schedule and related graph/stream fields.

### 4.4 DRL Environment (`env.py`)
- Observation:
  - `stream_placement` (`N x P`)
  - `stream_features` (`N x 5`)
  - `port_metrics` (`P x 3`)
  - `adjacency_matrix` (`2 x E_max`)
- Action:
  - matrix `N x P`, decoded into stream first-hop offsets.
  - supports:
    - multi-move mode,
    - single-move-per-step mode.
- Reward:
  - `delta_fragmentation - move_penalty`.
  - `fragmentation_metric` options:
    - `gap_count`
    - `useless_gap_count`
    - `hybrid`.

### 4.5 ILP -> Platform Conversion (`ilp_to_platform.py`)
- Loads ILP run pickles.
- Extracts active streams (`a`) and selected path (`z`).
- Reads start times from `x`.
- Derives duration from `stream_dict` (`times`, `deviations`, `gamma`) with mode:
  - `best_case`, `worst_case`, or `robust`.
- Builds platform `Schedule` with deterministic port mapping.
- Supports:
  - exact per-repetition slots,
  - compact periodic slots (if repetition consistency holds).

---

## 5. Technologies Used

### Languages
- Python (primary)
- Jupyter notebooks for experimentation

### Libraries
- `numpy`
- `networkx`
- `matplotlib`
- `gymnasium` (DRL environment)
- `pulp` + `gurobi` (in `rctsn` ILP pipeline)
- `dataclasses_json` (used by `schedule.py`)

### Tools
- Gurobi solver for ILP runs (`rctsn`)
- pickle artifacts for solver output persistence

---

## 6. Current Development State

### Implemented Features
- Problem/instance generation and import (`TSNScheduling`, `UsecaseGenerator`).
- Schedule model with conflict checking and plotting (`schedule.py`).
- Instance-level schedule attachment (`Usecase.addSchedule/removeSchedule`).
- DRL environment for defragmentation with configurable action semantics and reward metrics.
- Single-move-per-step option in environment.
- ILP output conversion to platform schedule (`ilp_to_platform.py`).
- Notebook for conversion testing (`test_conversion.ipynb`).

### Partially Implemented / In Progress
- Tight integration between converted ILP schedules and full `Usecase` object pipeline in notebooks/workflow.
- Wrapped-vs-unwrapped visualization parity between platform plotting and `rctsn` plotting semantics.
- Scenario curriculum and robust training pipeline for RL (concept discussed; not finalized).

### Known Bugs / Technical Debt
- **API drift risk:** `env.py` still uses older `Schedule/Slot` constructor field names (`cycle_ms`, `start_ms`, `end_ms`) while current `schedule.py` uses `cycle_ns`, `start_ns`, `end_ns`. This likely needs harmonization.
- `schedule.py` imports `dataclasses_json`; environments missing this dependency will fail imports.
- ILP conversion currently focuses on transmission windows; does not encode blocked periods from `rctsn` wrapped charts.
- `Usecase.addSchedule()` assumes stream slots exist and may require guards if some streams are unscheduled.

---

## 7. Important Knowledge From AI Discussions

### Architecture Decisions
- Keep the platform as reference source of truth; external solver outputs must adapt to platform schedule schema.
- Link schedule to instance through `Usecase` (not just loose objects).
- Prefer reusable conversion/adaptation helpers over hardcoded notebook logic.

### Debugging Discoveries
- Zero/flat rewards were caused by action decoding behavior and weak random moves near neutral offsets.
- Distinction between wrapped and unwrapped ILP plots:
  - `rctsn` wrapped charts merge intervals and include blocked periods.
  - platform schedule plot shows slot occurrences directly.

### Performance Considerations
- For single-move RL, touched-port incremental metrics were identified as future optimization opportunity.
- External ILP baselines are useful for strong initial schedules and benchmarking RL improvements.

### Constraints / Assumptions
- Stream IDs expected 0-based in most platform logic.
- Port IDs in problem mapping are 1-based; queue/port indexing conversions must be explicit.
- Unit consistency is critical; platform schedule uses `*_ns` fields.

### Rejected / Revised Approaches
- Hard forcing only “good moves” in policy simulation was rejected (exploration should allow bad moves).
- TSNKit integration idea considered but deprioritized in favor of colleague ILP for easier collaboration/support.
- Strict macro-cycle validation in schedule setter was discussed and removed per team preference.

---

## 8. Important Prompts and AI Answers

### Prompt
Design a DRL env MDP for TSN defragmentation with action/state/reward encodings.

### AI Answer / Solution
Implemented `env.py` with:
- matrix action decoding,
- stream/port observations,
- fragmentation-based reward,
- schedule update logic.

### Impact on the Project
Established a functional Gymnasium environment foundation for RL experimentation.

---

### Prompt
Need visualization in notebook (no popup), frame-by-frame and GIF.

### AI Answer / Solution
Created notebook-based visualization flows and plotting via existing `Schedule.plot()`, with per-frame outputs and GIF generation.

### Impact on the Project
Improved observability/debugging of schedule behavior and reward response.

---

### Prompt
Single move or no move per action is required.

### AI Answer / Solution
Added `single_move_per_step` mode in `env.py`; step applies at most one stream move and reports moved stream/port in `info`.

### Impact on the Project
Aligned environment behavior with requested action semantics and easier debugging.

---

### Prompt
Need explicit link between instance and schedule.

### AI Answer / Solution
Added schedule binding support in `Usecase` (`schedule` attribute + add/remove behavior); later refined validation approach.

### Impact on the Project
Instance-specific schedule ownership became explicit, simplifying downstream workflows.

---

### Prompt
Convert colleague ILP outputs to platform `Schedule`, general not hardcoded.

### AI Answer / Solution
Implemented `ilp_to_platform.py`:
- loads ILP pickles,
- extracts active streams/paths,
- computes durations from ILP metadata,
- converts to platform `Schedule`,
- includes conversion summary and exact/compact modes.

### Impact on the Project
Enabled interoperability between external ILP scheduling and platform-native schedule tools.

---

## 9. How to Run the Project

### Basic Platform (root)
1. Create/activate Python env.
2. Install core dependencies used by platform files:
   - `numpy`, `networkx`, `matplotlib`, `gymnasium`, `dataclasses-json`.
3. Use notebooks (`test.ipynb`, `test_conversion.ipynb`) or scripts as entry points.

### DRL Environment Example (if scripts available in local branch)
- Construct:
  - `problem = TSNScheduling("uc_config.cfg")`
  - `instance = problem.instantiate()`
  - `env = TSNDefragEnv(problem, instance, ...)`

### ILP Conversion Test
In notebook `test_conversion.ipynb`:
- Set `RUN_DIR` to run artifact folder.
- Set `time_scale_to_ns` (typically `1000` for us->ns).
- Run conversion cells and plotting.

### rctsn ILP Pipeline
Inside `rctsn/`:
1. Install requirements (`requirements.txt`) and solver (Gurobi).
2. Prepare `data/input/*`.
3. Run:
   - `python pipeline.py`
4. Use output run folder artifacts in `data/output/run_*`.

### Environment Variables
- No mandatory global env vars defined in platform root.
- Gurobi licensing/environment may be required by `rctsn`.

---

## 10. Future Work

### Planned Improvements
- Harmonize `env.py` with current `schedule.py` API (`*_ns` naming).
- Add robust mapping from converted ILP schedule directly into `Usecase.addSchedule` workflow.
- Add wrapped-interval plotting mode in platform to match `rctsn` wrapped semantics (including blocked periods).

### Missing Features
- Automated end-to-end validation suite for:
  - stream/path/port ID consistency,
  - unit consistency,
  - latency/jitter checks post-conversion.
- Unified scenario curriculum tooling for RL training/evaluation.

### Possible Optimizations
- Incremental fragmentation metric updates (touched-port only).
- Fast batched conversion/validation for many ILP runs.
- Strong baseline policy generation (sorted first-fit / best-fit) for RL comparisons.

---

## Quick Handoff Notes
- Treat platform files (`schedule.py`, `usecase_generator.py`, `tsn_scheduling_problem.py`) as source of truth.
- Use `ilp_to_platform.py` as adapter boundary; avoid mixing `rctsn` internals into platform modules.
- Confirm time-scale and ID conventions early in any new integration path.
