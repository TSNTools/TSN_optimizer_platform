# LLM Library-Assistance Evaluation — TSN Scheduling Codebase

**Evaluated artifacts:** `A.txt`, `B.txt`, `C.txt`, `D.txt` (one LLM/pipeline run)
**Ground truth (knowledge base):** `evaluator.py`, `tsn_scheduling_problem.py`, `usecase_generator.py`
**Evaluation question:** *Can the LLM actually help someone use the library?*

---

## 0. Method

Every answer was checked line-by-line against the actual source code of the three knowledge-base files. Scoring rubrics used:

**Section A (0–5 each):**
- **Correctness** — 5 = no factual errors; 4 = one minor slip; 3 = a noticeable factual error; 2 = a significant error (wrong shape/unit/usage); 1 = mostly wrong or false claim about the codebase; 0 = answer is about the wrong thing entirely.
- **Completeness** — 5 = purpose + inputs + outputs + logic + relevant caveats; 3 = core idea only; 1–2 = one-liner or fragment; 0 = does not answer the question.
- **Easy to understand** — readability, structure, absence of noise/repetition.

**Section B accuracy levels (per the user's rubric):**
- **L1** = exactly the right function(s) returned; **L2** = right answer present plus unnecessary/less-precise extras; **L3** = partial answer (a required function missing or wrongly denied); **L4** = wrong answer.
For multi-part questions the question-level grade is the **weakest sub-answer**. Sub-answer accuracy, precision and recall are also reported. Incorrect *auxiliary* claims (e.g. a wrong usage snippet next to a correctly named function) do not change the retrieval level but are counted separately.

**Section C:** semantic correctness (0–5, my code review), executability (syntax + runtime, separating genuine code bugs from harness/environment failures), integrability (0–5: does it use the library's real API/data model?), and unit-test pass rate (as recorded in the file's EXECUTION blocks).

**Section D:** runtime, correctness, executability, completeness against the request ("design/write the algorithm").

A note on scope: the evaluation plan mentions comparing several LLMs and larger request counts (100 per section). The uploaded files contain **one** model/pipeline run with 50 (A), 25 (B), 10 (C) and 1 (D) recorded answers; all numbers below refer to that run. The tables are structured so a second model's run can be scored identically and placed side by side.

---

## A. Explain existing code / variables

### A.1 Functions (25 queries)

| # | Query (abridged) | Corr. | Compl. | Underst. | Key observations |
|---|---|---:|---:|---:|---|
| F1 | Explain `streamGenerator` | 4 | 4 | 4 | Accurate overall; describes append-before-path-assignment (code applies `shortestPath`/`randomPath` *before* appending) — minor ordering slip. |
| F2 | What does `talkerEmbedding` do? (no module given) | 1 | 0 | 3 | **Failure:** routed the query to `tsn_scheduling_problem` and answered "There is no function named `talkerEmbedding`". The function exists in `usecase_generator.py`. False non-existence claim w.r.t. the knowledge base. |
| F3 | `talkerEmbedding` (module specified) | 5 | 2 | 4 | Correct, but is essentially the docstring verbatim; no mention of hotspot/stride/staggered mechanics or the return value. |
| F4 | How does `initialize_problem()` configure evaluator checks? | 4 | 3 | 3 | `check_routing = not(fr and ftp)` correctly restated, but never decodes `fr`/`ftp` (fixPath flag; talker & listener both fixed to 1 placement) nor states `check_task_placement = not ftp`; buries the answer under 4 unrelated function explanations. |
| F5 | Compute the macro-cycle (LCM of stream intervals) | 0 | 0 | 2 | **Wrong function.** Returned `make_default_stream_config` with a fabricated usage (`TSNScheduling().make_default_stream_config(stream={'talker': ...})`). Correct answer is the `macro_cycle` property. Usage also invalid (`TSNScheduling()` lacks the required `problem_definition` arg; stream keys invented). |
| F6 | Draw fat-tree layout on a matplotlib axis | 2 | 2 | 3 | Right function (`drawFattree`) but broken usage: `drawFattree(graph={'typ': 'ft'})` — the argument must be a networkx graph, not a dict; the call would crash. |
| F7 | Count streams with zero violations | 3 | 3 | 4 | Right function (`feasible_streams`) and correct concept ("count True values"), but usage `TSNConstraintEvaluator().feasible_streams(...)` omits the mandatory `problem` constructor argument and never shows the actual count (`.sum()`); chromosome given as list-of-lists. |
| F8 | Generate a Waxman random graph with 10 switches | 2 | 2 | 3 | Right function (`makeRandomGraph`) but usage `makeRandomGraph('waxman', directed=False)` doesn't match the signature `(cls, usecase, directed=True)`; never explains that `typ='w'` and `numswitchs=10` come from the usecase config. Assumptions list is confused. |
| F9 | Units of time/speed/bandwidth/data size | 1 | 3 | 3 | **Half wrong.** Time=ns ✓ and datasize=bytes ✓, but speed and bandwidth reported as "bps" (actually **Mbps** per the class docstrings), and both conversion explanations are wrong (period is µs→ns via ×1000, not ms→ns; maxLatency is ms→ns via ×1e6, not s→ns). Ignores the explicit `UsecaseGenerator` docstring stating the units. |
| F10 | What does `generateUC(configFile)` do / return? | 5 | 5 | 4 | Fully accurate incl. generator consumption via `list(...)`, index side effect, and returned `Usecase` object. |
| F11 | `remove_bias`/`add_bias`/`gene_value_bias` in `suggest_corrections` | 3 | 4 | 4 | Purpose and remove/add shapes correct, but claims `gene_value_bias` is `(I, 8, P, S)` — the START_TIME stack has **9** planes; in PRIORITY mode it is `(I, 1, P, S)`. |
| F12 | What does `pathConvenient` check? | 5 | 4 | 4 | Accurate: bit-size, per-hop trans+prop+proc sum, `maxLatency > minLatency`, try/except robustness note. |
| F13 | What does `periodConvenient` check? | 3 | 4 | 4 | Logic correct, but invents a unit: claims port speed is "kbps"; it is Mbps and `/1000` converts Mbps→bits/ns. Also mislabels a local variable as a "side effect". |
| F14 | Topology when `graphConfig["typ"]=="ft"` | 5 | 3 | 4 | Correct (fat-tree via `k_ary_fat_tree` inside `makeTopology`); could quantify pods/cores/hosts as a function of k. |
| F15 | What is stored in `Usecase.ports`? | 5 | 4 | 5 | Correct keys (sourceNode, destPort, destNode, speed, queueCount, nodeName) with a helpful example; omits only the `name` key. |
| F16 | What does `Usecase.removeRouting()` modify? | 5 | 5 | 4 | Exact: sets `fixPath=False` in config, deletes `pathLists` per stream with try/except. |
| F17 | Explain `_per_stream_host_role_violations` | 5 | 4 | 4 | Correct (5, n_streams) output with the row order matching the code; logic steps accurate. |
| F18 | Explain `_per_stream_flow_conservation_violations` | 5 | 4 | 4 | Correct (3, n_streams): backwarding, |in−out| at switches, in-degree>1 looping. |
| F19 | Explain `per_stream_all_violations` | 5 | 2 | 4 | Correct one-liner; omits the `tobytes()` result caching. |
| F20 | Explain `suggest_corrections` | 5 | 4 | 4 | Accurate and well-organized summary of remove/add pressures and both scheduling-type branches. |
| F21 | Explain `configure_stream_ports` | 5 | 3 | 4 | Correct summary; skips the listener→incoming-sourcePort conversion and the always-active-edge protection filter. |
| F22 | Explain `get_concurrent_ports` | 5 | 4 | 4 | Exact per-type behavior (switch / talker / listener) and the ValueError branch. |
| F23 | Explain `make_default_stream_config` | 5 | 2 | 4 | Correct but a single sentence; omits the `pathLists` forced-in/forced-out switch logic and the fixed-placement defaults. |
| F24 | Explain `bfs_backward` | 5 | 4 | 4 | Accurate: starts at (end_port, maxLatency), subtracts hop cost, builds `ss_max`, visited-pair set. |
| F25 | Explain `build_metadata` | 4 | 4 | 4 | Logic exact incl. `initialize_problem()` call, but states `self.usecase` "is a list of dictionaries" — it is a `Usecase` object (`usecase.streams` is the list of dicts). |

### A.2 Variables (25 queries)

| # | Query (abridged) | Corr. | Compl. | Underst. | Key observations |
|---|---|---:|---:|---:|---|
| V1 | `chromosome[p, s]` in start-time checks | 5 | 4 | 5 | Exact: start time of stream s at port p; 0 = inactive. |
| V2 | `weights` param of `TSNConstraintEvaluator.__init__` | 5 | 4 | 5 | Correct (optional dict merged over defaults); doesn't mention that only checks-relevant keys are kept. |
| V3 | `scheduling_type` param | 5 | 2 | 4 | Type and default correct, but never says what it *does* (selects priority-conflict vs. start-time violation checks). |
| V4 | `phase = start % interval` | 5 | 5 | 5 | Excellent: correct meaning, worked example (7 % 10), correct code cite. |
| V5 | `gene_value_bias` returned by `suggest_corrections` | 3 | 2 | 4 | One-sided: explains only the PRIORITY branch and implies the array is only used then; the far larger START_TIME 9-plane stack is omitted. |
| V6 | `s_idx` passed to `group_connected_ports` | 5 | 2 | 4 | Literally true ("loop variable over range(n_streams)") but shallow — never says it selects which stream's chromosome column is grouped. |
| V7 | `active_nodes` in `group_penalty` | 5 | 3 | 4 | Correct definition + code quote; doesn't explain why (node vs. edge counting for the penalty). |
| V8 | `expanded_act` in `suggest_corrections` | 2 | 2 | 3 | **Shape error:** claims result is `(I, 1, N, P, S)` (5-D). `act` is `(I,P,S)`; `act[:,None,:,:]` is `(I,1,P,S)`, broadcast against masks to `(I,N,P,S)`. |
| V9 | `sorted_hops` in `_per_stream_latency_violations_priority` | 5 | 4 | 4 | Correct: hop costs reordered by priority order, feeding cumulative delay. |
| V10 | `active_talkers` in `_per_stream_host_role_violations` | 5 | 3 | 4 | Correct (`act & talker_mask`); usage in talker_count/overlap not mentioned. |
| V11 | `self.config_matrix` in `build_metadata` | 5 | 2 | 4 | Only restates the allocation line; never explains the semantics (1 = port forced always-active for that stream). |
| V12 | `imp_uc` in `import_scheduling_usecase` | 5 | 3 | 4 | Correct (raw JSON dict); conversion to `Usecase.from_json` not mentioned. |
| V13 | `mapping` in `port_name_to_id` | 5 | 4 | 4 | Exact incl. '' → 0. |
| V14 | `destNode` in `hops_costs` | 5 | 4 | 4 | Correct: destination node used for propDelay edge lookup and procDelay node lookup. |
| V15 | `always_active_ids` in `configure_stream_ports` | 5 | 4 | 4 | Exact (0-based indices of forced-active ports). |
| V16 | `frame_size` in `hops_costs` | 5 | 3 | 4 | Correct (bits, drives transmission delay); omits that it is frames-per-interval × frame size × 8 × 1.1 tolerance. |
| V17 | `cur_egp` vs `next_egp` in `hop_cost_eg2eg` | 4 | 3 | 4 | First sentence right; "next_egp is the destination port" is loose — it is the *egress* port at the next hop, not the ingress/destination port. |
| V18 | `ports_in`/`ports_out` in `make_default_stream_config` | 5 | 3 | 4 | Correct roles/keys; the actual default contents are not described. |
| V19 | What does `n_ports` mean? | 5 | 5 | 4 | Correct with code cites and two genuine usage examples. |
| V20 | `maxFramesPerInterval` in a generated stream | 5 | 4 | 4 | Correct, with the config-range provenance. |
| V21 | `window_length` for a stream on a port | 5 | 5 | 4 | Correct incl. Mbps→bits/ns conversion and the eps guard. |
| V22 | `hotspot` in `listenerEmbedding` | 4 | 3 | 4 | Mechanism right, but says previously chosen "listener **or talker**" hosts get boosted — in `listenerEmbedding` only prior *listeners* (plus passed-in listeners) are boosted. |
| V23 | `pathLists` in a stream dict | 4 | 4 | 4 | Mostly right (list of hop-sourcePort paths from `shortestPath`/`randomPath`; used by `make_default_stream_config`); misses that the last element stays the listener node name; repeats the same snippet twice. |
| V24 | `eligibleListeners` in `findAllPaths` | 5 | 4 | 4 | Exact incl. the optional restriction to a passed list. |
| V25 | `k` in `k_ary_fat_tree` | 5 | 3 | 4 | Correct (ports per switch, standard k-ary fat-tree); could quantify (k²/4 cores, k pods, ...). |

### A.3 Section A aggregates

| Group | Correctness (avg /5) | Completeness (avg /5) | Understandability (avg /5) |
|---|---:|---:|---:|
| Functions (25) | **3.88** | 3.16 | 3.76 |
| Variables (25) | **4.68** | 3.40 | 4.08 |
| **All 50** | **4.28** | **3.28** | **3.92** |

**Distribution and hallucination analysis (all 50):**
- Correctness ≥ 4 (essentially reliable): **40/50 = 80 %** (functions 17/25, variables 23/25).
- Substantially wrong / hallucinated (correctness ≤ 1): **3/50 = 6 %** — F2 (false "function does not exist"), F5 (wrong function + fabricated usage), F9 (wrong units + wrong conversion story).
- Answers with a notable factual slip (correctness 2–3): 7/50 — F6, F7, F8, F11, F13, V5, V8.
- Recurrent error patterns: (a) **invented usage snippets** that would not run (F5–F8: missing constructor args, dicts passed where graphs/objects are required); (b) **unit hallucinations** (F9 "bps", F13 "kbps") despite explicit docstrings; (c) **array-shape errors** in the vectorized evaluator code (F11 "8 planes", V8 5-D shape); (d) **cross-module routing failure** when the module is not named (F2 vs. F3 shows the same question succeeding once the module is specified).
- Understandability was never the bottleneck (min 2, mode 4): answers are well-structured; the risk is confident wrongness, not confusion.
- Recorded latencies where present: 41–375 s per query (median of logged values ≈ 112 s).

---

## B. Find an existing functionality (25 queries)

Ground-truth target functions were determined from the three source files. "Keyword" states whether the query's wording appears in the target's **name**, its **docstring/description**, or only as a **synonym**.

| # | Query (abridged) | Funcs needed | Keyword | Answer given | Level | Notes |
|---|---|---:|---|---|:---:|---|
| 1 | Build k-ary fat-tree topology | 1 | name | `k_ary_fat_tree` | 1 | Exact. |
| 2 | Assign shortest routing path to a stream | 1 | name | `shortestPath` | 1 | Exact. |
| 3 | Save a use case to a JSON file | 1 | synonym ("save"→`to_file`) | `to_file` | 1 | Exact. |
| 4 | Incrementally generate use cases (one stream at a time) | 1 | docstring | `ucGenerator` | 1 | Exact (docstring keyword). |
| 5 | Forward BFS for minimum delay budgets | 1 | name+docstring | `bfs_forward` | 1 | Exact. |
| 6 | Force/exclude ports when configuring a stream | 1 | docstring | `configure_stream_ports` | 1 | Exact. |
| 7 | Draw a graph in fat-tree layout | 1 | name | `drawFattree` | 1 | Exact. |
| 8 | Which streams are feasible for a chromosome | 1 | name | `feasible_streams` | 1 | Exact. |
| 9 | Metrics: congestion, load imbalance, path length | 1 | docstring | `compute_metrics` | 1 | Exact. |
| 10 | Generate fat-tree graph **and** draw it | 2 | name | `k_ary_fat_tree` + `drawFattree` | 1 | Both exact. |
| 11 | Window length at a port **and** per-egress hop cost | 2 | name | `window_length` + `hops_costs` | 1 | Both exact, with correct parameter descriptions. |
| 12 | Generate UC from config **and** build config matrix / custom graphs | 2 | name / docstring | `generateUC` + `build_metadata` (+`make_default_stream_config`, `configure_stream_ports`, `make_stream_config`) | 1 | Extras are genuinely relevant alternatives, correctly described. |
| 13 | Evaluate a candidate: violations + metrics | 1 | name | `evaluate` | 1 | Exact incl. return tuple. |
| 14 | with-block instance context **and** per-stream feasibility | 2 | docstring / name | `use_instance` + `feasible_streams` | 1 | Both exact; correct with-block example. |
| 15 | Macro-cycle **and** suggest corrections | 2 | name | `macro_cycle` + `suggest_corrections` | 1 | Functions exact, **but** usage flaw: `macro_cycle` is a `@property`; the suggested `instance.macro_cycle()` call would raise `TypeError`. |
| 16 | Fix listeners to one host **and** host-role violations | 2 | name / synonym | `fixListeners` + `_per_stream_host_role_violations` | 1 | Both exact, with correct behavior summaries. |
| 17 | Random routing paths **and** flow-continuity violations | 2 | name | `addRouting(overwrite_config=True)` / `streamGenerator` + `_per_stream_flow_continuity_violations` | 2 | `addRouting` is a valid wrapper (random when `shortestPath` is false) and the violations function is exact, but the most direct answer `randomPath` is never named and `streamGenerator` is an unnecessary, conditional extra. |
| 18 | Generate UC, instantiate problem + metadata, evaluate | 3 | name | `generateUC` + `instantiate`/`build_metadata` + `evaluate` | 2 | Right functions, but the illustrative context-manager example `with SchedulingProblem(config_file) as problem:` is wrong — the public API is `problem.use_instance(uc)`; `TSNScheduling` itself is not a context manager. |
| 19 | Import UC from JSON, remove routing, set active instance, flow-conservation check | 3 | name | `removeRouting` + `build_metadata` + `_per_stream_flow_conservation_violations` | 3 | **Partial with a false denial:** claims "the provided code does not include a function to load a use case from a JSON file" — `import_scheduling_usecase(file_path, remove_routing=...)` and `Usecase.from_json` both exist. Also fails to find `use_instance` for setting the active instance. |
| 20 | Fat-tree use case + per-port hop costs + priority latency check | 3 | name | `makeTopology(typ='ft')` + `hops_costs` + `_per_stream_latency_violations_priority` | 2 | Parts 2–3 exact. Part 1 names the topology builder rather than the use-case generator (`generateUC` with an ft config); usable but component-level. |
| 21 | Incremental UCs + instantiate problem per UC + metrics | 3 | name | `ucGenerator` + `instantiate` + `compute_metrics` | 3 | Parts 1 & 3 exact (though the loop example passes `cls` explicitly). Part 2 mismatch: `instantiate()` *creates a new* usecase from the config file; to attach each already-generated UC the answer needed `use_instance`. |
| 22 | Shortest path + default allowed-ports config + macrocycle overlap check | 3 | name | `shortestPath` + `make_default_stream_config` + `_per_stream_macrocycle_overlap_violations` | 1 | All three exact, with a correct call example for `shortestPath`. |
| 23 | Generate UC + forward BFS + window-sequence check | 3 | name | `generateUC` + `bfs_forward` + `_per_stream_window_sequence_violations` | 1 | Retrieval exact (all three), **but** the auxiliary text contains multiple false claims: "generateUC ... returns no value" (it returns the `Usecase`), a dict-shaped graph in the `bfs_forward` usage, and "`_per_stream_window_sequence_violations(chromosome=None)` ... result will be an empty list" (it would crash). |
| 24 | Fixed talkers + macro-cycle + total per-stream violations | 3 | name | `fixTalkers` + `macro_cycle` + `per_stream_all_violations` | 1 | All three exact with correct behavior summaries. |
| 25 | Export UC to file + load as scheduling problem + suggest corrections | 3 | name | `to_file` + `import_scheduling_usecase` + `suggest_corrections` | 1 | All three exact. |

### B.1 Numerical results

**Question-level accuracy (25 questions):**

| Level | Meaning | Count | Share |
|---|---|---:|---:|
| 1 | Exactly right | **20** | **80 %** |
| 2 | Right + unnecessary/less-precise content | 3 (Q17, Q18, Q20) | 12 % |
| 3 | Partial | 2 (Q19, Q21) | 8 % |
| 4 | Wrong | 0 | 0 % |

**Complexity breakdown:**

| Funcs per question | Questions | Sub-answers | Exactly right sub-answers | Level-1 questions in bucket |
|---:|---:|---:|---:|---:|
| 1 | 10 (Q1–9, 13) | 10 | 10 | 10/10 (100 %) |
| 2 | 7 (Q10–12, 14–17) | 14 | 13 | 6/7 (86 %) |
| 3 | 8 (Q18–25) | 24 | 21 | 4/8 (50 %) |
| **Total** | **25** | **48** | **44 (91.7 %)** | 20/25 |

Sub-answer detail: 44 exact, 3 partial (Q19 "set active instance", Q20 "generate fat-tree use case", Q21 "instantiate for a given UC"), 1 wrong (Q19 false denial of a JSON loader). Accuracy degrades monotonically with the number of required functions.

**Precision / recall (function level):**
- Functions named in final answers: 54; correct/relevant: 51 → **precision ≈ 94.4 %** (misses: `streamGenerator` in Q17, `__enter__`-based example in Q18, `instantiate` in Q21).
- Required functions across all questions: 48; correctly recovered: 44 → **recall ≈ 91.7 %**.

**Keyword difficulty:** 20/25 questions had the keyword in the target's *name*, 4 in the *docstring* (Q4, Q6, Q9, Q14-part1), 1 only as a *synonym* (Q3). All 5 non-name-keyword questions were answered at Level 1, so retrieval was robust to docstring/synonym phrasing at this difficulty level; the observed failures came instead from multi-hop questions (3-function bucket) and from one cross-module routing gap (Q19's loader).

**Auxiliary-claim errors (retrieval right, guidance wrong):** 4/25 answers (Q15 property-as-call, Q18 wrong with-block, Q21 `cls` in call, Q23 three false statements). These do not change the level counts above but matter for real usability: a user pasting the Q15/Q18/Q23 snippets would hit immediate exceptions.

---

## C. Compose a missing functionality (10 requests)

All ten requests target functionality genuinely absent from the codebase (verified: no solver, GCL/NETCONF exporter, simulator, Gantt plot, jitter metric, FRER, gPTP, or GA operators exist in the three files). The pipeline correctly classified 9/10 as NO_MATCH or PARTIAL_MATCH and generated code; for gPTP (C7) it generated nothing and instead produced a gap analysis.

Scores: **Sem.** = semantic correctness of the generated code (0–5, code review); **Integ.** = integrability with the real library API (0–5); **Tests** = pass/total as recorded in the EXECUTION blocks.

| # | Request | Code generated | Sem. | Executable (code itself) | Integ. | Tests | Dominant failure cause |
|---|---|---|---:|---|---:|---:|---|
| C1 | Solver / optimizer for a feasible schedule | `solve_scheduling_problem` | 1 | Syntax ok; runtime would break (iterates over the single `Usecase` returned by `generate_scheduling_usecase` as if it were a list) | 3 | not executed | Core logic left as `# TO DO`; no optimization is actually performed. |
| C2 | GCL / 802.1Qbv gate configuration | `generate_gate_control_lists` | 2 | Runs only on a self-invented `problem` dict schema | 1 | 2/6 | Code–test contract mismatch: code expects `topology['ports'][pid]['speed']`; its own tests pass `{'ports': 2}` → `TypeError`. Also duration math yields **seconds**, not the claimed ns, and it uses a `frame_size` key instead of the library's `maxFrameSize`/`_frame_size` (no `maxFramesPerInterval`, no 1.1 tolerance). |
| C3 | Discrete-event simulation (latency + queueing) | `run_discrete_event_simulation` | 1.5 | **Crashes on the library's standard int chromosomes** (`priorities[~mask]=np.inf` → OverflowError); latent `NameError` (`queue_delay` used before assignment on the first port entry) | 2.5 | 4/6 | Not a real DES: `transmission_times` starts at 0 and is used as its own base; latency = Σ(start+hop cost) over ports. Test also expects keys (`end_to_end_latencies`) the code never returns. Good duck-typing fallbacks to `problem.hops_costs` / `usecase._frame_size`, though. |
| C4 | Gantt chart of transmission windows per port | `plot_schedule_gantt` | 3.5 | Plausible in a normal env; harness lacked matplotlib | 3.5 | 0/6 | **Environment failure** (`ModuleNotFoundError: matplotlib`), not a code fault. Latent bug: `np.max(chromosome[chromosome>0])` fails on an all-inactive schedule; tests also mock `windLengths` as 1-D where code indexes `[p, s]`. |
| C5 | Per-stream jitter | `_per_stream_jitter` | 4.5 | Runs clean | 4.5 | **4/4** | Best result. Mirrors the evaluator's `_per_stream_latency_violations_start_time` idiom exactly (inf-masking, per-stream min/max). "Jitter" as max−min start spread is a proxy definition (a static schedule has zero cycle-to-cycle jitter), but it is a defensible one and clearly documented. |
| C6 | FRER redundant / replicated paths (802.1CB) | `addRedundantPaths` | 2.5 | Fails as delivered: written with a `cls` first parameter but shipped/tested as a module-level function → argument shift `TypeError` | 3.5 | 1/4 | Good API reuse (`findAllPaths`, `pathConvenient`, `periodConvenient`) but classmethod confusion breaks everything; silently falls back to **non-disjoint** paths (contradicts FRER redundancy); mutates the stream before raising on failure. |
| C7 | gPTP time sync / clock offsets (802.1AS) | **none** | – | – | – | – | Produced an accurate, well-argued explanation that the codebase has no gPTP logic — honest, and every cited detail checks out — but for a *composition* benchmark it delivered no code, so it scores 0 on this section's task. |
| C8 | GA crossover + mutation | `crossover`, `mutation` | 3.5 | Runs clean | 3 | **5/5** | Passes, **but** one passing test enshrines a semantic bug: at `crossover_rate == 1.0` the code returns *unmodified* parents (uniform crossover should swap ~all genes), and test B4 asserts exactly that wrong behavior. Mutation/clipping otherwise fine; generic (no library coupling needed for this request). |
| C9 | NETCONF/YANG schedule export | `export_to_netconf` (two versions; duplicated query) | 2 | v2 fails at runtime: `etree.Element("<config>")` is an invalid tag name; also depends on `lxml` | 3 | 0/5 (all ERRORED) | Tests unrunnable: `mocker` fixture (pytest-mock) missing from the harness. v1 (untested) has a deeper bug: it treats `port_id_to_node` values as numeric ids and searches `self.nodes` for a matching id — but `port_id_to_node` already returns node-name strings, so `switch_name` is never found and the output would be empty. |
| C10 | Admission control (link capacity) | `check_admission_control` (+stubs) and `check_link_capacity_violations` | 1.5 / 4 | R2 runs but is hollow; R3's code is sound | 1.5 / 4.5 | 4/5 and 0/4 | R2 re-declares `bfs_forward`/`bfs_backward`/`macro_cycle` as **placeholder stubs returning {}** instead of calling the real `TSNScheduling` methods → `admissible` is always False (test B5 fails for exactly that reason). R3 is a genuinely good capacity check on `self.windLengths` vs `macro_cycle`, correctly styled as an evaluator method — but its tests import a nonexistent `Evaluator` class (the real class is `TSNConstraintEvaluator`), so all 4 error out. The existing-function pointer to `add_stream` in R1 is correct. |

### C.1 Numerical results

| Metric | Value |
|---|---|
| Requests with code generated | 9/10 (90 %) — C7 produced analysis only |
| Test suites executed (per EXECUTION blocks) | 9 (C2, C3, C4, C5, C6, C8, C9v2, C10-R2, C10-R3); C1 and C9-v1 were never tested |
| **Unit-test pass rate (raw, all executed tests)** | **20 / 45 = 44.4 %** |
| Suites fully green (harness `Success: True`) | 2/9 = 22 % (C5, C8) |
| Requests where at least one suite fully passed | 2/10 = 20 % |
| Requests whose code is semantically sound as delivered (Sem ≥ 4) | 2/10 (C5; C10-R3) — 2.5/10 counting C4's env-blocked but plausible code |
| Mean semantic correctness (9 coded requests; C10 averaged) | ≈ 2.4 / 5 |
| Mean integrability (9 coded requests; C10 averaged) | ≈ 2.9 / 5 |

**Failure-cause decomposition of the 25 failed/errored tests:**
- Genuine code bugs: 7 (C3 OverflowError ×2 … counted per failing test; C6 signature ×3; C9v2 partially; C10-R2 stub logic ×1)
- Code ↔ test contract mismatches (both generated by the same pipeline): 7 (C2 data model ×4, C3 result keys ×1, C10-R3 wrong class name ×4 — overlaps with import environment)
- Environment/harness gaps: 11 (C4 missing matplotlib ×6, C9v2 missing pytest-mock ×5)

**Interpretation.** The headline 44 % pass rate understates code quality in two places (C4, C9's tests never ran for environment reasons) and *overstates* it in one (C8 passes 5/5 while validating an incorrect crossover edge case — a reminder that self-generated "spec-based" tests are not the verified tests the evaluation criterion calls for). The most consistent weakness is **integration**: generated code repeatedly invents its own data model (C2), stubs out existing library functions instead of calling them (C10-R2), or misunderstands the library's conventions (`cls` methods in C6, `port_id_to_node` semantics in C9-v1, int-dtype chromosomes in C3). The two successes (C5, C10-R3) are precisely the cases where the code closely imitates existing evaluator idioms.

Recorded latencies: 271–913 s per request (mean of logged values ≈ 558 s).

---

## D. Write complete algorithm code (1 request)

**Request:** design a metaheuristic (e.g., a genetic algorithm) that generates candidate chromosomes and uses the evaluator's **violation and correction feedback** to converge on a feasible schedule.

**What was delivered:** two explanation blocks — (1) how `TSNScheduling` represents the problem (`build_metadata`, `config_matrix`, `customGraph`, `configure_stream_ports`), and (2) how `TSNConstraintEvaluator` computes violations and metrics (`_per_stream_host_role_violations`, `_per_stream_window_sequence_violations`, `compute_metrics`, `evaluate`) — each with verbatim code excerpts. **No algorithm was written**: no GA loop, no population/selection/crossover/mutation, no pseudo-code, no use of `suggest_corrections`, and no unit tests.

| Criterion | Result | Assessment |
|---|---|---|
| **Executable** | **Fail (no code to run)** | Zero lines of new algorithm code were produced; only library excerpts were quoted. |
| **Correctness** | Mixed — excerpts 5/5, synthesis flawed | Every quoted snippet matches the source verbatim, and the fitness-function framing (minimize violations, then optimize metrics) is sound. However, the answer contains one **major false claim**: "The provided code does not explicitly generate correction suggestions." `TSNConstraintEvaluator.suggest_corrections(Y)` exists and returns exactly the remove/add/value-correction pressures the user asked to exploit — the query even named "correction feedback". It also asserts the evaluator aggregates "two primary sources" of violations, omitting flow-conservation/continuity, priority-conflict, latency, macrocycle-overlap and talker-offset checks. |
| **Completeness** | ≈ 1.5 / 5 | The two prerequisite explanations are useful background, but the deliverable itself (the algorithm) is entirely missing. A user could not run or even sketch the GA from this answer without doing the design work themselves. |
| **Runtime** | N/A (nothing executable) | Response latency logged: 219.8 s. There is no algorithm runtime to measure and no benchmark instance was solved. |

**Section D verdict: FAIL** for the stated purpose. The response answered "explain the pieces I would use" instead of "write the algorithm", and in doing so hid the one library function (`suggest_corrections`) that the requested algorithm was supposed to be built around.

---

## Overall summary — "Can the LLM actually help someone use the library?"

Defining success per section as: A = correctness ≥ 4; B = accuracy Level 1; C = generated code whose recorded tests all pass; D = a runnable, correct algorithm delivered.

| Purpose | Success rate | One-line verdict |
|---|---:|---|
| A. Explain code / variables | 40/50 = **80 %** | Strong on direct, single-module explanation; fails on cross-module routing (F2), units (F9), tensor shapes (F11, V8), and any answer that includes an invented usage snippet (F5–F8). Completeness is the weak axis (3.28/5): many answers are correct one-liners. |
| B. Find existing functionality | 20/25 = **80 %** (0 wrong; precision 94 %, recall 92 %) | Excellent at 1–2-function lookups (16/17 exact); reliability drops on 3-function pipeline questions (4/8 exact) and one false "no such function" denial (Q19). 4 answers pair the right function with unrunnable usage code. |
| C. Compose missing functionality | tests: **44 %** pass; fully-green requests: 2/10 = **20 %** | Correctly recognizes gaps and produces plausible scaffolding, but the code routinely fails to integrate with the real API (invented schemas, stubbed helpers, wrong dtypes/signatures). Self-generated tests are themselves unreliable — some encode bugs, some can't run. |
| D. Complete algorithm | 0/1 = **0 %** | No algorithm produced; additionally denies the existence of the library's `suggest_corrections`, the exact hook the request centered on. |

**Cross-cutting findings**
1. **Retrieval ≫ generation.** The pipeline is genuinely useful as a code-navigation assistant (A, B) and near-useless, as evaluated, for autonomous code delivery (C, D). Capability degrades monotonically along explain → find → compose → full algorithm.
2. **The dominant failure mode is confident fabrication of *usage*, not of *facts*.** Function identification and code paraphrase are mostly right; hand-written call examples are wrong in ≥ 10 places across A–B (missing constructor args, properties called as methods, dicts where objects belong, "returns no value").
3. **Cross-module reasoning is fragile.** Both outright non-existence errors (A-F2, B-Q19) happened when the answer required looking in — or trusting — a module other than the one the router picked.
4. **Self-generated unit tests are not a substitute for verified tests.** In C, tests failed for three unrelated reasons (code bugs, code–test mismatches, missing harness packages) and in one case *passed while asserting wrong behavior* (C8). Any cross-LLM comparison on the C pass-rate metric should fix the test suite and environment first, then rerun each model's code against the same verified tests.
5. **Latency** (where logged) scales with task class: A ≈ 0.7–6 min, B ≈ 3 min, C ≈ 4.5–15 min, D ≈ 3.7 min per query.

**For the planned multi-LLM comparison:** the per-item tables above (50 + 25 + 10 + 1 rows) define the ground truth and grading decisions explicitly, so a second model's transcript can be scored cell-for-cell and the four headline numbers (80 %, 80 %, 44 %/20 %, 0 %) compared directly.
