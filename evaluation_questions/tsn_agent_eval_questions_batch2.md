# TSN Scheduling Agent — Evaluation Question Bank (Batch 2)

Questions are grouped by difficulty tier and tagged with the source file(s) they target.

---

## Tier 1 — Simple (single-file)

---

### `usecase_generator.py`

**Q1**
```
Generate a Waxman random graph with 10 switches.
```

**Q2**
```
Reconstruct a Usecase object from a JSON dictionary that was previously exported.
```

**Q3**
```
Remove all fixed routing (pathLists) from every stream in a use case, reverting them to free routing.
```

**Q4**
```
Fix each stream's talker to a single randomly chosen host from its current possible placements.
```

**Q5**
```
Given a node in the graph, increase its degree to a target value by connecting it to randomly chosen unconnected nodes.
```

**Q6**
```
Check whether a given cycle length is long enough for a talker to fully transmit its frame on each hop of a path.
```

**Q7**
```
Infer and reconstruct a configparser config object from a raw imported JSON use case dictionary.
```

**Q8**
```
Add a new randomly generated stream to an existing live use case instance.
```

**Q9**
```
Compute the Dijkstra distance from every host to every other node in the network graph.
```

---

### `tsn_scheduling_problem.py`

**Q10**
```
Return the transmission delay, propagation delay, and processing delay breakdown for forwarding a frame from one egress port to the next.
```

**Q11**
```
Given a stream with a fixed path, identify which switch egress ports are forced-in and which are forced-out when building its configuration column.
```

**Q12**
```
Perform a backward BFS from a listener ingress port and return the maximum remaining delay budget at every reachable egress port.
```

**Q13**
```
Return the 1-based port ID for every egress port whose source node is a given switch.
```

**Q14**
```
Plot all stream custom graphs side-by-side in a fat-tree layout with a configurable number of columns.
```

**Q15**
```
Import a use case from a JSON file and strip all routing information from its streams.
```

---

### `evaluator.py`

**Q16**
```
Given a chromosome, return a boolean array indicating which streams satisfy every active constraint.
```

**Q17**
```
Detect intra-stream sequencing violations: a downstream port is scheduled to transmit before the frame could physically arrive from its upstream port.
```

**Q18**
```
Compute, for each stream, the total number of active egress ports in the chromosome (i.e. the path length).
```

**Q19**
```
Detect talker-listener overlap violations: a host port that is active as a talker egress is also reachable as a listener ingress for the same stream.
```

**Q20**
```
Given a chromosome using start-time encoding, identify streams whose talker offset falls outside the first period interval.
```

---

## Tier 2 — Medium (one or two files, non-trivial reasoning)

---

**Q21** *(usecase_generator.py)*
```
I want to generate multiple use cases that share the same topology but differ only in their stream sets. Does the codebase provide a way to reuse an existing topology graph across generations without rebuilding it? If not, generate a function that does.
```

**Q22** *(usecase_generator.py)*
```
Generate a random path (non-shortest) for a stream, making sure each listener is assigned at most one path and that the chosen path satisfies both the latency and period convenience checks.
```

**Q23** *(tsn_scheduling_problem.py)*
```
Build an explicit (port × port) adjacency matrix where entry (i, j) is 1 if egress port i directly feeds egress port j in one hop. Determine whether this structure is precomputed anywhere in the class.
```

**Q24** *(tsn_scheduling_problem.py)*
```
I want to configure a stream so that a specific set of switch ports are permanently excluded while all others remain free. Show which existing function handles this and what arguments it requires.
```

**Q25** *(tsn_scheduling_problem.py + evaluator.py)*
```
For every pair of streams that share at least one active egress port in a given chromosome, compute the combined window length and check whether it fits within the port's macro-cycle slot. Determine whether this check exists; if not, generate it.
```

**Q26** *(evaluator.py)*
```
Compute the standard deviation of per-port load across all ports and return the index of the most congested port. Does `compute_metrics` already expose this, or does it need to be extracted separately?
```

**Q27** *(evaluator.py)*
```
I need the violation count broken down per constraint category (task placement, routing, scheduling) rather than summed. Does `evaluate` return this breakdown, or does a wrapper need to be written?
```

**Q28** *(usecase_generator.py + tsn_scheduling_problem.py)*
```
After generating a use case with a fat-tree topology, I want to verify that every host can reach every other host via at least one switch. Does the codebase perform this connectivity check anywhere, or does it need to be added?
```

---

## Tier 3 — Complex (multi-file, algorithmic or architectural)

---

**Q29** *(usecase_generator.py + tsn_scheduling_problem.py)*
```
I want to generate a sequence of incrementally growing use cases (one extra stream per step) and, for each step, rebuild only the newly affected column of the config matrix rather than rebuilding the entire metadata. Determine whether incremental metadata updates are supported; if not, implement them.
```

**Q30** *(tsn_scheduling_problem.py + evaluator.py)*
```
Implement a function that, given a chromosome with routing violations, uses the forward and backward BFS utilities to propose a corrected path for each violating stream and returns an updated chromosome with those paths applied.
```

**Q31** *(usecase_generator.py + evaluator.py)*
```
I want to score each randomly generated topology by its worst-case latency feasibility: for a given stream set and shortest-path routing, what fraction of streams can be feasibly scheduled? Design and implement this topology scoring function using the existing generator and evaluator components.
```

**Q32** *(usecase_generator.py + tsn_scheduling_problem.py + evaluator.py)*
```
Build an experiment harness that (1) generates N use cases with the same config file, (2) for each use case constructs a chromosome from the default stream configurations, (3) evaluates it, and (4) returns a summary DataFrame with per-use-case violation counts and metric scores. Determine what is missing from the codebase and implement it.
```

**Q33** *(tsn_scheduling_problem.py + evaluator.py)*
```
I need a sensitivity analysis: for each stream independently, temporarily deactivate it in the chromosome (set its column to zero) and measure how much the congestion score improves. Does anything in the codebase support per-stream knockout analysis? If not, implement it using the existing `compute_metrics` function.
```

**Q34** *(usecase_generator.py + tsn_scheduling_problem.py + evaluator.py)*
```
Design a function that detects bandwidth hotspots in the network: links (port pairs) whose combined stream load exceeds a given fraction of their link bandwidth, using the frame sizes and intervals from the use case and the port speeds from the scheduling problem. Determine whether this exists anywhere across the three files; if not, implement it.
```

**Q35** *(usecase_generator.py + tsn_scheduling_problem.py + evaluator.py)*
```
I want to run a two-phase pipeline: first find all streams that are infeasible due to latency violations using the evaluator, then use the BFS utilities from the scheduling problem to check whether any alternative (non-shortest) path would bring those streams within their latency budget, and finally update the use case routing accordingly. Implement this pipeline, reusing as much existing code as possible.
```

**Q36** *(tsn_scheduling_problem.py + evaluator.py)*
```
The evaluator caches violation and metric results keyed by chromosome bytes. I want to add a cache eviction policy so that only the K most recently evaluated chromosomes are kept in memory. Determine whether the codebase supports this and implement an LRU-bounded cache wrapper if not.
```

**Q37** *(usecase_generator.py + tsn_scheduling_problem.py)*
```
Given a fat-tree use case, partition the streams into intra-pod and inter-pod groups based on whether the talker and all listeners reside in the same pod. Then build separate custom graphs for each group, restricting inter-pod streams to use only core and aggregation switches. Determine whether this partitioning and graph restriction logic exists; if not, implement it.
```
