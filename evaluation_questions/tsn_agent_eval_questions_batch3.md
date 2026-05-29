# TSN Scheduling Agent — Evaluation Question Bank (Batch 3)

Questions are grouped by difficulty tier and tagged with the source file(s) they target.

---

## Tier 1 — Simple (single-file)

---

### `usecase_generator.py`

**Q1**
```
Draw the topology of an imported use case (graphType == 'imported') using fat-tree layer positions, falling back to a standard NetworkX layout if the layer inference fails.
```

**Q2**
```
Given a stream, compute the frame size in bits including the tolerance factor.
```

**Q3**
```
Given a stream and a port name, return the transmission window length of that stream on that port in nanoseconds.
```

**Q4**
```
Return a flat list of all switch nodes in the current use case graph.
```

**Q5**
```
Remove a set of streams by their indices from a live use case.
```

**Q6**
```
Apply a computed schedule to a use case, updating graph port attributes and stream offsets accordingly.
```

**Q7**
```
Strip all schedule data (offsets, basetime, scheduleCycle) from a use case without touching its routing or task placement.
```

**Q8**
```
Re-apply routing to all streams in an existing use case using shortest paths, overwriting any existing pathLists.
```

**Q9**
```
Return the name string that encodes the full configuration of a use case (topology + stream parameters).
```

**Q10**
```
Serialize the topology of a use case to a JSON-compatible dictionary, including node types, port data, and edge delays.
```

---

### `tsn_scheduling_problem.py`

**Q11**
```
Given a source node name and a destination node name, return the propagation delay on the edge between them from the usecase graph.
```

**Q12**
```
Temporarily set an active use case instance using a context manager and confirm that metadata is only rebuilt once per unique instance.
```

**Q13**
```
Return the enriched stream list where talker and listener possible placements are expressed as 1-based port IDs instead of node names.
```

**Q14**
```
Compute the per-port forwarding cost vector (transmission + propagation + processing delay) for a given frame size.
```

**Q15**
```
Return the list of all link data dicts (sourcePort, destPort, sourceNode, destNode, propDelay, linkBw) for the current use case, with port names converted to 1-based IDs.
```

**Q16**
```
Given an explicit list of always-in and always-out port indices, build a config column and pruned custom graph for a stream, respecting original talker/listener placements.
```

---

### `evaluator.py`

**Q17**
```
Return the per-port utilisation vector: the fraction of the macro-cycle each port is occupied by active stream windows.
```

**Q18**
```
Return the indices of streams that are infeasible (have at least one violation across any active constraint check).
```

**Q19**
```
Given a chromosome, compute the number of active ports (path length) for each stream individually.
```

**Q20**
```
Check whether the evaluator was initialised with routing constraint checks enabled or disabled.
```

**Q21**
```
Retrieve the precomputed pairwise GCD matrix of stream intervals.
```

---

## Tier 2 — Medium (one or two files, non-trivial reasoning)

---

**Q22** *(usecase_generator.py)*
```
I want to add a new directed link between two existing switches in a live use case, including proper port speed, propagation delay, and link bandwidth from the config. Determine whether `add_link` fully handles this end-to-end or whether it has a bug; if buggy, fix it.
```

**Q23** *(usecase_generator.py)*
```
I have a use case with hotspot talker placement active. I want to know which host node has been chosen most often as a talker across all streams, and bias future stream generation away from it. Does the codebase provide this, or does it need to be written?
```

**Q24** *(tsn_scheduling_problem.py)*
```
After calling `build_metadata`, I want to retrieve the custom graph for a specific stream by its index without iterating the full stream list. Is this directly accessible, and if so, how?
```

**Q25** *(tsn_scheduling_problem.py)*
```
I need the mapping from every egress port ID to all the egress port IDs that are exactly one hop downstream of it (i.e. the ports reachable in one step via the destination node). Show whether this is precomputed and how to access it.
```

**Q26** *(evaluator.py)*
```
For a given chromosome, retrieve the raw per-stream, per-constraint violation matrix (before summing across constraints) and identify which constraint type causes the most total violations.
```

**Q27** *(evaluator.py)*
```
I want to check whether two specific streams (by index) share at least one active port in a chromosome. Does the evaluator expose this directly, or does it need to be derived from the chromosome?
```

**Q28** *(tsn_scheduling_problem.py + evaluator.py)*
```
Given a chromosome, identify all ports with a utilisation above 70% and return the list of stream indices that are active on each of those ports.
```

**Q29** *(usecase_generator.py + tsn_scheduling_problem.py)*
```
I want to generate a use case, export it to JSON, re-import it, and verify that the reconstructed port-name-to-ID mapping is identical to the original. Does the full round-trip preserve port ordering? Determine and test this.
```

---

## Tier 3 — Complex (multi-file, algorithmic or architectural)

---

**Q30** *(evaluator.py)*
```
The `suggest_corrections` function returns `remove_bias`, `add_bias`, and `gene_value_bias` for a batch of chromosomes. I need a wrapper that applies these biases directly to modify the chromosomes by one guided step (adding, removing, or adjusting one port per stream), returning the updated batch. Determine whether this application step exists; if not, implement it.
```

**Q31** *(tsn_scheduling_problem.py + evaluator.py)*
```
I want to compute, for each stream, the minimum achievable latency under the current routing (i.e. the sum of hop costs along all active ports in the chromosome). Compare this against each stream's `maxLatency` and return a per-stream slack vector. Does the codebase expose this directly, or does it need to be assembled from existing components?
```

**Q32** *(usecase_generator.py + tsn_scheduling_problem.py + evaluator.py)*
```
Design a stress-test generator: produce N use cases of increasing stream count (1 stream, 2 streams, …, N streams) using the incremental generator, build metadata for each, construct the default chromosome, evaluate it, and return the first use case size at which a routing violation appears. Implement this using existing pipeline components.
```

**Q33** *(usecase_generator.py + evaluator.py)*
```
I want to compare two placement strategies — uniform random vs. staggered — by generating one use case under each strategy and scoring them using the feasible_streams ratio as a proxy for placement quality. Determine what config changes are needed and implement the comparison function.
```

**Q34** *(tsn_scheduling_problem.py + evaluator.py)*
```
Implement a greedy stream-deactivation function: starting from a fully active chromosome, iteratively deactivate the stream whose removal most reduces the total violation count (re-evaluating after each step), stopping when the chromosome becomes fully feasible or no improvement is possible. Reuse the existing `evaluate` and `feasible_streams` calls.
```

**Q35** *(usecase_generator.py + tsn_scheduling_problem.py)*
```
Given a use case with free routing (no pathLists), I want to enumerate all simple paths between each talker and each of its listeners up to a maximum hop count, then filter out paths that fail the latency convenience check, and return only feasible candidate paths. Determine whether `findAllPaths` and `pathConvenient` together cover this, or whether a hop-limit filter is missing.
```

**Q36** *(tsn_scheduling_problem.py + evaluator.py)*
```
I need to build a port conflict graph: an undirected graph where nodes are (port, stream) pairs and edges connect pairs that share a port and have a window overlap violation. Does the codebase build this structure anywhere? If not, implement it using the macrocycle overlap and window sequence check results.
```

**Q37** *(usecase_generator.py + tsn_scheduling_problem.py + evaluator.py)*
```
I want to run a Monte Carlo feasibility sweep: for a fixed config file, generate M use cases, for each build the default chromosome and evaluate it, then report (a) the mean fraction of feasible streams, (b) the most frequently violated constraint type, and (c) the topology port that appears most often in a violated stream's active set. Implement this sweep, assembling it entirely from existing building blocks.
```

**Q38** *(evaluator.py)*
```
The evaluator caches both violation and metric results keyed by the raw byte representation of the chromosome. I want to add a cache hit-rate tracker that records how many evaluations were served from cache vs. recomputed, and exposes a summary dict on demand. Determine whether this instrumentation exists and add it if not.
```
