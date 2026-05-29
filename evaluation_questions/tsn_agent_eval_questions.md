# TSN Scheduling Agent — Evaluation Question Bank

Questions are grouped by difficulty tier and tagged with the source file(s) they target.

---

## Tier 1 — Simple (single-file)

These queries have a clear, narrow scope and should be resolved by retrieving one or two functions from a single file.

---

### `usecase_generator.py`

**Q1**
```
Generate a k=4 fat-tree topology.
```

**Q2**
```
Create an Erdos-Renyi random graph using 12 switches with a connection probability of 0.4.
```

**Q3**
```
Generate a use case incrementally, adding one stream at a time, using a given configuration file.
```

**Q4**
```
Select a random talker placement from all available hosts with uniform probability.
```

**Q5**
```
Compute the shortest path for a stream between its talker and each of its listeners.
```

**Q6**
```
Check whether a given path satisfies the maximum latency constraint for a stream.
```

**Q7**
```
Save a generated use case to a JSON file in a specified output folder.
```

**Q8**
```
Given a k-ary fat-tree, return the list of topology-aware switch regions (core region plus one region per pod).
```

---

### `tsn_scheduling_problem.py`

**Q9**
```
Compute the macro-cycle (LCM of all stream intervals) for the current use case instance.
```

**Q10**
```
For a given stream, build the default configuration column and pruned custom graph based on its talker and listener placements.
```

**Q11**
```
Perform a forward BFS from a talker egress port and return the minimum cumulative delay to every reachable egress port.
```

**Q12**
```
Return all egress ports of switches that are directly connected to listener hosts (network border ports).
```

**Q13**
```
Draw the fat-tree layout of a network graph on a matplotlib axis.
```

---

### `evaluator.py`

**Q14**
```
Given a chromosome matrix, count how many streams have zero violations across all active constraint checks.
```

**Q15**
```
Compute the congestion score, average utilisation, load imbalance, and average path length for a given chromosome.
```

**Q16**
```
Detect periodic window overlaps between pairs of streams sharing the same port over the macro-cycle.
```

**Q17**
```
Identify priority conflicts: two active streams assigned the same priority value on the same egress port.
```

---

## Tier 2 — Medium (one or two files, non-trivial reasoning)

These queries require combining multiple functions or understanding interactions between classes, but still map to a bounded region of the codebase.

---

**Q18** *(usecase_generator.py)*
```
I want to generate a Barabasi-Albert random graph with 20 switches and 2 edges per new node, then ensure every switch has at least 3 connections. Show whether the codebase already handles this minimum-degree correction.
```

**Q19** *(usecase_generator.py)*
```
Implement staggered listener placement so that listeners in the same pod as the talker are preferred, but listeners on the same edge switch get the highest probability.
```

**Q20** *(tsn_scheduling_problem.py)*
```
Force stream 0 to use a specific set of switch egress ports while excluding all host ports that are not its registered talker or listeners. Show whether a function exists for this and call it with appropriate arguments.
```

**Q21** *(tsn_scheduling_problem.py + evaluator.py)*
```
For a given chromosome, retrieve the per-port utilisation values and identify which ports exceed 80 % utilisation.
```

**Q22** *(evaluator.py)*
```
Check whether the intra-stream sequencing constraint (a frame may not leave a downstream port before it has arrived from upstream) is enforced, and describe which function handles it.
```

**Q23** *(evaluator.py)*
```
I need to evaluate a batch of chromosomes at once and obtain, for each chromosome, both the total violation count per constraint type and the performance metrics. Does the codebase support batched evaluation? If not, generate a function that does.
```

---

## Tier 3 — Complex (multi-file, algorithmic or architectural)

These queries require cross-file reasoning, understanding of the full pipeline, or generation of non-trivial new functionality.

---

**Q24** *(usecase_generator.py + tsn_scheduling_problem.py)*
```
I want to distribute streams across the network to avoid link congestion while respecting shortest-path constraints. Determine whether such functionality exists. If not, generate a function that implements this algorithm.
```

**Q25** *(usecase_generator.py + tsn_scheduling_problem.py)*
```
After importing a use case from a JSON file, I need to rebuild its metadata (config matrix and custom graphs) and then verify that every stream has a valid path from its talker to each listener in its pruned custom graph. Does the codebase cover this full pipeline end-to-end?
```

**Q26** *(tsn_scheduling_problem.py + evaluator.py)*
```
I want to repair an infeasible chromosome by identifying which streams are infeasible due to flow-conservation violations and then resetting their port activations to match the default stream configuration. Determine whether this repair logic exists; if not, generate it.
```

**Q27** *(usecase_generator.py + evaluator.py)*
```
When a new stream is added to a live use case (using `Usecase.add_stream()`), the evaluator's pre-computed masks and cost arrays become stale. Does the codebase handle incremental evaluator updates? If not, write a function that refreshes only the affected evaluator data structures.
```

**Q28** *(usecase_generator.py + tsn_scheduling_problem.py + evaluator.py)*
```
I need to rank a set of candidate topologies by the minimum achievable latency violation count across all streams, assuming shortest-path routing and priority-based scheduling. Design a pipeline that (1) generates each topology, (2) instantiates the scheduling problem, (3) constructs a feasibility-optimised chromosome, and (4) evaluates it with the constraint evaluator.
```

**Q29** *(tsn_scheduling_problem.py + evaluator.py)*
```
Implement a load-balancing post-processor: after scheduling, identify over-utilised ports (above a configurable threshold) and attempt to reroute one stream per port to an alternative path with lower utilisation, then re-evaluate the modified chromosome.
```

**Q30** *(usecase_generator.py + tsn_scheduling_problem.py + evaluator.py)*
```
Given a fat-tree topology and a set of streams with fixed shortest-path routing, compute for each pod the total bandwidth demand (sum of stream frame sizes over their intervals) and flag any pod whose internal links would be overloaded assuming uniform traffic distribution. Determine whether this analysis exists in the codebase; if not, generate it.
```
