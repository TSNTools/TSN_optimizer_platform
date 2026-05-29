| ID | Difficulty | Required Modules | Question |
|---:|---:|---|---|
| Q001 | 1 | Usecase | What units are used for time, speed, bandwidth, and data size in `UsecaseGenerator` outputs? |
| Q002 | 1 | Usecase | What does `UsecaseGenerator.generateUC(configFile)` do, and what object does it return? |
| Q003 | 1 | Usecase | In `UsecaseGenerator.streamGenerator`, when is a fixed path generated for a stream? |
| Q004 | 1 | Usecase | What fields are included in the dictionary returned by `UsecaseGenerator.generateStream`? |
| Q005 | 1 | Usecase | How does `UsecaseGenerator.talkerEmbedding` select possible talker placements when no stride or staggering is active? |
| Q006 | 1 | Usecase | What is the purpose of the `hotspot` option in talker/listener embedding? |
| Q007 | 1 | Usecase | What does `UsecaseGenerator.shortestPath` add to a stream? |
| Q008 | 1 | Usecase | What is the difference between `shortestPath` and `randomPath` in the usecase generator? |
| Q009 | 1 | Usecase | What does `UsecaseGenerator.pathConvenient` check? |
| Q010 | 1 | Usecase | What does `UsecaseGenerator.periodConvenient` check? |
| Q011 | 1 | Usecase | What topology is generated when `graphConfig["typ"] == "ft"`? |
| Q012 | 1 | Usecase | What node types are assigned during topology generation? |
| Q013 | 1 | Usecase | What information is stored in `Usecase.ports`? |
| Q014 | 1 | Usecase | What does `Usecase.removeRouting()` modify? |
| Q015 | 1 | Usecase | What does `Usecase.fixTalkers()` do to each stream? |
| Q016 | 1 | Usecase | What does `Usecase.fixListeners()` do to each stream? |
| Q017 | 1 | Usecase | What does `Usecase._frame_size(stream_idx)` compute? |
| Q018 | 1 | Usecase | What does `Usecase.window_length(stream_idx, port_name)` compute? |
| Q019 | 1 | Usecase | What does `Usecase.to_json()` exclude from each stream? |
| Q020 | 1 | Usecase | What regions are produced by `Usecase.generate_fattree_regions(k)`? |
| Q021 | 1 | Scheduling | What is the purpose of the `require_instance` decorator? |
| Q022 | 1 | Scheduling | What does `InstanceContext.__enter__` do when entering a `with problem.use_instance(instance)` block? |
| Q023 | 1 | Scheduling | Why does `InstanceContext` store IDs in `_metadata_built_for`? |
| Q024 | 1 | Scheduling | What does `TSNScheduling.instantiate(export=False)` return? |
| Q025 | 1 | Scheduling | What does `TSNScheduling.import_scheduling_usecase` optionally do after importing JSON? |
| Q026 | 1 | Scheduling | What is stored in `TSNScheduling.config_matrix`? |
| Q027 | 1 | Scheduling | What does `TSNScheduling.build_metadata()` attach to each stream? |
| Q028 | 1 | Scheduling | How are port names converted to port IDs in `TSNScheduling.port_name_to_id`? |
| Q029 | 1 | Scheduling | What does the `TSNScheduling.ports` property return? |
| Q030 | 1 | Scheduling | What is the purpose of `TSNScheduling.node_to_port_id`? |
| Q031 | 1 | Scheduling | How is the macro-cycle computed in `TSNScheduling.macro_cycle`? |
| Q032 | 1 | Scheduling | What does `TSNScheduling.streams` change compared to `usecase.streams`? |
| Q033 | 1 | Scheduling | What does `TSNScheduling.configure_stream_ports` return? |
| Q034 | 1 | Scheduling | What does `TSNScheduling.get_concurrent_ports(..., typ="switch")` return? |
| Q035 | 1 | Scheduling | What is the purpose of `TSNScheduling.make_default_stream_config`? |
| Q036 | 1 | Scheduling | What does `TSNScheduling.hops_costs(frame_size)` compute? |
| Q037 | 1 | Scheduling | What does `TSNScheduling.bfs_forward` compute? |
| Q038 | 1 | Scheduling | What does `TSNScheduling.bfs_backward` compute? |
| Q039 | 1 | Scheduling | What does `TSNScheduling.network_borders(graph)` return? |
| Q040 | 1 | Scheduling | What is the purpose of `TSNScheduling.plot_all_stream_custom_graphs`? |
| Q041 | 1 | Evaluator | What are the two scheduling encodings represented by `ScheduleEncodingType`? |
| Q042 | 1 | Evaluator | What default penalty weights are defined in `TSNConstraintEvaluator.__init__`? |
| Q043 | 1 | Evaluator | How does `TSNConstraintEvaluator` decide which constraint checks are active? |
| Q044 | 1 | Evaluator | What masks are precomputed for talkers and listeners? |
| Q045 | 1 | Evaluator | What is `self.windLengths` and how is it computed? |
| Q046 | 1 | Evaluator | What is the purpose of `self.next_hop_matrix`? |
| Q047 | 1 | Evaluator | What does `_per_stream_host_role_violations` return? |
| Q048 | 1 | Evaluator | What does `_per_stream_flow_conservation_violations` check? |
| Q049 | 1 | Evaluator | What does `_per_stream_flow_continuity_violations` check? |
| Q050 | 1 | Evaluator | What does `_per_stream_priority_conflicts` detect? |
| Q051 | 1 | Evaluator | How does `_per_stream_latency_violations_priority` estimate latency? |
| Q052 | 1 | Evaluator | What does `_per_stream_latency_violations_start_time` check? |
| Q053 | 1 | Evaluator | What does `_per_stream_talker_offset_violations` check? |
| Q054 | 1 | Evaluator | What condition is used to detect macro-cycle overlaps? |
| Q055 | 1 | Evaluator | What two constraints are handled by `_per_stream_window_sequence_violations`? |
| Q056 | 1 | Evaluator | What does `per_stream_all_violations(chromosome)` return? |
| Q057 | 1 | Evaluator | What does `feasible_streams(chromosome)` return? |
| Q058 | 1 | Evaluator | What metrics are returned by `compute_metrics`? |
| Q059 | 1 | Evaluator | What does `evaluate(chromosome)` return? |
| Q060 | 1 | Evaluator | What are `remove_bias`, `add_bias`, and `gene_value_bias` in `suggest_corrections` intended for? |
| Q061 | 2 | Scheduling + Usecase | Explain the full flow from calling `TSNScheduling.instantiate()` to having a usecase with generated streams. |
| Q062 | 2 | Scheduling + Usecase | How does `TSNScheduling.build_metadata()` rely on `Usecase.streams` and `Usecase.graph`? |
| Q063 | 2 | Scheduling + Usecase | How are ports represented differently in `Usecase.ports` and `TSNScheduling.ports`? |
| Q064 | 2 | Scheduling + Usecase | Why does `TSNScheduling.streams` convert host names in `tlkrPssblPlcmnt` to port IDs? |
| Q065 | 2 | Scheduling + Usecase | How does `Usecase.removeRouting()` affect `TSNScheduling.initialize_problem()`? |
| Q066 | 2 | Scheduling + Usecase | How does `Usecase.fixTalkers()` affect the default stream configuration built by `TSNScheduling.make_default_stream_config`? |
| Q067 | 2 | Scheduling + Usecase | How does `Usecase.fixListeners()` affect listener handling in `make_default_stream_config`? |
| Q068 | 2 | Scheduling + Usecase | Explain how a fixed path in `UsecaseGenerator.shortestPath` becomes forced switch ports in `TSNScheduling.make_default_stream_config`. |
| Q069 | 2 | Scheduling + Usecase | What assumptions does `TSNScheduling.network_borders` make about host port names generated by the usecase topology? |
| Q070 | 2 | Scheduling + Usecase | Compare `Usecase._frame_size(stream_idx)` and `TSNScheduling._frame_size(stream)`. |
| Q071 | 2 | Scheduling + Usecase | How does `TSNScheduling.hops_costs(frame_size)` use graph edge and node data created by `UsecaseGenerator.makeTopology`? |
| Q072 | 2 | Scheduling + Usecase | How do `UsecaseGenerator.set_link_data` and `TSNScheduling.node_to_port_id` relate? |
| Q073 | 2 | Scheduling + Usecase | What information from the generated graph is needed by `TSNScheduling.configure_stream_ports` to prune custom graphs? |
| Q074 | 2 | Scheduling + Usecase | What happens if a listener has exactly one possible placement when `make_default_stream_config` is called? |
| Q075 | 2 | Scheduling + Usecase | How does `Usecase.to_file()` interact with `TSNScheduling.import_scheduling_usecase()`? |
| Q076 | 2 | Evaluator + Scheduling | What attributes must `TSNScheduling` provide for `TSNConstraintEvaluator.__init__` to succeed? |
| Q077 | 2 | Evaluator + Scheduling | How does `TSNScheduling.initialize_problem()` choose whether routing constraints are checked? |
| Q078 | 2 | Evaluator + Scheduling | How does `TSNScheduling.initialize_problem()` choose whether task-placement constraints are checked? |
| Q079 | 2 | Evaluator + Scheduling | How is `problem.streams` used differently from `problem.usecase.streams` in the evaluator constructor? |
| Q080 | 2 | Evaluator + Scheduling | How does the evaluator use `problem.node_to_port_id` to build routing masks? |
| Q081 | 2 | Evaluator + Scheduling | Explain how `problem.hops_costs(frame_size)` is used by scheduling constraints. |
| Q082 | 2 | Evaluator + Scheduling | Why must `build_metadata()` be called before constructing `TSNConstraintEvaluator`? |
| Q083 | 2 | Evaluator + Scheduling | What chromosome shape does the evaluator expect, and how does that relate to `TSNScheduling.config_matrix`? |
| Q084 | 2 | Evaluator + Scheduling | How does the evaluator determine talker and listener candidate ports from `TSNScheduling.node_to_port_id`? |
| Q085 | 2 | Evaluator + Scheduling | How does `TSNScheduling.macro_cycle` affect evaluator metric computation? |
| Q086 | 2 | Evaluator + Scheduling | What problem attributes are needed for `_per_stream_flow_conservation_violations` to work? |
| Q087 | 2 | Evaluator + Scheduling | How does the evaluator use the custom graph attached by `TSNScheduling.build_metadata()`? |
| Q088 | 2 | Evaluator + Scheduling | Why does `TSNConstraintEvaluator` use 1-based port IDs internally but NumPy arrays are 0-based? |
| Q089 | 2 | Evaluator + Usecase | Which stream fields generated by `UsecaseGenerator.generateStream` are required by `TSNConstraintEvaluator`? |
| Q090 | 2 | Evaluator + Usecase | How does `Usecase._frame_size` relate to evaluator window length computation? |
| Q091 | 2 | Evaluator + Usecase | How do generated stream intervals affect macro-cycle overlap checks? |
| Q092 | 2 | Evaluator + Usecase | How do generated stream max latencies affect both priority and start-time scheduling checks? |
| Q093 | 2 | Evaluator + Usecase | How does the generated topology’s port speed affect evaluator latency/window calculations? |
| Q094 | 2 | Evaluator + Usecase | How does `UsecaseGenerator.k_ary_fat_tree` influence the routing masks built by the evaluator? |
| Q095 | 2 | Evaluator + Usecase | How do fixed talker/listener placements in the usecase influence host-role violations? |
| Q096 | 2 | Evaluator + Usecase | How can a stream generated with multiple possible talkers lead to `talker_not_fixed` violations? |
| Q097 | 2 | Evaluator + Usecase | What usecase topology information is needed for evaluator backwarding detection? |
| Q098 | 2 | Evaluator + Usecase | How does a stream’s `pathLists` field affect evaluator routing feasibility indirectly? |
| Q099 | 2 | Evaluator + Usecase | What generated fields are needed for `TSNConstraintEvaluator._per_stream_talker_offset_violations`? |
| Q100 | 2 | Evaluator + Usecase | How do generated host names and switch names influence evaluator host/switch masks? |
| Q101 | 3 | Evaluator + Scheduling + Usecase | Describe the complete lifecycle from a config file to evaluating a chromosome with `TSNConstraintEvaluator.evaluate`. |
| Q102 | 3 | Evaluator + Scheduling + Usecase | What must happen before `problem.evaluator.evaluate(problem.config_matrix)` can be called safely? |
| Q103 | 3 | Evaluator + Scheduling + Usecase | How does a generated fixed route become an evaluator constraint through `customGraph`, `config_matrix`, and routing checks? |
| Q104 | 3 | Evaluator + Scheduling + Usecase | Explain how talker/listener placement constraints are generated, converted to port masks, and evaluated. |
| Q105 | 3 | Evaluator + Scheduling + Usecase | Explain how stream frame size flows from usecase generation into evaluator latency checks. |
| Q106 | 3 | Evaluator + Scheduling + Usecase | Explain how topology edge attributes generated by `UsecaseGenerator` are used in evaluator scheduling constraints. |
| Q107 | 3 | Evaluator + Scheduling + Usecase | If a usecase has fixed paths and fixed talker/listener placement, which evaluator checks are disabled by `TSNScheduling.initialize_problem()` and why? |
| Q108 | 3 | Evaluator + Scheduling + Usecase | How would changing `numplacesTalker` in the config affect stream generation, metadata building, and host-role violations? |
| Q109 | 3 | Evaluator + Scheduling + Usecase | How would changing `routing["fixPath"]` affect generated streams, custom graphs, and evaluator routing checks? |
| Q110 | 3 | Evaluator + Scheduling + Usecase | How does a fat-tree topology generated by `UsecaseGenerator.k_ary_fat_tree` propagate into evaluator masks such as `egress_mask`, `ingress_mask`, and `next_hop_matrix`? |
| Q111 | 3 | Evaluator + Scheduling + Usecase | Explain the difference between a port being forbidden by `customGraph` and a port being inactive in a chromosome. |
| Q112 | 3 | Evaluator + Scheduling + Usecase | Given a stream with one fixed talker, one fixed listener, and no fixed path, what metadata and evaluator checks are expected? |
| Q113 | 3 | Evaluator + Scheduling + Usecase | Given a stream with multiple possible talkers and listeners, how are violations computed if the chromosome activates two talker ports and no listener-adjacent port? |
| Q114 | 3 | Evaluator + Scheduling + Usecase | How do port IDs and port names flow through `Usecase`, `TSNScheduling`, and `TSNConstraintEvaluator`? |
| Q115 | 3 | Evaluator + Scheduling + Usecase | How does the evaluator distinguish host ports from switch ports, and where does that information originate? |
| Q116 | 3 | Evaluator + Scheduling + Usecase | Explain how an imported JSON usecase would be transformed into a problem instance and evaluated. |
| Q117 | 3 | Evaluator + Scheduling + Usecase | What data dependencies exist between `UsecaseGenerator.set_link_data`, `TSNScheduling.ports`, and evaluator `adjacency_mask`? |
| Q118 | 3 | Evaluator + Scheduling + Usecase | How is a chromosome interpreted differently under `ScheduleEncodingType.PRIORITY` and `ScheduleEncodingType.START_TIME`? |
| Q119 | 3 | Evaluator + Scheduling + Usecase | What changes would be needed in `TSNScheduling.initialize_problem()` to evaluate start-time encoded chromosomes? |
| Q120 | 3 | Evaluator + Scheduling + Usecase | Explain how stream intervals generated by the usecase affect macro-cycle, metric computation, and macro-cycle overlap violations. |
| Q121 | 3 | Evaluator + Scheduling + Usecase | How does a generated stream’s `maxLatency` constrain both route generation and evaluator latency violations? |
| Q122 | 3 | Evaluator + Scheduling + Usecase | How does `config_matrix` represent always-active ports, and how might it be used as an initial chromosome for the evaluator? |
| Q123 | 3 | Evaluator + Scheduling + Usecase | What happens across all modules if `Usecase.removeRouting()` is called before building metadata? |
| Q124 | 3 | Evaluator + Scheduling + Usecase | What happens across all modules if `Usecase.fixTalkers()` and `Usecase.fixListeners()` are called before metadata building? |
| Q125 | 3 | Evaluator + Scheduling + Usecase | Explain how listener ports are handled differently from talker ports across generation, metadata, and evaluation. |
| Q126 | 3 | Evaluator + Scheduling + Usecase | Why does `make_default_stream_config` convert fixed listener host placements to incoming source ports, and how does the evaluator later use listener-adjacent ports? |
| Q127 | 3 | Evaluator + Scheduling + Usecase | How does the evaluator use routing graph connectivity that originated from `UsecaseGenerator.makeTopology`? |
| Q128 | 3 | Evaluator + Scheduling + Usecase | Explain how `stream["customGraph"].edges()` is used to compute disconnected-path penalties. |
| Q129 | 3 | Evaluator + Scheduling + Usecase | How would increasing port speed in the config affect generated topology, hop costs, window lengths, and scheduling feasibility? |
| Q130 | 3 | Evaluator + Scheduling + Usecase | How would increasing frame size in stream generation affect path convenience, window lengths, hop costs, and latency violations? |
| Q131 | 3 | Evaluator + Scheduling + Usecase | Trace the data path for `propDelay` from config file to evaluator latency calculations. |
| Q132 | 3 | Evaluator + Scheduling + Usecase | Trace the data path for `procDelay` from topology generation to evaluator hop costs. |
| Q133 | 3 | Evaluator + Scheduling + Usecase | How do `Usecase.hosts`, `TSNScheduling.hosts`, and evaluator `host_mask` relate? |
| Q134 | 3 | Evaluator + Scheduling + Usecase | What assumptions about host naming are shared by all three modules? |
| Q135 | 3 | Evaluator + Scheduling + Usecase | Identify possible failure points if a topology contains nodes that do not include `"host"` or `"switch"` in their names. |
| Q136 | 3 | Evaluator + Scheduling + Usecase | Explain how a stream with `pathLists` influences `ports_in`, `ports_out`, `customGraph`, `config_matrix`, and evaluator routing checks. |
| Q137 | 3 | Evaluator + Scheduling + Usecase | What fields must be present in a manually constructed stream dictionary for the full system to work? |
| Q138 | 3 | Evaluator + Scheduling + Usecase | What fields must be present in a manually constructed port/link dictionary for the full system to work? |
| Q139 | 3 | Evaluator + Scheduling + Usecase | How does a chromosome with active host-to-host forwarding violate constraints across task placement and routing checks? |
| Q140 | 3 | Evaluator + Scheduling + Usecase | How does `suggest_corrections` use masks derived from generated topology and stream metadata to guide changes? |