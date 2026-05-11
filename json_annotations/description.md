the annotation are:
* symbol-level indexed
* modular
* graph-based: separate json description and connections between them

Instead of loading a huge file-level blob, retrieval becomes:
* smaller chunks
* semantically precise
* composable
* graph searchable

Per function/method Json example:
{
  "id": "Stream.generate",
  "type": "classmethod",
  "module": "stream",
  "class": "Stream",

  "signature": "generate(config: UsecaseConfig) -> Stream",

  "summary": "Generate randomized stream from configuration ranges.",

  "inputs": {
    "config": "UsecaseConfig"
  },

  "outputs": {
    "return": "Stream"
  },

  "depends_on": [],

  "used_by": [
    "Usecase.add_stream"
  ],

  "calls": [],

  "reads_attributes": [
    "UsecaseConfig.period_range",
    "UsecaseConfig.datasize_range",
    "UsecaseConfig.max_latency_range"
  ],

  "writes_attributes": [
    "Stream.interval",
    "Stream.maxFrameSize",
    "Stream.maxLatency"
  ],

  "mutates_state": false,
  "pure_function": true,

  "examples": [
    "stream = Stream.generate(config)"
  ]
}

Per class Json example:
{
  "id": "Stream",
  "module": "stream",

  "type": "dataclass",

  "summary": "Represents a TSN traffic stream.",

  "fields": {
    "interval": "Transmission interval [ns]",
    "maxLatency": "Maximum allowed latency [ns]",
    "pathLists": "Assigned routing paths"
  },

  "methods": [
    "Stream.generate",
    "Stream.window_length",
    "Stream.build_stream_graph",
    "Stream.distance_to_hosts"
  ],

  "related_classes": [
    "Usecase",
    "TopologyHandler",
    "TrafficHandler"
  ]
}

Per module/file Json:
{
  "module": "stream",

  "classes": [
    "Stream",
    "TrafficHandler"
  ],

  "functions": [],

  "imports": [
    "networkx",
    "numpy",
    "random"
  ]
}