import ast
import configparser
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Union
from dataclasses_json import DataClassJsonMixin, config    
from pathlib import Path

@dataclass
class UsecaseConfig(DataClassJsonMixin):
    # ---------------- Topology Config ----------------
    num_hosts: int = 0
    num_switches: int = 0
    port_speed: int = 0       # Mbps
    proc_delay: int = 0       # μs
    prop_delay: int = 0       # ns
    link_bw: int = 0          # Mbps
    queue_count_per_port: int = 1
    graph_config: Dict = field(default_factory=dict)

    # ---------------- Streams Config ----------------
    num_of_streams_per_instance: int = 0
    max_streams_per_node: int = 100
    period_range: Dict = field(default_factory=lambda: {"start":0,"stop":0,"step":0})
    numberframes_range: Dict = field(default_factory=lambda: {"start":0,"stop":0})
    datasize_range: Dict = field(default_factory=lambda: {"start":0,"stop":0})
    max_latency_range: Union[List[float], float] = field(default_factory=list)
    num_places_talker: int = 1
    num_places_listener: int = 1
    num_listeners_per_stream: int = 1
    routing: Dict = field(default_factory=dict)
    stride: Dict = field(default_factory=lambda: {"active": False, "step": 0, "increase_factor":0})
    staggered: Dict = field(default_factory=lambda: {"active": False, "edgep": 0.0, "podp":0.0})
    hotspot: Dict = field(default_factory=lambda: {"active": False, "increase_factor":0})
    expiration: Dict = field(default_factory=lambda: {"finite_duration": False, "start":0, "stop":0})
    generation: Dict = field(default_factory=lambda: {"incremental": False})

    # ---------------- Methods ----------------
    @classmethod
    def from_configparser(cls, cfg: configparser.RawConfigParser):
        """Load UsecaseConfig from a RawConfigParser object."""
        # Topology
        topology = cls(
            num_hosts=cfg.getint('topology_config', 'numhosts'),
            num_switches=cfg.getint('topology_config', 'numswitchs'),
            port_speed=cfg.getint('topology_config', 'portspeed'),
            proc_delay=cfg.getint('topology_config', 'procdelay'),
            prop_delay=cfg.getint('topology_config', 'propdelay'),
            link_bw=cfg.getint('topology_config', 'linkBw'),
            queue_count_per_port=cfg.getint('topology_config', 'queuecountperport'),
            graph_config=ast.literal_eval(cfg.get('topology_config', 'graphConfig'))
        )

        # Streams
        topology.num_of_streams_per_instance = cfg.getint('streams_config', 'numofstreamsperinstance')
        topology.max_streams_per_node = cfg.getint('streams_config', 'maxstrpernode')
        topology.period_range = ast.literal_eval(cfg.get('streams_config', 'period_range'))
        topology.numberframes_range = ast.literal_eval(cfg.get('streams_config', 'numberframes_range'))
        topology.datasize_range = ast.literal_eval(cfg.get('streams_config', 'datasize_range'))

        max_latency = cfg.get('streams_config', 'maxlatency_range')
        try:
            topology.max_latency_range = [float(i) for i in max_latency.split(',')]
        except Exception:
            topology.max_latency_range = float(max_latency)

        topology.num_places_talker = cfg.getint('streams_config', 'numplacestalker')
        topology.num_places_listener = cfg.getint('streams_config', 'numplaceslistener')
        topology.num_listeners_per_stream = cfg.getint('streams_config', 'numlistenersperstr')
        topology.routing = ast.literal_eval(cfg.get('streams_config', 'routing'))
        topology.stride = ast.literal_eval(cfg.get('streams_config', 'stride'))
        topology.staggered = ast.literal_eval(cfg.get('streams_config', 'staggered'))
        topology.hotspot = ast.literal_eval(cfg.get('streams_config', 'hotspot'))
        topology.expiration = ast.literal_eval(cfg.get('streams_config', 'expiration'))
        topology.generation = ast.literal_eval(cfg.get('streams_config', 'generation'))

        return topology

    def to_configparser(self) -> configparser.RawConfigParser:
        """Export UsecaseConfig to a RawConfigParser object."""
        cfg = configparser.RawConfigParser()
        cfg.add_section('topology_config')
        cfg.set('topology_config', 'numhosts', str(self.num_hosts))
        cfg.set('topology_config', 'numswitchs', str(self.num_switches))
        cfg.set('topology_config', 'portspeed', str(self.port_speed))
        cfg.set('topology_config', 'procdelay', str(self.proc_delay))
        cfg.set('topology_config', 'propdelay', str(self.prop_delay))
        cfg.set('topology_config', 'linkBw', str(self.link_bw))
        cfg.set('topology_config', 'queuecountperport', str(self.queue_count_per_port))
        cfg.set('topology_config', 'graphConfig', str(self.graph_config))

        cfg.add_section('streams_config')
        cfg.set('streams_config', 'numofstreamsperinstance', str(self.num_of_streams_per_instance))
        cfg.set('streams_config', 'maxstrpernode', str(self.max_streams_per_node))
        cfg.set('streams_config', 'period_range', str(self.period_range))
        cfg.set('streams_config', 'numberframes_range', str(self.numberframes_range))
        cfg.set('streams_config', 'datasize_range', str(self.datasize_range))

        if isinstance(self.max_latency_range, list):
            cfg.set('streams_config', 'maxlatency_range', ','.join(str(v) for v in self.max_latency_range))
        else:
            cfg.set('streams_config', 'maxlatency_range', str(self.max_latency_range))

        cfg.set('streams_config', 'numplacestalker', str(self.num_places_talker))
        cfg.set('streams_config', 'numplaceslistener', str(self.num_places_listener))
        cfg.set('streams_config', 'numlistenersperstr', str(self.num_listeners_per_stream))
        cfg.set('streams_config', 'routing', str(self.routing))
        cfg.set('streams_config', 'stride', str(self.stride))
        cfg.set('streams_config', 'staggered', str(self.staggered))
        cfg.set('streams_config', 'hotspot', str(self.hotspot))
        cfg.set('streams_config', 'expiration', str(self.expiration))
        cfg.set('streams_config', 'generation', str(self.generation))

        return cfg
    
    # ---------------- Dataset CSV Loader ----------------
    @classmethod
    def from_dataset_csv(cls, csv_file: str, output_dir: str = "generated_configs_v2") -> List["UsecaseConfig"]:
        """
        Load multiple UsecaseConfig objects from a CSV dataset.
        
        Each row must contain:
        experiment_id,N_streams,k,topology_type,max_latency,payload,period,placement_strategy,ga_mode
        """
        configs = []
        #Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map row values to proper types
                N_streams = int(row["N_streams"])
                k = int(row.get("k", 4))
                topology_type = row["topology_type"]
                max_latency = float(row["max_latency"])
                payload = int(row["payload"])
                period_us = int(row["period"])
                placement_strategy = row.get("placement_strategy", "random")
                ga_mode = row.get("ga_mode", "joint")

                # ---------------- Topology Config ----------------
                graph_cfg = {}
                if topology_type == "fat-tree":
                    graph_cfg = {"typ": "ft", "k": k}
                elif topology_type == "er":
                    graph_cfg = {"typ": "er", "prob": 0.3, "numcoreswitches": 3, "minnodedegreepersw": 3}
                elif topology_type == "ba":
                    graph_cfg = {"typ": "ba", "numEdgN": 3, "numcoreswitches": 3, "minnodedegreepersw": 3}
                else:
                    raise ValueError(f"Unknown topology type: {topology_type}")

                # ---------------- Streams Config ----------------
                period_range = {"start": period_us, "stop": period_us + 1, "step": 1}
                datasize_range = {"start": payload, "stop": payload + 1, "step": 1}
                numberframes_range = {"start": 1, "stop": 3}
                max_latency_range = max_latency

                # Placement strategy flags
                stride = {"active": False, "step": 5, "increase_factor": 50}
                staggered = {"active": False, "edgep": 0.74, "podp": 0.25}
                hotspot = {"active": False, "increase_factor": 10}

                if placement_strategy in ["stride", "staggered", "hotspot"]:
                    locals()[placement_strategy]["active"] = True

                # Routing and talker/listener
                num_places_talker = 1
                num_places_listener = 1
                fix_path = False
                if ga_mode == "joint":
                    fix_path = False
                    num_places_talker = 3
                    num_places_listener = 3
                elif ga_mode == "routing+scheduling":
                    fix_path = False
                    num_places_talker = 1
                    num_places_listener = 1
                elif ga_mode == "scheduling-only":
                    fix_path = True
                    num_places_talker = 1
                    num_places_listener = 1

                routing = {"fixPath": fix_path, "shortestPath": True}

                # Create instance
                cfg_obj = cls(
                    num_hosts=0,
                    num_switches=0,
                    port_speed=1000,
                    proc_delay=40,
                    prop_delay=10,
                    link_bw=1000,
                    queue_count_per_port=8,
                    graph_config=graph_cfg,
                    num_of_streams_per_instance=N_streams,
                    period_range=period_range,
                    datasize_range=datasize_range,
                    numberframes_range=numberframes_range,
                    max_latency_range=max_latency_range,
                    num_places_talker=num_places_talker,
                    num_places_listener=num_places_listener,
                    num_listeners_per_stream=1,
                    routing=routing,
                    stride=stride,
                    staggered=staggered,
                    hotspot=hotspot
                )

                # Optional: save individual config to file
                #cfg_obj.to_file(Path(output_dir) / f"config_exp{row['experiment_id']}.cfg")

                configs.append(cfg_obj)

        return configs
    
    def get_config(self):
        """
        Generate a concise deterministic configuration string.
        """

        def flatten(value):
            if isinstance(value, dict):
                return "-".join(
                    flatten(v) for _, v in sorted(value.items())
                )

            if isinstance(value, list):
                return "-".join(map(str, value))

            return str(value)

        # ---------------- Topology ----------------
        topology_values = []

        topology_values.append(flatten(self.graph_config))

        topology_values.extend([
            str(self.num_hosts),
            str(self.num_switches),
            str(self.port_speed),
            str(self.prop_delay),
            str(self.proc_delay),
            str(self.link_bw),
            str(self.queue_count_per_port)
        ])

        topology_str = "-".join(topology_values)

        # ---------------- Streams ----------------
        stream_values = []

        stream_values.extend([
            str(self.num_of_streams_per_instance),
            str(self.max_streams_per_node),
            str(self.num_places_talker),
            str(self.num_places_listener),
            str(self.num_listeners_per_stream)
        ])

        for cfg in [
            self.routing,
            self.stride,
            self.staggered,
            self.hotspot,
            self.expiration,
            self.generation
        ]:
            stream_values.append(flatten(cfg))

        stream_values.append(flatten(self.period_range))
        stream_values.append(flatten(self.numberframes_range))
        stream_values.append(flatten(self.datasize_range))
        stream_values.append(flatten(self.max_latency_range))

        stream_str = "-".join(stream_values)

        return f"{topology_str}|{stream_str}"

    # ---------------- File Helpers ----------------
    @classmethod
    def from_file(cls, file_path: str) -> "UsecaseConfig":
        """Load UsecaseConfig from a .ini file."""
        cfg = configparser.RawConfigParser(inline_comment_prefixes="#", allow_no_value=True)
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        cfg.read(file_path)
        return cls.from_configparser(cfg)

    def to_file(self, file_path: str):
        """Save UsecaseConfig to a .ini file."""
        cfg = self.to_configparser()
        with open(file_path, 'w') as f:
            cfg.write(f)
        print(f"UsecaseConfig saved to {file_path}")
        return file_path
    