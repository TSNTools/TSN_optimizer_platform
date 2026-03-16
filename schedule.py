from dataclasses import dataclass, field
from typing import Optional, List, Dict
from collections import defaultdict
from itertools import count
import math
    
import numpy as np

import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import matplotlib.cm as cm

from dataclasses_json import DataClassJsonMixin

MS_TO_NS = 1_000_000

@dataclass
class Slot:
    """
    A periodic slot definition.

    One Slot represents a repeating transmission window of a cyclic stream.
    Occurrences within a schedule cycle can be generated
    using `all_occurrences()`.
    
    N.B. If a slot is defined for a set of streams sent back to back, but having different periods 
    then repeating the slot and checking overlap during schedule cycle becomes harder.
    """
    # shared class-level counter
    _counter = count()
    
    start_ns: int
    end_ns: int
    period_ns: float
    stream_id: Optional[int] = None
    port_id: Optional[int] = None
    queue_id: Optional[int] = None
    # unique slot definition id
    index: int = field(default_factory=lambda: next(Slot._counter))

    # -----------------------------
    # Derived properties
    # -----------------------------
    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    # -----------------------------
    # Ordering logic
    # -----------------------------
    def __lt__(self, other: "Slot") -> bool:
        return self.end_ns <= other.start_ns

    # -----------------------------
    # Conflict logic
    # -----------------------------
    def conflicts_with(self, other: "Slot", touching_allowed: bool = True) -> bool:
        if touching_allowed:
            return self.start_ns < other.end_ns and self.end_ns > other.start_ns
        else:
            return self.start_ns <= other.end_ns and self.end_ns >= other.start_ns
            
    # -----------------------------
    # Occurrence expansion
    # -----------------------------
    def all_occurrences(self, span_ns: int) -> List["Slot"]:
        """
        Expand this periodic slot into all occurrences
        within the span.

        All occurrences share the same index.
        """

        if self.period_ns <= 0:
            return []

        occurrences: List[Slot] = []

        k_max = math.floor(
            (span_ns - self.start_ns) // self.period_ns
        )

        for k in range(max(0, k_max) + 1):
            start = int(self.start_ns + k * self.period_ns)
            end = int(self.end_ns + k * self.period_ns)

            occurrences.append(
                Slot(
                    index=self.index,          # preserve same definition id
                    start_ns=start,
                    end_ns=end,
                    period_ns=self.period_ns,
                    queue_id=self.queue_id,
                    stream_id=self.stream_id,
                    port_id=self.port_id,
                )
            )

            if end > span_ns:
                break
        return occurrences

        
class Schedule(list):
    """
    A schedule of periodic slots.
    Inherits from list, so you can append/remove Slot objects directly.
    Stores schedule metadata and provides validation and compilation utilities.
    """

    # ----------------------------
    # Internal TransmissionWindow
    # ----------------------------
    @dataclass
    class TransmissionWindow(DataClassJsonMixin):
        """A concrete, non-periodic transmission window that spams all the queues"""
        start: int
        duration: int
        state: str     
        stream_sequence: Optional[List[int]] = None  # tracks all contributing streams

        @property
        def end(self):
            return self.start+self.duration

    # ----------------------------
    # Schedule constructor
    # ----------------------------
    def __init__(self, cycle_ns: int, basetime: Optional[int] = None,
                 capacity: Optional[int] = None, iterable: Optional[List] = None):
        """
        Initialize the Schedule.
        :param cycle_ns: schedule cycle in ms
        :param basetime: optional reference start time
        :param capacity: optional max number of concurrent slots
        :param port_id: optional port identifier
        :param iterable: optional initial list of Slot objects
        """
        super().__init__(iterable or [])
        self.cycle_ns = cycle_ns
        self.basetime = basetime
        self.capacity = capacity


    # ----------------------------
    # Utility properties
    # ----------------------------
    @property
    def total_occupied_time(self) -> Dict[int, int]:
        """Return total time occupied by compiled transmission windows for each port_id"""
        return {
            port_id: sum(w.duration_ns for w in windows)
            for port_id, windows in self.compile().items()
        }

    @property
    def utilization(self) -> Dict[int, float]:
        """Return fraction of cycle occupied per port_id"""
        # Get the total occupied time for all port_ids
        occupied_times = self.total_occupied_time()
    
        if not self.cycle_ns:
            # Return 0.0 for each port if cycle_ns is 0 (avoiding division by zero)
            return {port_id: 0.0 for port_id in occupied_times}
    
        # Calculate utilization for each port_id
        return {port_id: occupied_time / self.cycle_ns for port_id, occupied_time in occupied_times.items()}

    # ----------------------------
    # Conflict checking
    # ----------------------------
    def validate_conflicts(self, touching_allowed: bool = True) -> List[tuple]:
        """
        Check all occurrences of all slots for conflicts.
        Returns a list of tuples of conflicting Slot objects.
        """
        conflicts = []
        occurrences = defaultdict(list)

        # Collect occurrences by port_id
        for slot in self:
            occurrences[slot.port_id].extend(slot.all_occurrences(self.cycle_ns))

        for port_id, port_slots in occurrences.items():
            # Sort by start time
            port_slots.sort(key=lambda s: s.start_ns)
            # Check pairwise for conflicts
            for i in range(len(port_slots) - 1):
                if port_slots[i].conflicts_with(port_slots[i+1], touching_allowed):
                    conflicts.append((port_slots[i], port_slots[i+1]))

        return conflicts

    # ----------------------------
    # Compile schedule
    # ----------------------------  
    @classmethod
    def from_usecase(cls, usecase, port_name_to_id):
    
        def transmission_window_to_slots(tw, port_name):
            slots = []
            start = int(tw["start"])
            open_queues = [i for i, ch in enumerate(tw["state"]) if ch == "o"]
            stream_idxs = [int(s) for s in tw["stream_sequence"]]
            for s_idx in stream_idxs:
                period_ns = usecase.streams[s_idx]['interval'] #* MS_TO_NS
                duration_ns = usecase.window_length(s_idx, port_name)
                for qid in open_queues:
                    slots.append(
                        Slot(
                            start_ns=start,
                            end_ns=start + duration_ns,
                            period_ns=period_ns,
                            stream_id=s_idx,
                            port_id=port_id,
                            queue_id=qid,
                        )
                    )
            return slots
    
        all_slots = defaultdict(list)
        cycle_ns = None
        for node, nodeData in usecase.graph.nodes(data=True):
            for port, portData in nodeData["ports"].items():
                port_name = f"{node}.{port}"
                port_id = port_name_to_id[port_name]
                cycle_ns = portData.get("scheduleCycle", cycle_ns)
                schedule = portData.get("schedule", [])
                for tw in schedule:
                    slots = transmission_window_to_slots(
                        tw,
                        port_name
                    )
                    for s in slots:
                        s.port_id = port_id
                        offset = s.start_ns % s.period_ns if s.period_ns else s.start_ns
                        key = (port_id, s.stream_id, s.queue_id, offset)
                        all_slots[key].append(s)

        schedule = cls(cycle_ns)
        for slots in all_slots.values():
            base = min(slots, key=lambda s: s.start_ns)
            schedule.append(base)
    
        return schedule
        
    def update_graph(self, graph, port_id_to_name):
        """
        Expand all periodic slots into their occurrences and merge overlapping/touching windows.
        Returns a list of TransmissionWindow objects (non-periodic).
        """
        occurrences = []
    
        for slot in self:
            occurrences.extend(slot.all_occurrences(self.cycle_ns))
    
        occurrences.sort(key=lambda s: s.start_ns)
    
        compiled: Dict[int, List[Schedule.TransmissionWindow]] = defaultdict(list)
    
        for occ in occurrences:
            node_name, port = port_id_to_name[occ.port_id].split('.')
            port_data = graph.nodes[node_name]['ports'][port]
            queueCount = port_data.get("queueCount", 8)
            windows = compiled[occ.port_id]
    
            # Build state string for this window
            state_list = list('C' * queueCount)
            if occ.queue_id is not None:
                state_list[occ.queue_id] = 'o'
            state = ''.join(state_list)
    
            # Create new window
            new_window = self.TransmissionWindow(
                start=int(occ.start_ns),
                duration=int(occ.end_ns - occ.start_ns),
                state=state,
                stream_sequence=[occ.stream_id] if occ.stream_id is not None else []
            )
    
            # Merge if overlapping and same state
            if windows and (occ.start_ns <= windows[-1].start + windows[-1].duration) and state == windows[-1].state:
                last = windows[-1]
                # Extend duration
                end_time = max(last.start + last.duration, new_window.start + new_window.duration)
                last.duration = end_time - last.start
                # Merge stream_sequence
                if new_window.stream_sequence:
                    if last.stream_sequence is None:
                        last.stream_sequence = new_window.stream_sequence
                    else:
                        for sid in new_window.stream_sequence:
                            if sid not in last.stream_sequence:
                                last.stream_sequence.append(sid)
            else:
                # Append window
                windows.append(new_window)
    
            # Always update the graph port
            port_data['basetime'] = self.basetime
            port_data['scheduleCycle'] = self.cycle_ns
            port_data.setdefault('schedule', [])
            # Only append **new windows** if not already in the list
            if new_window.to_dict() not in port_data['schedule']:
                port_data['schedule'].append(new_window.to_dict())
    
    def update_streams(self, streams, ports, port_id_to_name):
        """
        add schedule data to streams
        """        
        for s_idx, stream in enumerate(streams):
            n_listeners = len(stream['lstnsrPssblPlcmnt'])
            stream_slots = self.slots_by_stream(s_idx)
            stream_slots.sort()
            # talker data from first slot
            talker_slot = stream_slots[0]
            talker_port = port_id_to_name[talker_slot.port_id]
            talker_node = talker_port.split('.')[0]
            stream['offset'] = talker_slot.start_ns
            stream['tlkrPssblPlcmnt'] = [talker_node]
            stream['accLatency'] = [[] for l_idx in range(n_listeners)]
            stream['pathLists'] = [[talker_port] for l_idx in range(n_listeners)]
            
            for slot in stream_slots[1:]:
                port = port_id_to_name[slot.port_id]
                node_name = port.split('.')[0]
                pathLists = stream.get("pathLists")
                for l_idx in range(n_listeners):
                    pathList = pathLists[l_idx]
                    last = pathList[-1] # assuming talker is already there
                    if ports[last]['destNode']==node_name:
                        pathList.append(port)
                        stream['accLatency'][l_idx] = slot.end_ns - stream['offset']
            # use the last slot (forwarding to listener)
            for l_idx in range(n_listeners):
                last = pathList[-1]
                stream['lstnsrPssblPlcmnt'][l_idx] = [ports[port]['destNode']]
    
    @classmethod
    def from_compact_encoding(
        cls,
        compact_encoding,
        windLengths, stream_intervals,
        cycle_ns: Optional[int] = None,
        basetime:Optional[int] = 0,
        tt_queue_id = 0
    ):
        """
        Build a Schedule instance from a compact_encoding array.
    
        Parameters
        ----------
        compact_encoding : 2D numpy array (n_ports , n_streams)
            Nonzero value = start_time + 1
        windLengths : 2D numpy array (n_ports , n_streams)
            streams[i] = (period_ns, duration_ns)
        cycle_ns : int
            Schedule cycle
        """
        n_ports, n_streams = compact_encoding.shape

        schedule = cls(cycle_ns=cycle_ns, basetime=basetime)
        
        for port_idx in range(n_ports):
            for stream_id in range(n_streams):
                value = compact_encoding[port_idx, stream_id]
                if value != 0:
                    start_time = int(value) - 1
                    period_ns = stream_intervals[stream_id]
                    duration_ns = windLengths[port_idx, stream_id]
    
                    slot = Slot(
                        start_ns=start_time,
                        end_ns=int(start_time + duration_ns),
                        period_ns=int(period_ns),
                        stream_id=stream_id,
                        port_id=port_idx+1,
                        queue_id=tt_queue_id
                    )
                    schedule.append(slot)
        return schedule

    def to_compact_encoding(self, n_ports, n_streams):
        """
        Convert Schedule into compact_encoding array.
    
        Parameters
        ----------
        streams : list of tuples
            streams[i] = (period_ns, duration_ns)
    
        Returns
        -------
        compact_encoding : np.ndarray (n_ports x n_streams)
        """
        if not self:
            raise ValueError("Schedule is empty")
    
        compact = np.zeros((n_ports, n_streams), dtype=int)
    
        for slot in self:
            if slot.stream_id is None or slot.port_id is None: # unknown stream or port
                continue

            port_idx = slot.port_id -1
            stream_id = slot.stream_id
    
            if compact[port_idx, stream_id] != 0:
                raise ValueError(
                    f"Multiple slots found for port {port_idx}, stream {stream_id}. "
                    "Compact encoding allows only one."
                )
    
            compact[port_idx, stream_id] = int(slot.start_ns) + 1
    
        return compact

    def slots_by_port(self, port_id: int) -> List[Slot]:
        """
        Return all Slot definitions belonging to a given port.
        """
        return [slot for slot in self if slot.port_id == port_id]

    def slots_by_stream(self, stream_id: int) -> List[Slot]:
        """
        Return all Slot definitions belonging to a given stream.
        """
        return [slot for slot in self if slot.stream_id == stream_id]

    def plot(self, tick: float = 1.0, span: Optional[int] = None, label: bool = False, 
             port_ids: Optional[List[int]] = None,
             stream_ids: Optional[List[int]] = None):
        """
        Visualize schedule occurrences with one axis per port.
    
        Parameters
        ----------
        tick : float
            Time unit scaling.
        span : int
            Max time to display.
        label : bool
            Annotate bars with stream ids.
        port_id : int or None
            If provided, plot only this port.
    
        Returns
        -------
        fig
        axs
        """
    
        # Expand occurrences
        occurrences = []
        for slot in self:
            occurrences.extend(slot.all_occurrences(self.cycle_ns))

         # Filter port if requested
        if port_ids is not None:
            occurrences = [o for o in occurrences if o.port_id in port_ids]
            
        # Filter streams if requested
        if stream_ids is not None:
            stream_ids = set(stream_ids)
            occurrences = [o for o in occurrences if o.stream_id in stream_ids]
            
        if not occurrences:
            fig, ax = plt.subplots()
            ax.set_title("Empty schedule")
            return fig, [ax]
    
        if span is None:
            span = int(self.cycle_ns / tick)
    
        # Group occurrences by port
        ports = sorted({o.port_id for o in occurrences if o.port_id is not None})
    
        port_occ = {p: [] for p in ports}
        for o in occurrences:
            port_occ[o.port_id].append(o)
    
        n_ports = len(ports)
    
        fig, axs = plt.subplots(
            n_ports,
            1,
            figsize=(20, max(3, n_ports * 2)),
            sharex=True
        )
    
        if n_ports == 1:
            axs = [axs]
    
        # Stream coloring
        all_stream_ids = sorted({o.stream_id for o in occurrences if o.stream_id is not None})
        cmap = plt.get_cmap("plasma", len(all_stream_ids))
        stream_to_color = {sid: i for i, sid in enumerate(all_stream_ids)}
    
        for ax, port in zip(axs, ports):
    
            for s in port_occ[port]:
    
                start_ticks = s.start_ns / tick
                width_ticks = (s.end_ns - s.start_ns) / tick
    
                color_idx = stream_to_color.get(s.stream_id, 0)
                color = cmap(color_idx / max(1, len(all_stream_ids)-1))
    
                ax.barh(
                    y=0,
                    width=width_ticks,
                    left=start_ticks,
                    height=0.8,
                    color=color,
                    edgecolor="black"
                )
    
                if label and s.stream_id is not None:
                    ax.text(
                        start_ticks + width_ticks/2,
                        0,
                        str(s.stream_id),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9
                    )
    
            ax.set_ylabel(f"Port {port}")
            ax.set_yticks([])

        # legend
        legend_handles = []
        
        for sid in all_stream_ids:
            color_idx = stream_to_color[sid]
            color = cmap(color_idx / max(1, len(all_stream_ids)-1))
        
            legend_handles.append(
                Patch(facecolor=color, edgecolor="black", label=f"Stream {sid}")
            )
        
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            title="Streams"
        )
    
        axs[-1].set_xlabel(f"Time (tick={tick})")
        axs[-1].set_xlim(0, span)
    
        fig.suptitle(f"Schedule visualization (cycle={self.cycle_ns})")
    
        plt.tight_layout()
        #plt.close(fig)
        plt.show()
    
        return fig, axs