from dataclasses import dataclass, field
from typing import Optional, List
from itertools import count
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

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
    
    start_ms: int
    end_ms: int
    period_ms: float
    queue_id: Optional[int] = None
    stream_id: Optional[int] = None
    # unique slot definition id
    index: int = field(default_factory=lambda: next(Slot._counter))

    # -----------------------------
    # Derived properties
    # -----------------------------
    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    # -----------------------------
    # Ordering logic
    # -----------------------------
    def __lt__(self, other: "Slot") -> bool:
        return self.end_ms <= other.start_ms

    # -----------------------------
    # Conflict logic
    # -----------------------------
    def conflicts_with(self, other: "Slot", touching_allowed: bool = True) -> bool:
        if touching_allowed:
            return self.start_ms < other.end_ms and self.end_ms > other.start_ms
        else:
            return self.start_ms <= other.end_ms and self.end_ms >= other.start_ms
            
    # -----------------------------
    # Occurrence expansion
    # -----------------------------
    def all_occurrences(self, schedule_cycle_ms: int) -> List["Slot"]:
        """
        Expand this periodic slot into all occurrences
        within one schedule cycle.

        All occurrences share the same index.
        """

        if self.period_ms <= 0:
            return []

        occurrences: List[Slot] = []

        k_max = math.floor(
            (schedule_cycle_ms - self.start_ms) / self.period_ms
        )

        for k in range(max(0, k_max + 1)):
            start = self.start_ms + k * self.period_ms
            end = self.end_ms + k * self.period_ms

            if end > schedule_cycle_ms:
                continue

            occurrences.append(
                Slot(
                    index=self.index,          # preserve same definition id
                    start_ms=start,
                    end_ms=end,
                    period_ms=self.period_ms,
                    queue_id=self.queue_id,
                    stream_id=self.stream_id,
                )
            )
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
    class TransmissionWindow:
        """A concrete, non-periodic transmission window"""
        start_ms: int
        end_ms: int
        queue_id: Optional[int] = None
        stream_sequence: Optional[List[int]] = None  # tracks all contributing streams

        @property
        def duration_ms(self):
            return self.end_ms - self.start_ms

    # ----------------------------
    # Schedule constructor
    # ----------------------------
    def __init__(self, cycle_ms: int, basetime: Optional[int] = None,
                 capacity: Optional[int] = None, port_id: Optional[int] = None,
                 iterable: Optional[List] = None):
        """
        Initialize the Schedule.
        :param cycle_ms: schedule cycle in ms
        :param basetime: optional reference start time
        :param capacity: optional max number of concurrent slots
        :param port_id: optional port identifier
        :param iterable: optional initial list of Slot objects
        """
        super().__init__(iterable or [])
        self.cycle_ms = cycle_ms
        self.basetime = basetime
        self.capacity = capacity
        self.port_id = port_id

    # ----------------------------
    # Utility properties
    # ----------------------------
    @property
    def total_occupied_time(self) -> int:
        """Return total time occupied by compiled transmission windows"""
        return sum(w.duration_ms for w in self.compile())

    @property
    def utilization(self) -> float:
        """Return fraction of cycle occupied"""
        if not self.cycle_ms:
            return 0.0
        return self.total_occupied_time / self.cycle_ms

    # ----------------------------
    # Conflict checking
    # ----------------------------
    def validate_conflicts(self, touching_allowed: bool = True) -> List[tuple]:
        """
        Check all occurrences of all slots for conflicts.
        Returns a list of tuples of conflicting Slot objects.
        """
        conflicts = []
        occurrences = []

        for slot in self:
            occurrences.extend(slot.all_occurrences(self.cycle_ms))

        # Sort by start time
        occurrences.sort(key=lambda s: s.start_ms)

        # Check pairwise for conflicts
        for i in range(len(occurrences) - 1):
            if occurrences[i].conflicts_with(occurrences[i+1], touching_allowed):
                conflicts.append((occurrences[i], occurrences[i+1]))

        return conflicts

    # ----------------------------
    # Compile schedule
    # ----------------------------
    def compile(self) -> List["Schedule.TransmissionWindow"]:
        """
        Expand all periodic slots into their occurrences and merge overlapping/touching windows.
        Returns a list of TransmissionWindow objects (non-periodic).
        """
        occurrences = []

        for slot in self:
            occurrences.extend(slot.all_occurrences(self.cycle_ms))

        occurrences.sort(key=lambda s: s.start_ms)

        compiled: List[Schedule.TransmissionWindow] = []

        for occ in occurrences:
            if not compiled:
                compiled.append(self.TransmissionWindow(
                    start_ms=int(occ.start_ms),
                    end_ms=int(occ.end_ms),
                    queue_id=occ.queue_id,
                    stream_sequence=[occ.stream_id] if occ.stream_id is not None else []
                ))
            else:
                last = compiled[-1]
                if occ.start_ms <= last.end_ms:  # overlapping or touching
                    last.end_ms = max(last.end_ms, occ.end_ms)
                    if occ.stream_id is not None:
                        if last.stream_sequence is None:
                            last.stream_sequence = [occ.stream_id]
                        elif occ.stream_id not in last.stream_sequence:
                            last.stream_sequence.append(occ.stream_id)
                else:
                    compiled.append(self.TransmissionWindow(
                        start_ms=int(occ.start_ms),
                        end_ms=int(occ.end_ms),
                        queue_id=occ.queue_id,
                        stream_sequence=[occ.stream_id] if occ.stream_id is not None else []
                    ))
        return compiled

    def plot(self, tick: float = 1.0, span: Optional[int] = None, label: bool = True):
        """
        Visualize the Schedule using individual slot occurrences (not compiled windows).
    
        Parameters
        ----------
        tick : float, optional
            Time unit in same units as start_ms/end_ms (default: 1.0)
        span : int, optional
            Total number of ticks to display. If None, auto-computed from cycle_ms.
        label : bool, optional
            Annotate bars with stream indices.
        """
        # Expand all slots to their occurrences
        occurrences = []
        for slot in self:
            occurrences.extend(slot.all_occurrences(self.cycle_ms))
    
        if span is None:
            span = int(self.cycle_ms / tick)
    
        # Gather all streams for coloring
        all_stream_ids = sorted({s.stream_id for s in occurrences if s.stream_id is not None})
        stream_to_color = {sid: i for i, sid in enumerate(all_stream_ids)}
    
        # Modern Matplotlib colormap
        cmap = plt.get_cmap("plasma", len(all_stream_ids))
    
        # Gather all queues/ports
        all_ports = sorted({s.queue_id for s in occurrences if s.queue_id is not None})
        port_to_y = {port: i for i, port in enumerate(all_ports)}
    
        fig, ax = plt.subplots(figsize=(20, max(5, len(all_ports))))
    
        # Plot each slot occurrence
        for s in occurrences:
            y = port_to_y[s.queue_id] if s.queue_id is not None else 0
            start_ticks = s.start_ms / tick
            width_ticks = s.duration_ms / tick
    
            # Color per stream
            if s.stream_id is not None:
                color_idx = stream_to_color[s.stream_id]
            else:
                color_idx = 0
            color = cmap(color_idx / max(1, len(all_stream_ids)-1))
    
            # Draw the horizontal bar
            ax.barh(
                y=y,
                width=width_ticks,
                left=start_ticks,
                height=0.8,
                color=color,
                edgecolor="black",
                linewidth=1,
            )
    
            # Annotate with stream ID
            if label and s.stream_id is not None:
                ax.text(
                    start_ticks + width_ticks / 2,
                    y,
                    str(s.stream_id),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                )
    
        # Axes labels and limits
        ax.set_xlabel(f"Time (ticks, tick={tick})")
        ax.set_ylabel("Ports")
        ax.set_yticks(range(len(all_ports)))
        ax.set_yticklabels([f"Port {port}" for port in all_ports])
        ax.set_xlim(0, span)
        ax.set_title(f"Schedule visualization (cycle={self.cycle_ms} ms)")
    
        # Legend: one color per stream
        legend_patches = [
            mpatches.Patch(
                color=cmap(i / max(1, len(all_stream_ids)-1)),
                label=f"Stream {sid}"
            )
            for i, sid in enumerate(all_stream_ids)
        ]
        ax.legend(handles=legend_patches, title="Streams", bbox_to_anchor=(1.05, 1), loc='upper left')
    
        plt.tight_layout()
        return fig
            