import numpy as np

from utils import require_instance
from evaluator import TSNConstraintEvaluator



class InstanceContext:
    """
    Context manager to temporarily set the current usecase (instance).
    It will call parent.build_metadata() once per usecase (by id).
    """
    def __init__(self, parent, instance, rebuild=False):
        self.parent = parent
        self.instance = instance
        self.rebuild = rebuild

    def __enter__(self):
        self.parent.usecase = self.instance

        # create to track built usecases (store ids to avoid relying on hashability)
        if not hasattr(self.parent, "_metadata_built_for"):
            self.parent._metadata_built_for = set()

        uid = id(self.instance)
        if uid not in self.parent._metadata_built_for or self.rebuild:
            self.parent.build_metadata()
            self.parent._metadata_built_for.add(uid)

        return self.parent

    def __exit__(self, exc_type, exc_value, traceback):
        self.parent.usecase = None
        # do NOT remove from _metadata_built_for; metadata is cached


class TSNScheduling():
    """
    Cleaned TSN scheduling problem class (step 1).
    - uses lists/dicts for streams/ports
    - metadata is built lazily and cached per-instance (usecase)
    """

    def __init__(self, usecase_specifications, ContextHandler=InstanceContext, **kwargs):
        # core helpers
        self.config_file = usecase_specifications

        # instance & context
        self.usecase = None
        self.ContextHandler = ContextHandler

        # metadata (initialized per-instance in build_metadata)
        self.config_matrix = None
        self.compiled_custom_graphs = None

    # -------------------------
    # metadata
    # -------------------------
    def build_metadata(self):
        """
        Build config_matrix and attach customGraph to each stream dict.
        - Expects self.usecase to be set.
        - Uses make_default_stream_config(s_idx, stream) which returns (config_column, custom_graph).
        """
        if not self.usecase:
            return self.no_instance_assigned()

        # allocate matrix
        self.config_matrix = np.zeros((self.n_ports, self.n_streams), dtype=int)

        # iterate streams in the raw usecase (list of dicts)
        for s_idx, stream in enumerate(self.usecase.streams):
            config_column, custom_graph = self.make_default_stream_config(s_idx)

            # sanity check: config_column must match number of ports
            if len(config_column) != self.n_ports:
                raise ValueError(
                    f"config column length ({len(config_column)}) != n_ports ({self.n_ports})"
                )

            self.config_matrix[:, s_idx] = config_column
            # attach custom graph to the stream dict so downstream code can use it
            stream["customGraph"] = custom_graph

        self.compiled_custom_graphs = self.compile_custom_graphs()
        # search space calculation
        self.slack = self._precompute_search_spaces()
            
        # After metadata is built, initialize Problem
        self.initialize_problem()

    def initialize_problem(self):
        n_var = self.n_ports * self.n_streams
        fr, ftp = self.get_task()
        check_routing = not (fr and ftp)
        self.evaluator = TSNConstraintEvaluator(self, check_routing=check_routing, check_task_placement=not ftp)
    
    # -------------------------
    # context helper
    # -------------------------
    def use_instance(self, instance, rebuild=False):
        """Return a context manager that sets .usecase to instance inside a with-block."""
        return self.ContextHandler(self, instance, rebuild)

    def no_instance_assigned(self):
        raise ValueError("No current instance is set! Use a 'with problem.use_instance(instance):' block.")
