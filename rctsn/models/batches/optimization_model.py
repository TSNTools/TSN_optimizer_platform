import pulp
import os
from models.data_factory import DataFactory


class OptimizationModel:
    """
    OptimizationModel class for scheduling streams in a network.
    """

    def __init__(self, data_factory: DataFactory, output_path: str, time_limit: int = None):
        """
        Initialize the optimization model.

        Args:
            datafactory: DataFactory instance with all stream data
            output_path: Directory to store solver outputs
            time_limit: Maximum solver time in seconds
        """
        self.datafactory = data_factory
        self.output_path = output_path
        self.time_limit = time_limit

        # Stream categorization
        self.variable_streams = []  # Streams being optimized in current iteration
        self.fixed_streams = []  # Streams with fixed schedules from prior iterations

        # Constraint tracking for dynamic updates
        self.variable_variable_constraints = []
        self.variable_fixed_constraints = []

        # Decision variables
        self.dv_x = {}  # Start times
        self.dv_z = {}  # Path selection
        self.dv_a = {}  # Stream scheduling
        self.dv_yv = {}  # Variable-variable ordering (port level)
        self.dv_yf = {}  # Variable-fixed ordering (port level)

        # Initialize objective tracking
        self.objective_name = None

        # Create the problem and variables
        self.create_problem()
        self.create_base_variables()
        self.update_objective_function()  # Initialize with dummy objective
        self.setup_base_constraints()

    def create_problem(self):
        """Create the PuLP problem instance with the solver"""
        log_file_path = os.path.join(self.output_path, "solver_log.log")
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        self.solver = pulp.GUROBI(
            msg=True, logPath=log_file_path, timeLimit=self.time_limit
        )
        self.prob = pulp.LpProblem("Stream_Scheduling", pulp.LpMaximize)

    def create_base_variables(self):
        """Create all decision variables for the model"""
        # Start time variables
        self.dv_x = {
            (stream, path, repetition, port): pulp.LpVariable(
                f"x_{stream}_{path}_{repetition}_({port[0]},{port[1]})",
                lowBound=0,
                cat="Integer",  # Using Integer for potentially better performance with some solvers
            )
            for stream in self.datafactory.stream_dict.keys()
            for path in self.datafactory.stream_dict[stream].keys()
            if "ports" in self.datafactory.stream_dict[stream][path] # Ensure path has ports
            for port in self.datafactory.stream_dict[stream][path]["ports"]
            for repetition in self.datafactory.stream_dict[stream][path]["repetitions"]
        }

        # Path selection variables (binary)
        self.dv_z = {
            (stream, path): pulp.LpVariable(
                f"z_{stream}_{path}",
                cat="Binary",
            )
            for stream in self.datafactory.stream_dict
            for path in self.datafactory.stream_dict[stream]
        }

        # Stream scheduling variables (binary)
        self.dv_a = {
            (stream): pulp.LpVariable(
                f"a_{stream}",
                cat="Binary",
            )
            for stream in self.datafactory.stream_dict
        }

    def update_objective_function(self, streams=None):
        """
        Define or update the objective function based on specified streams.

        Args:
            streams: List of stream indices to include in the objective.
                    If None, creates a dummy objective.
        """
        # Create new objective function
        if streams is None or len(streams) == 0:
            # Create a dummy objective (constant 1) if no streams are specified
            dummy_var = pulp.LpVariable(
                "dummy_objective", lowBound=0, upBound=1, cat="Integer"
            )
            self.prob.setObjective(dummy_var)
        else:
            # Create objective using specified streams
            self.prob.setObjective(
                pulp.lpSum(self.dv_a[(stream)] for stream in streams)
            )

    def setup_base_constraints(self):
        """Set up constraints that remain constant across all iterations"""
        self.add_constraint_ns()  # Consecutive port timing
        self.add_constraint_periodic_repetitions()  # Periodic repetitions
        self.add_constraint_path_selection()  # Path selection
        self.add_constraint_max_latency()  # Maximum latency
        self.add_constraint_max_jitter()  # Jitter constraint

    def _get_effective_start_end(
        self, stream, path, port, scheduled_start_time
    ):
        """Calculates the effective start and end time of a stream transmission at a port."""
        time = self.datafactory.get_time_for_stream_port(stream, path, port)
        deviation = self.datafactory.get_deviation_for_stream_port(stream, path, port)
        
        # Get the single gamma value for the stream's path
        gamma = self.datafactory.stream_dict[stream][path]["gamma"]

        duration = time + (gamma * deviation)
        end_time = scheduled_start_time + duration

        return scheduled_start_time, end_time

    def _merge_time_ranges(
        self, sorted_effective_occupancies, min_gap_to_fit_item, epsilon=1e-9
    ):
        """
        Merges time ranges representing fixed stream occupancies to find overall blocked periods.
        A new blocked range is started if the gap between fixed occupancies is large enough
        to fit an item of size min_gap_to_fit_item.

        Args:
            sorted_effective_occupancies: List of (effective_start, effective_end) tuples for fixed streams,
                                          sorted by effective_start.
            min_gap_to_fit_item: The length of the variable stream we are trying to schedule.
                                 If a gap between fixed streams is smaller than this,
                                 the gap is considered part of the blocked range from the perspective
                                 of the variable stream.
            epsilon: Small value for floating point comparisons.
        Returns:
            List of (block_start, block_duration) tuples.
        """
        if not sorted_effective_occupancies:
            return []

        blocked_ranges = []
        # The first fixed stream occupancy forms the start of our first potential blocked range
        current_block_start, current_block_end = sorted_effective_occupancies[0]

        for i in range(1, len(sorted_effective_occupancies)):
            next_occupancy_start, next_occupancy_end = sorted_effective_occupancies[i]

            gap_between_occupancies = next_occupancy_start - current_block_end

            # If the gap is too small to fit the item, merge the next_occupancy into the current_block.
            if gap_between_occupancies < min_gap_to_fit_item - epsilon:
                current_block_end = max(current_block_end, next_occupancy_end)
            else:
                # Gap is large enough. Finalize current_block.
                duration = current_block_end - current_block_start
                if duration > epsilon:
                    blocked_ranges.append((current_block_start, duration))

                # The next_occupancy starts a new potential blocked range
                current_block_start, current_block_end = (
                    next_occupancy_start,
                    next_occupancy_end,
                )

        # Add the last processed/merged block
        duration = current_block_end - current_block_start
        if duration > epsilon:
            blocked_ranges.append((current_block_start, duration))

        return blocked_ranges

    def add_constraint_ns(self):
        """Add constraints for consecutive port timing within a path"""
        for stream in self.datafactory.stream_dict:
            for path in self.datafactory.stream_dict[stream]:
                if "ports" not in self.datafactory.stream_dict[stream][path]: continue # Skip if path has no ports
                for repetition in self.datafactory.stream_dict[stream][path][
                    "repetitions"
                ]:
                    for i, port in enumerate(
                        self.datafactory.stream_dict[stream][path]["ports"]
                    ):
                        if (
                            i
                            < len(self.datafactory.stream_dict[stream][path]["ports"])
                            - 1
                        ):
                            next_port = self.datafactory.get_next_port(
                                stream, path, port
                            )
                            # x_next = x_current + ns_current * z
                            self.prob += (
                                self.dv_x[(stream, path, repetition, port)]
                                + (
                                    (
                                        self.datafactory.stream_dict[stream][path][
                                            "ns"
                                        ][port]
                                    )
                                    * self.dv_z[(stream, path)]
                                )
                                == self.dv_x[(stream, path, repetition, next_port)]
                            )

    def add_constraint_periodic_repetitions(self):
        """Enforce periodic repetitions start time lower bound"""
        for stream in self.datafactory.stream_dict:
            for path in self.datafactory.stream_dict[stream]:
                if "ports" not in self.datafactory.stream_dict[stream][path] or not self.datafactory.stream_dict[stream][path]["ports"]: continue
                first_port = self.datafactory.stream_dict[stream][path]["ports"][0]
                period = self.datafactory.stream_dict[stream][path]["period"]
                for repetition in self.datafactory.stream_dict[stream][path][
                    "repetitions"
                ]:
                    # x_rep >= (rep-1)*Period*z
                    self.prob += (
                        self.dv_x[(stream, path, repetition, first_port)]
                        >= (repetition - 1) * period * self.dv_z[(stream, path)]
                    )

    def add_constraint_path_selection(self):
        """Path selection constraint: a stream is scheduled (a=1) iff exactly one path is selected (sum(z)=1)"""
        for stream in self.datafactory.stream_dict:
            self.prob += (
                pulp.lpSum(
                    self.dv_z[(stream, path)]
                    for path in self.datafactory.stream_dict[stream]
                )
                == self.dv_a[(stream)]
            )

    def add_constraint_max_latency(self):
        """Maximum latency constraint"""
        for stream in self.datafactory.stream_dict:
            for path in self.datafactory.stream_dict[stream]:
                if "ports" not in self.datafactory.stream_dict[stream][path] or not self.datafactory.stream_dict[stream][path]["ports"]: continue
                if "max_latency" not in self.datafactory.stream_dict[stream][path]: continue

                for repetition in self.datafactory.stream_dict[stream][path][
                    "repetitions"
                ]:
                    port = self.datafactory.stream_dict[stream][path]["ports"][-1]
                    # Get the single gamma value for the stream's path
                    gamma_val = self.datafactory.stream_dict[stream][path]["gamma"]
                    self.prob += self.dv_x[(stream, path, repetition, port)] + (
                        self.datafactory.get_time_for_stream_port(stream, path, port)
                        + self.datafactory.get_deviation_for_stream_port(
                            stream, path, port
                        )
                        * gamma_val 
                    ) * self.dv_z[(stream, path)] <= self.datafactory.stream_dict[
                        stream
                    ][
                        path
                    ][
                        "max_latency"
                    ] + (
                        self.datafactory.stream_dict[stream][path]["period"]
                        * (repetition - 1)
                    )

    def add_constraint_max_jitter(self):
        """Jitter constraint"""
        for stream in self.datafactory.stream_dict:
            for path in self.datafactory.stream_dict[stream]:
                if "ports" not in self.datafactory.stream_dict[stream][path] or not self.datafactory.stream_dict[stream][path]["ports"]: continue
                if "max_jitter" not in self.datafactory.stream_dict[stream][path]: continue

                first_port = self.datafactory.stream_dict[stream][path]["ports"][0]
                for repetition in self.datafactory.stream_dict[stream][path][
                    "repetitions"
                ]:
                    for repetition_2 in self.datafactory.stream_dict[stream][path][
                        "repetitions"
                    ]:
                        if repetition < repetition_2:
                            # Calculate the normalized offsets (arrival time minus expected periodic time)
                            offset_1 = self.dv_x[
                                (stream, path, repetition, first_port)
                            ] - (
                                (repetition - 1)
                                * self.datafactory.stream_dict[stream][path]["period"]
                            )
                            offset_2 = self.dv_x[
                                (stream, path, repetition_2, first_port)
                            ] - (
                                (repetition_2 - 1)
                                * self.datafactory.stream_dict[stream][path]["period"]
                            )

                            # First constraint: offset_1 - offset_2 >= -jitter (if path is selected)
                            # Equivalent to: offset_1 - offset_2 + M*(1-z) >= -jitter
                            # Where M is a large number
                            self.prob += (
                                offset_1
                                - offset_2
                                + self.datafactory.M * (1 - self.dv_z[(stream, path)])
                                >= -self.datafactory.stream_dict[stream][path][
                                    "max_jitter"
                                ]
                            )

                            # Second constraint: offset_1 - offset_2 <= jitter (if path is selected)
                            # Equivalent to: offset_1 - offset_2 - M*(1-z) <= jitter
                            self.prob += (
                                offset_1
                                - offset_2
                                - self.datafactory.M * (1 - self.dv_z[(stream, path)])
                                <= self.datafactory.stream_dict[stream][path][
                                    "max_jitter"
                                ]
                            )

    def set_variable_streams(self, stream_indices):
        """Set which streams are variable in the current iteration"""
        # Reset variable streams list
        self.variable_streams = []

        # Add specified streams to variable list if they aren't already fixed
        for idx in stream_indices:
            if idx not in self.fixed_streams:
                self.variable_streams.append(idx)

        print(
            f"Set {len(self.variable_streams)} streams as variable: {self.variable_streams}"
        )
        print(f"Fixed streams: {self.fixed_streams}")

    def update_variable_variable_constraints(self):
        """Add ordering constraints between variable streams"""
        # Create a unique counter for this update
        update_id = id(self.variable_streams)

        # Clear existing port-level constraints and variables
        for constraint_name in self.variable_variable_constraints:
            if constraint_name in self.prob.constraints: # Check if constraint exists before deleting
                del self.prob.constraints[constraint_name]
        self.variable_variable_constraints = []
        self.dv_yv = {}  # Reset binary variables

        # Update the objective function to focus on variable streams
        self.update_objective_function(self.variable_streams)

        # Create yv variables for ordering decisions with unique names
        self.dv_yv.update(
            {
                (
                    stream,
                    path,
                    stream_2,
                    path_2,
                    repetition,
                    repetition_2,
                    port,
                ): pulp.LpVariable(
                    # Using a simpler naming scheme for potentially shorter names
                    f"yv_{update_id}_{stream}_{path}_{stream_2}_{path_2}_{repetition}_{repetition_2}_{port[0]}_{port[1]}",
                    cat="Binary",
                )
                for stream in self.variable_streams
                for path in self.datafactory.stream_dict[stream].keys()
                if "ports" in self.datafactory.stream_dict[stream][path] # Ensure path has ports
                for repetition in self.datafactory.stream_dict[stream][path][
                    "repetitions"
                ]
                for port in self.datafactory.stream_dict[stream][path]["ports"]
                # Iterate through potentially conflicting streams on the same port
                for stream_2 in self.datafactory.get_streams_for_port(port)
                # Ensure both streams are variable and avoid duplicate pairs (stream < stream_2)
                if stream < stream_2 and stream_2 in self.variable_streams
                for path_2 in self.datafactory.get_paths_for_port_stream(stream_2, port)
                if "ports" in self.datafactory.stream_dict[stream_2][path_2] # Ensure path_2 has ports
                for repetition_2 in self.datafactory.stream_dict[stream_2][path_2][
                    "repetitions"
                ]
            }
        )

        # Add ordering constraints for variable streams sharing a port
        constraint_count = 0
        for stream in self.variable_streams:
            for path in self.datafactory.stream_dict[stream].keys():
                if "ports" not in self.datafactory.stream_dict[stream][path]: continue
                for repetition in self.datafactory.stream_dict[stream][path][
                    "repetitions"
                ]:
                    for port in self.datafactory.stream_dict[stream][path]["ports"]:
                        for stream_2 in self.datafactory.get_streams_for_port(port):
                            # Only process pairs where both are variable and stream < stream_2
                            if stream < stream_2 and stream_2 in self.variable_streams:
                                for (
                                    path_2
                                ) in self.datafactory.get_paths_for_port_stream(
                                    stream_2, port
                                ):
                                    if "ports" not in self.datafactory.stream_dict[stream_2][path_2]: continue
                                    for repetition_2 in self.datafactory.stream_dict[
                                        stream_2
                                    ][path_2]["repetitions"]:
                                        # Key for the binary variable
                                        yv_key = (
                                            stream,
                                            path,
                                            stream_2,
                                            path_2,
                                            repetition,
                                            repetition_2,
                                            port,
                                        )
                                        if yv_key not in self.dv_yv: continue # Skip if var not created

                                        # Calculate proc_dur_1 for stream 1
                                        base_time_1 = self.datafactory.get_time_for_stream_port(stream, path, port)
                                        total_deviation_1 = self.datafactory.get_deviation_for_stream_port(stream, path, port)
                                        gamma_1 = self.datafactory.stream_dict[stream][path]["gamma"]
                                        proc_dur_1 = base_time_1 + (gamma_1 * total_deviation_1)
                                        bp_1 = 0 

                                        # Calculate proc_dur_2 for stream 2
                                        base_time_2 = self.datafactory.get_time_for_stream_port(stream_2, path_2, port)
                                        total_deviation_2 = self.datafactory.get_deviation_for_stream_port(stream_2, path_2, port)
                                        gamma_2 = self.datafactory.stream_dict[stream_2][path_2]["gamma"]
                                        proc_dur_2 = base_time_2 + (gamma_2 * total_deviation_2)
                                        bp_2 = 0


                                        # Constraint 1: stream finishes before stream_2 starts (if yv=1)
                                        # x1 + proc_dur_1*z1 <= x2 - bp2*z2 + M*yv
                                        constraint_name = (
                                            f"var_var_1_{constraint_count}"
                                        )
                                        self.prob.constraints[constraint_name] = (
                                            self.dv_x[(stream, path, repetition, port)]
                                            + proc_dur_1 * self.dv_z[(stream, path)]
                                            <= self.dv_x[
                                                (
                                                    stream_2,
                                                    path_2,
                                                    repetition_2,
                                                    port,
                                                )
                                            ]
                                            - bp_2 * self.dv_z[(stream_2, path_2)]
                                            + self.datafactory.M * self.dv_yv[yv_key]
                                        )
                                        self.variable_variable_constraints.append(
                                            constraint_name
                                        )

                                        # Constraint 2: stream_2 finishes before stream starts (if yv=0)
                                        # x2 + proc_dur_2*z2 <= x1 - bp1*z1 + M*(1-yv)
                                        constraint_name = (
                                            f"var_var_2_{constraint_count}"
                                        )
                                        self.prob.constraints[
                                            constraint_name
                                        ] = self.dv_x[
                                            (
                                                stream_2,
                                                path_2,
                                                repetition_2,
                                                port,
                                            )
                                        ] + proc_dur_2 * self.dv_z[
                                            (stream_2, path_2)
                                        ] <= self.dv_x[
                                            (stream, path, repetition, port)
                                        ] - bp_1 * self.dv_z[
                                            (stream, path)
                                        ] + self.datafactory.M * (
                                            1 - self.dv_yv[yv_key]
                                        )
                                        self.variable_variable_constraints.append(
                                            constraint_name
                                        )
                                        constraint_count += 1

        if constraint_count > 0:
            print(f"Added {constraint_count} variable-variable port constraints")

    def update_variable_fixed_constraints(self):
        """
        Add ordering constraints between variable streams and fixed blocked ranges at the port level.
        """
        # Create a unique counter for this update
        update_id = id(self.variable_streams) + id(self.fixed_streams)

        # Clear existing port-level constraints and variables
        for constraint_name in self.variable_fixed_constraints:
            if constraint_name in self.prob.constraints: # Check if constraint exists
                del self.prob.constraints[constraint_name]
        self.variable_fixed_constraints = []
        self.dv_yf = {}  # Reset binary variables

        # Create binary variables and constraints for each port
        constraint_count = 0

        # For each variable stream and path that uses this port
        for var_stream in self.variable_streams:
            for var_path in self.datafactory.stream_dict[var_stream]:
                if "ports" not in self.datafactory.stream_dict[var_stream][var_path]: continue
                for port in self.datafactory.stream_dict[var_stream][var_path]["ports"]:
                    # Calculate variable_proc_dur and variable_effective_length directly
                    var_base_time = self.datafactory.get_time_for_stream_port(var_stream, var_path, port)
                    var_total_deviation = self.datafactory.get_deviation_for_stream_port(var_stream, var_path, port)
                    var_gamma = self.datafactory.stream_dict[var_stream][var_path]["gamma"]
                    variable_proc_dur = var_base_time + (var_gamma * var_total_deviation)

                    var_path_data = self.datafactory.stream_dict[var_stream][var_path]
                    var_ns_value = var_path_data["ns"].get(port, 0)
                    variable_effective_length = max(variable_proc_dur, var_ns_value)


                    # Get blocked ranges based on the effective length needed by the variable stream
                    blocked_ranges = self.calculate_blocked_ranges(
                        port, variable_effective_length
                    )

                    # Skip if no blocked ranges at this port
                    if not blocked_ranges:
                        continue

                    # Create binary variables and constraints for each repetition and blocked range
                    for repetition in self.datafactory.stream_dict[var_stream][
                        var_path
                    ]["repetitions"]:
                        for block_idx, (block_start, block_duration) in enumerate(
                            blocked_ranges
                        ):
                            # Create a binary variable with the unique counter in the name
                            block_var_key = (
                                var_stream,
                                var_path,
                                repetition,
                                port,
                                block_idx,
                            )
                            self.dv_yf[block_var_key] = pulp.LpVariable(
                                f"yf_{update_id}_{var_stream}_{var_path}_{repetition}_{port[0]}_{port[1]}_{block_idx}",
                                cat="Binary",
                            )

                            # Calculate block end time
                            block_end = block_start + block_duration

                            # Constraint 1: Variable stream finishes before block starts (if yf=0)
                            # x_var + variable_proc_dur*z_var <= block_start + M * yf
                            constraint_name = f"var_fixed_before_{constraint_count}"
                            self.prob.constraints[constraint_name] = (
                                self.dv_x[(var_stream, var_path, repetition, port)]
                                + variable_proc_dur * self.dv_z[(var_stream, var_path)]
                                <= block_start
                                + self.datafactory.M * self.dv_yf[block_var_key]
                            )
                            self.variable_fixed_constraints.append(constraint_name)

                            # Constraint 2: Variable stream starts after block ends (if yf=1)
                            # x_var >= block_end - M * (1 - yf)
                            constraint_name = f"var_fixed_after_{constraint_count}"
                            self.prob.constraints[constraint_name] = self.dv_x[
                                (var_stream, var_path, repetition, port)
                            ] >= block_end - self.datafactory.M * (
                                1 - self.dv_yf[block_var_key]
                            )
                            self.variable_fixed_constraints.append(constraint_name)

                            constraint_count += 1

        if constraint_count > 0:
            print(f"Added {constraint_count} variable-fixed port constraints")

    def get_scheduled_path(self, stream):
        """Get the path selected for a scheduled stream"""
        # Check if stream 'a' variable exists and is scheduled
        a_var = self.dv_a.get(stream)
        if a_var is None or a_var.value() is None or a_var.value() < 0.5:
            return None  # Stream not scheduled or value not available

        # Find the path 'z' variable that is set to 1
        for path in self.datafactory.stream_dict[stream]:
            z_var = self.dv_z.get((stream, path))
            if z_var is not None and z_var.value() is not None and z_var.value() > 0.5:
                return path
        # This should ideally not happen if a=1, but return None as fallback
        # print(f"Warning: Stream {stream} has a=1 but no path z=1 found.")
        return None

    def get_newly_scheduled_streams(self):
        """Check which variable streams were scheduled in the current iteration"""
        newly_scheduled = []

        for stream in list(self.variable_streams):  # Iterate over a copy
            a_var = self.dv_a.get(stream)
            if a_var is not None and a_var.value() is not None and a_var.value() > 0.5:
                newly_scheduled.append(stream)

        print(f"Newly scheduled streams: {newly_scheduled}")
        print(f"Previously scheduled streams: {len(self.fixed_streams)}")

        return newly_scheduled

    def fix_stream(self, stream):
        """
        Fix variables for a single stream that was successfully scheduled.

        Args:
            stream: Stream index to fix

        Returns:
            bool: True if fixing was successful, False otherwise
        """
        # Check if stream is not fixed already
        if stream in self.fixed_streams:
            print(f"Warning: Stream {stream} is already fixed, nothing to do")
            return False

        # Check if stream is in variable streams
        if stream not in self.variable_streams:
            print(f"Warning: Stream {stream} is not in variable_streams, can't fix it")
            return False

        # Initialize constraint tracking dictionary if it doesn't exist
        if not hasattr(self, "fixed_constraints"):
            self.fixed_constraints = {}

        # Initialize list of constraints for this stream
        self.fixed_constraints[stream] = []

        # Get the scheduled path
        scheduled_path = self.get_scheduled_path(stream)
        if scheduled_path is None:
            a_val = self.dv_a.get(stream).value() if self.dv_a.get(stream) else "N/A"
            print(
                f"Error: Stream {stream} marked for fixing but no scheduled path found (a={a_val}). Cannot fix."
            )
            # Optionally, unschedule the stream if path is missing despite a=1
            # self.dv_a[(stream)].setInitialValue(0) # Requires solver re-run potentially
            return False
        
        if "ports" not in self.datafactory.stream_dict[stream][scheduled_path]:
            print(f"Error: Scheduled path {scheduled_path} for stream {stream} has no ports. Cannot fix.")
            return False


        # Store current values of x variables for the scheduled path
        current_values_x = {}
        for repetition in self.datafactory.stream_dict[stream][scheduled_path][
            "repetitions"
        ]:
            for port in self.datafactory.stream_dict[stream][scheduled_path]["ports"]:
                key = (stream, scheduled_path, repetition, port)
                x_var = self.dv_x.get(key)
                if x_var is not None and x_var.value() is not None:
                    current_values_x[key] = x_var.value()
                else:
                    print(
                        f"Error: Missing value for x variable {key} for stream {stream} to be fixed."
                    )
                    return False  # Cannot fix if values are missing

        # Fix a variable to 1
        constraint_name = f"fix_a_{stream}"
        self.prob += self.dv_a[(stream)] == 1, constraint_name
        self.fixed_constraints[stream].append(constraint_name)

        # Fix z variables - 1 for scheduled path, 0 for others
        for path in self.datafactory.stream_dict[stream]:
            value = 1 if path == scheduled_path else 0
            constraint_name = f"fix_z_{stream}_{path}"
            self.prob += self.dv_z[(stream, path)] == value, constraint_name
            self.fixed_constraints[stream].append(constraint_name)

        # Fix x variables for the scheduled path using stored values
        for key, value in current_values_x.items():
            # Construct constraint name carefully
            s, p, r, pt = key
            constraint_name = f"fix_x_{s}_{p}_{r}_{pt[0]}_{pt[1]}"
            self.prob += (
                self.dv_x[key] == value,
                constraint_name,
            )
            self.fixed_constraints[stream].append(constraint_name)

        # Update stream status
        self.fixed_streams.append(stream)
        self.variable_streams.remove(stream)

        print(
            f"Fixed stream {stream} with path {scheduled_path} using {len(self.fixed_constraints[stream])} constraints"
        )
        return True

    def fix_scheduled_stream_variables(self, streams_to_fix):
        """
        Fix variables for multiple streams that were successfully scheduled.

        Args:
            streams_to_fix: List of stream indices to fix
        """
        fixed_count = 0
        failed_count = 0
        for stream in streams_to_fix:
            if self.fix_stream(stream):
                fixed_count += 1
            else:
                failed_count += 1
        print(
            f"Attempted to fix {len(streams_to_fix)} streams. Success: {fixed_count}, Failed: {failed_count}"
        )

    def unfix_stream(self, stream):
        """
        Remove all fixation constraints for a specific stream, making it variable again.

        Args:
            stream: Stream index to unfix

        Returns:
            bool: True if unfixing was successful, False otherwise
        """
        # Check if stream is fixed
        if stream not in self.fixed_streams:
            print(f"Warning: Stream {stream} is not in fixed_streams, nothing to unfix")
            return False

        # Check if we have constraint records for this stream
        if (
            not hasattr(self, "fixed_constraints")
            or stream not in self.fixed_constraints
        ):
            print(f"Warning: No fixation constraints found for stream {stream}")
            # Proceed to update status anyway, assuming constraints might have been lost
            if stream in self.fixed_streams: self.fixed_streams.remove(stream)
            if stream not in self.variable_streams:
                self.variable_streams.append(stream)
            return True  # Indicate status updated, though constraints weren't removed

        # Remove all tracked constraints for this stream
        constraint_count = len(self.fixed_constraints[stream])
        removed_count = 0
        for constraint_name in self.fixed_constraints[stream]:
            if constraint_name in self.prob.constraints:
                del self.prob.constraints[constraint_name]
                removed_count += 1
            else:
                print(
                    f"Warning: Constraint {constraint_name} for stream {stream} not found in model."
                )

        # Clear constraint tracking for this stream
        del self.fixed_constraints[stream]

        # Update stream status
        if stream in self.fixed_streams: self.fixed_streams.remove(stream)
        if stream not in self.variable_streams:  # Avoid duplicates
            self.variable_streams.append(stream)

        print(
            f"Unfixed stream {stream}, removed {removed_count}/{constraint_count} constraints"
        )
        return True

    def get_solution_variables(self):
        """Get all solution variables with clean values"""
        # All variables raw values
        variables_raw = {var.name: var.value() for var in self.prob.variables()}

        # Extract x variables (start times) - only non-None
        variables_x = {
            key: var.value()
            for key, var in self.dv_x.items()
            if var.value() is not None
        }

        # Extract z variables (path selection) - cleaned to 0 or 1
        variables_z = {}
        for stream in self.datafactory.stream_dict:
            scheduled_path = self.get_scheduled_path(
                stream
            )  # Uses current solution values
            for path in self.datafactory.stream_dict[stream]:
                key = (stream, path)
                variables_z[key] = 1 if path == scheduled_path else 0

        # Extract a variables (stream scheduling) - cleaned to 0 or 1
        variables_a = {}
        for stream in self.datafactory.stream_dict:
            a_var = self.dv_a.get(stream)
            val = a_var.value() if a_var else None
            variables_a[stream] = 1 if val is not None and val > 0.5 else 0

        return variables_x, variables_z, variables_a

    def solve(self, iteration=None):
        """
        Solve the current model

        Args:
            iteration: The current iteration number (used for log files)

        Returns:
            String: Solution status
        """
        # Create iteration-specific log file if an iteration is provided
        if iteration is not None:
            log_file_path = os.path.join(
                self.output_path, f"solver_log_iteration_{iteration}.log"
            )
            lp_file_path = os.path.join(
                self.output_path, f"model_iteration_{iteration}.lp"
            )
            # self.prob.writeLP(lp_file_path) # Optionally write LP file for debugging
        else:
            log_file_path = os.path.join(self.output_path, "solver_log.log")
            lp_file_path = os.path.join(self.output_path, "model.lp")
            # self.prob.writeLP(lp_file_path) # Optionally write LP file for debugging


        # Update the solver with the potentially new log file path
        self.solver = pulp.GUROBI(
            msg=True, logPath=log_file_path, timeLimit=self.time_limit
        )

        # Solve the problem
        status = self.prob.solve(self.solver)
        return pulp.LpStatus[status]

    def get_fixed_stream_port_schedule(self, port):
        """
        Get the start times for all fixed streams at a specific port, using current variable values.

        Args:
            port: The port identifier tuple (e.g., ('0-SW1', '0-SW2'))

        Returns:
            List of tuples: [(start_time, stream, path, repetition, port_used), ...] sorted by start_time
        """
        scheduled_times = []

        # Only check fixed streams
        for stream in self.fixed_streams:
            # Get the path that was selected for this stream
            scheduled_path = self.get_scheduled_path(stream)
            if scheduled_path is None:
                continue  # Should not happen for fixed streams, but check anyway
            
            if "ports" not in self.datafactory.stream_dict[stream][scheduled_path]: continue

            # Check if this path uses the specified port
            if (
                port
                not in self.datafactory.stream_dict[stream][scheduled_path]["ports"]
            ):
                continue

            # Get all repetitions and their start times at this port
            for repetition in self.datafactory.stream_dict[stream][scheduled_path][
                "repetitions"
            ]:
                key = (stream, scheduled_path, repetition, port)
                x_var = self.dv_x.get(key)
                start_time = x_var.value() if x_var is not None else None

                if start_time is not None:
                    scheduled_times.append(
                        (
                            start_time,
                            stream,
                            scheduled_path,
                            repetition,
                            port,
                        )  # Include port used
                    )

        # Sort by start time
        scheduled_times.sort()

        return scheduled_times

    def calculate_blocked_ranges(self, port, variable_effective_length):
        """
        Calculate blocked time ranges at a specific port based on fixed streams using that port.

        A blocked range is a time interval where a variable stream of `variable_effective_length`
        cannot be scheduled without overlapping a fixed stream or fitting into a gap smaller
        than `variable_effective_length`.

        Args:
            port: The port identifier tuple (e.g., ('0-SW1', '0-SW2'))
            variable_effective_length: The total effective length (including its own buffer period)
                                       needed for the variable stream. This is used as `min_gap_to_fit_item`.
        Returns:
            List of tuples: [(start_time, duration), ...] for each blocked range.
        """
        fixed_schedule_at_port = self.get_fixed_stream_port_schedule(port)
        if not fixed_schedule_at_port:
            return []

        effective_occupancies = []
        for start_time, stream, path, rep, port_used in fixed_schedule_at_port:
            # Ensure path data exists before trying to get effective start/end
            if stream not in self.datafactory.stream_dict or \
               path not in self.datafactory.stream_dict[stream] or \
               "gamma" not in self.datafactory.stream_dict[stream][path]: # Check for gamma key
                print(f"Warning: Missing data for stream {stream}, path {path} in calculate_blocked_ranges. Skipping this occupancy.")
                continue

            eff_start, eff_end = self._get_effective_start_end(
                stream, path, port_used, start_time
            )
            effective_occupancies.append((eff_start, eff_end))

        effective_occupancies.sort()  # Sort by effective start time

        return self._merge_time_ranges(effective_occupancies, variable_effective_length)
