import os
import shutil # Keep shutil import if needed for future path operations within PathManager

class PathManager:
    """Manages input and output paths for a pipeline run."""

    def __init__(self, input_dir: str, output_dir: str):
        """
        Initializes PathManager, validates paths, and creates output subdirectories.

        Args:
            input_dir (str): Path to the input data directory.
            output_dir (str): Path to the main output directory for this specific run.
                               This directory will be created if it doesn't exist.

        Raises:
            FileNotFoundError: If the input directory does not exist.
            OSError: If the output directory or subdirectories cannot be created.
            ValueError: If input_dir or output_dir is not a valid directory path string.
        """
        if not isinstance(input_dir, str) or not input_dir:
            raise ValueError("input_dir must be a non-empty string.")
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("output_dir must be a non-empty string.")

        self._input_dir = os.path.abspath(input_dir)
        self._output_dir = os.path.abspath(output_dir)

        # Validate input directory
        if not os.path.isdir(self._input_dir):
            raise FileNotFoundError(f"Input directory not found: {self._input_dir}")

        # Define output subdirectories based on the provided output_dir
        self._processed_results_dir = os.path.join(self._output_dir, "processed_results")
        self._memory_dir = os.path.join(self._output_dir, "memory")
        # Add other subdirs if needed, e.g., for solver files
        self._solver_files_dir = os.path.join(self._output_dir, "solver_files")


        # Create output directories
        try:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(self._processed_results_dir, exist_ok=True)
            os.makedirs(self._memory_dir, exist_ok=True)
            os.makedirs(self._solver_files_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create output directories under {self._output_dir}: {e}")

    @property
    def input_dir(self) -> str:
        """Returns the absolute path to the input directory."""
        return self._input_dir

    @property
    def output_dir(self) -> str:
        """Returns the absolute path to the main output directory for the run."""
        return self._output_dir

    @property
    def processed_results_dir(self) -> str:
        """Returns the absolute path to the processed results subdirectory."""
        return self._processed_results_dir

    @property
    def memory_dir(self) -> str:
        """Returns the absolute path to the memory tracking subdirectory."""
        return self._memory_dir

    @property
    def solver_files_dir(self) -> str:
        """Returns the absolute path to the solver files subdirectory."""
        return self._solver_files_dir
    
    @property 
    def stream_dict_input_dir(self) -> str:
        """Returns the absolute path to the stream dictionary subdirectory."""
        return os.path.join(self._input_dir, "stream_dict.pkl")
    
    @property 
    def stream_dict_output_dir(self) -> str:
        """Returns the absolute path to the stream dictionary subdirectory."""
        return os.path.join(self._output_dir, "stream_dict.pkl")
    
    @property
    def variables_x_dir(self) -> str:
        """Returns the absolute path to the variables x subdirectory."""
        return os.path.join(self._output_dir, "variables_x.pkl")
    
    @property
    def variables_z_dir(self) -> str:
        """Returns the absolute path to the variables z subdirectory."""
        return os.path.join(self._output_dir, "variables_z.pkl")
    
    @property
    def variables_a_dir(self) -> str:
        """Returns the absolute path to the variables a subdirectory."""
        return os.path.join(self._output_dir, "variables_a.pkl")
    
    @property
    def network_graphml_output_dir(self) -> str:
        """Returns the absolute path to the network graphml subdirectory."""
        return os.path.join(self._output_dir, "network.graphml")
    
    @property
    def network_graphml_input_dir(self) -> str:
        """Returns the absolute path to the network graphml subdirectory."""
        return os.path.join(self._input_dir, "network.graphml")
    
    @property
    def streams_input_dir(self) -> str:
        """Returns the absolute path to the streams input subdirectory."""
        return os.path.join(self._input_dir, "streams.csv")
    
    @property
    def streams_output_dir(self) -> str:
        """Returns the absolute path to the streams output subdirectory."""
        return os.path.join(self._output_dir, "streams.csv")
    
    @property
    def histogram_config_input_dir(self) -> str:
        """Returns the absolute path to the histogram config input subdirectory."""
        return os.path.join(self._input_dir, "histogram_config.yaml")
    
    @property
    def histogram_config_output_dir(self) -> str:
        """Returns the absolute path to the histogram config output subdirectory."""
        return os.path.join(self._output_dir, "histogram_config.yaml")
    @property
    def link_config_input_dir(self) -> str:
        """Returns the absolute path to the link config input subdirectory."""
        return os.path.join(self._input_dir, "link_config.yaml")
    @property
    def link_config_output_dir(self) -> str:
        """Returns the absolute path to the link config output subdirectory."""
        return os.path.join(self._output_dir, "link_config.yaml")


    # --- Helper methods to get full paths ---

    def get_input_path(self, filename: str) -> str:
        """Constructs the full path for a file within the input directory."""
        return os.path.join(self._input_dir, filename)

    def get_output_path(self, filename: str) -> str:
        """Constructs the full path for a file within the main output directory."""
        return os.path.join(self._output_dir, filename)

    def get_processed_results_path(self, filename: str) -> str:
        """Constructs the full path for a file within the processed results subdirectory."""
        return os.path.join(self._processed_results_dir, filename)

    def get_memory_path(self, filename: str) -> str:
        """Constructs the full path for a file within the memory subdirectory."""
        return os.path.join(self._memory_dir, filename)

    def get_solver_files_path(self, filename: str) -> str:
        """Constructs the full path for a file within the solver files subdirectory."""
        return os.path.join(self._solver_files_dir, filename)
