import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import multiprocessing
from typing import Dict


def _calculate_rdm_parallel(activation_data, rdm_calculator):
    layer_id, layer_activation = activation_data
    return layer_id, rdm_calculator(layer_activation)

def _calculate_rdm_parallel_temp_brain_rdms(segment_data, rdm_calculator):
    time_segment_id, t_segment_start, t_segment_end, brain_element_id, brain_element_activity = segment_data
    segment_activity = brain_element_activity[:, t_segment_start:t_segment_end]
    rdm = rdm_calculator(segment_activity)
    return time_segment_id, brain_element_id, rdm

def _calculate_rdm_parallel_brain_rdm_movie(segment_data, rdm_calculator):
    t_segment_start, t_segment_end, brain_element_id, brain_element_activity = segment_data
    segment_activity = brain_element_activity[:, :, t_segment_start:t_segment_end]
    rdm = rdm_calculator(segment_activity.squeeze())
    return t_segment_start, t_segment_end, brain_element_id, rdm

class RDM:
    def __init__(self, num_conditions: int, calculate_absolute: bool = False):
        """
        Initializes an RDMCalculator.

        Args:
            num_conditions (int): The number of conditions (stimuli) used.
            calculate_absolute (bool, optional): Calculate the absolute value of Pearson r or not, default is False.
        """
        self.num_conditions = num_conditions
        self.calculate_absolute = calculate_absolute

    @staticmethod
    def round_to_zero_if_close(value):
        """
        Rounds a float value to zero if it's close enough (within 1e-15).

        Args:
            value (float): The float value to round to zero.

        Returns:
            float: The original value rounded to zero if it's close enough, otherwise the original value.
        """
        if abs(value) < 1e-15:
            return 0
        return value

    def calculate_rdm(self, activity):
        """
        Calculates the Representational Dissimilarity Matrix (RDM) for one Artificial Neural Network layer
        activations or for a brain region.

        Args:
            activity (np.ndarray): The model layer/brain region activity patterns.

        Returns:
            np.ndarray: The activations RDM. The shape is [num_conditions, num_conditions].
        """
        rdm = np.zeros((self.num_conditions, self.num_conditions), dtype=np.float64)

        for i in range(self.num_conditions):
            for j in range(i + 1, self.num_conditions):
                r = pearsonr(activity[i], activity[j])[0]
                value = 1 - np.abs(r) if self.calculate_absolute else 1 - r
                rdm[i, j] = self.round_to_zero_if_close(value)

        return rdm

    def save(self, rdm, file_path):
        """
        Saves an RDM to a NumPy file.

        Args:
            rdm (np.ndarray): The RDM to save.
            file_path (str): The path to the NumPy file where the RDM will be saved.
        """
        np.save(file_path, rdm)

    def load(self, file_path):
        """
        Loads an RDM from a NumPy file.

        Args:
            file_path (str): The path to the NumPy file containing the RDM.

        Returns:
            np.ndarray: The loaded RDM.
        """
        return np.load(file_path)


    def brain_rdm_movie(self, brain_activity: Dict[str, np.ndarray], t_start: int = 0, t_end: int = 1101) -> np.ndarray:
        """
        Compute a movie of Representational Dissimilarity Matrices (RDMs) for brain activity over time.

        This function calculates a sequence of RDMs over time from the brain activity data provided.
        Each RDM in the movie represents the dissimilarity between conditions at a specific time point.

        Args:
            brain_activity (dict): A dictionary containing the brain regions/sensors activity patterns.
            t_start (int, optional): The starting time point for the RDM movie, default is 0.
            t_end (int, optional): The ending time point for the RDM movie, default is 1101.

        Returns:
            np.ndarray: A 4D array representing the RDM movie. Shape: [num_brain_elements, rdm_movie_length, num_conditions, num_conditions].

        Note:
        - The function computes RDMs over a specified time range for each brain element in the input data.
        - The resulting RDM movie captures temporal changes in dissimilarity patterns between conditions.
        """
        rdm_movie_length = t_end - t_start
        num_brain_elements = len(brain_activity)
        rdms = np.zeros((num_brain_elements, rdm_movie_length, self.num_conditions, self.num_conditions), dtype=np.float64)
        for brain_element_id, (_, brain_element_activity) in tqdm(enumerate(brain_activity.items()), desc="Calculating brain RDMs movie", total=num_brain_elements):
            for t in range(rdm_movie_length):
                segment_activity = brain_element_activity[:, :, t_start + t]
                rdms[brain_element_id, t] = self.calculate_rdm(segment_activity)

        return rdms

    def brain_rdm_movie_parallel(self, brain_activity: Dict[str, np.ndarray], t_start: int = 0, t_end: int = 1101) -> np.ndarray:
        """
        Compute a movie of Representational Dissimilarity Matrices (RDMs) for brain activity over time using parallel processing.

        This function calculates a sequence of RDMs over time from the brain activity data provided.
        Each RDM in the movie represents the dissimilarity between conditions at a specific time point.

        Args:
            brain_activity (dict): A dictionary containing the brain regions/sensors activity patterns.
            t_start (int, optional): The starting time point for the RDM movie, default is 0.
            t_end (int, optional): The ending time point for the RDM movie, default is 1101.

        Returns:
            np.ndarray: A 4D array representing the RDM movie. Shape: [num_brain_elements, rdm_movie_length, num_conditions, num_conditions].

        Note:
        - The function computes RDMs over a specified time range for each brain element in the input data.
        - The resulting RDM movie captures temporal changes in dissimilarity patterns between conditions.
        """
        rdm_movie_length = t_end - t_start
        num_brain_elements = len(brain_activity)
        rdms = np.zeros((num_brain_elements, rdm_movie_length, self.num_conditions, self.num_conditions), dtype=np.float64)
        
        # Create a list of segment data for parallel processing
        segment_data_list = []

        for brain_element_id, (_, brain_element_activity) in enumerate(brain_activity.items()):
            for t in range(rdm_movie_length):
                t_segment_start = t_start + t
                t_segment_end = t_segment_start + 1  # Use a single time point for each segment
                segment_data = (t_segment_start, t_segment_end, brain_element_id, brain_element_activity)
                segment_data_list.append(segment_data)
        # Parallel processing
        with multiprocessing.Pool() as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        _calculate_rdm_parallel_brain_rdm_movie,
                        [(data, self.calculate_rdm) for data in segment_data_list],
                    ),
                    total=len(segment_data_list),
                    desc="Calculating brain RDMs movie in parallel",
                )
            )

        # Organize the results into the rdms array
        for result in results:
            t_segment_start, t_segment_end, brain_element_id, rdm = result
            rdms[brain_element_id, t_segment_start - t_start] = rdm

        return rdms


    def temp_brain_rdms(self, brain_activity: Dict[str, np.ndarray], time_segment: int = 550, sliding_window: int = 50, t_start: int = 220, t_end: int = 1101, ) -> np.ndarray:
        """
        Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for brain regions/sensors activity patterns across time.

        Args:
            brain_activity (dict): A dictionary containing the brain regions/sensors activity patterns.
            time_segment (int, optional): The time segment length to use to compute the RDMs, default is 550.
            sliding_window (int, optional): The sliding window to use to compute the RDMs, default is 50.
            t_start (int, optional): The start time from where we will start computing RDMs, default is 220.
            t_end (int, optional): The end time from where we will stop computing RDMs, default is 1101.

        Returns:
            np.ndarray: An array of RDMs for each time segment. The shape is [num_time_segments, num_brain_elements, num_conditions, num_conditions].
        """

        num_time_segments = ((t_end - t_start - time_segment )  // sliding_window ) + 1
        num_brain_elements = len(brain_activity)
        rdms = np.zeros((num_brain_elements, num_time_segments, self.num_conditions, self.num_conditions), dtype=np.float64)

        for time_segment_id in tqdm(range(num_time_segments), desc="Calculating brain temporal RDMs"):
            t_segment_start = t_start + time_segment_id * sliding_window
            t_segment_end = t_segment_start + time_segment
            
            for brain_element_id, (_, brain_element_activity) in enumerate(brain_activity.items()):
                segment_activity = brain_element_activity[:, t_segment_start:t_segment_end]
                rdms[brain_element_id, time_segment_id] = self.calculate_rdm(segment_activity)

        return rdms

    def temp_brain_rdms_parallel(self, brain_activity: Dict[str, np.ndarray], time_segment: int = 550, sliding_window: int = 50, t_start: int = 220, t_end: int = 1101) -> np.ndarray:
        """
        Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for brain regions/sensors activity patterns across time using parallel processing.

        Args:
            brain_activity (dict): A dictionary containing the brain regions/sensors activity patterns.
            time_segment (int, optional): The time segment length to use to compute the RDMs, default is 550.
            sliding_window (int, optional): The sliding window to use to compute the RDMs, default is 50.
            t_start (int, optional): The start time from where we will start computing RDMs, default is 220.
            t_end (int, optional): The end time from where we will stop computing RDMs, default is 1101.

        Returns:
            np.ndarray: An array of RDMs for each time segment. The shape is [num_brain_elements, num_time_segments, num_conditions, num_conditions].
        """

        num_time_segments = ((t_end - t_start - time_segment) // sliding_window) + 1
        num_brain_elements = len(brain_activity)
        rdms = np.zeros((num_brain_elements, num_time_segments, self.num_conditions, self.num_conditions), dtype=np.float64)

        # Create a list of segment data for parallel processing
        segment_data_list = []

        for time_segment_id in range(num_time_segments):
            t_segment_start = t_start + time_segment_id * sliding_window
            t_segment_end = t_segment_start + time_segment

            for brain_element_id, (_, brain_element_activity) in enumerate(brain_activity.items()):
                segment_data = (time_segment_id, t_segment_start, t_segment_end, brain_element_id, brain_element_activity)
                segment_data_list.append(segment_data)

        # Parallel processing
        with multiprocessing.Pool() as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        _calculate_rdm_parallel_temp_brain_rdms,
                        [(data, self.calculate_rdm) for data in segment_data_list],
                    ),
                    total=len(segment_data_list),
                    desc="Calculating brain temporal RDMs in parallel",
                )
            )

        # Organize the results into the rdms array
        for result in results:
            time_segment_id, brain_element_id, rdm = result
            rdms[brain_element_id, time_segment_id] = rdm

        return rdms


    def model_rdms(self, model_activations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for Artificial Neural Networks layers activations.

        Args:
            model_activations (dict): A dictionary containing the activations of the Artificial Neural Network layers.
            
        Returns:
            np.ndarray: An array of RDMs for each layer. The shape is [num_layers, num_conditions, num_conditions].
        """
        num_layers = len(model_activations)
        rdms = np.zeros((num_layers, self.num_conditions, self.num_conditions), dtype=np.float64)
        
        for layer_id, (_, layer_activation) in tqdm(enumerate(model_activations.items()), desc="Calculating ANN RDMs", total=num_layers):
            rdms[layer_id] = self.calculate_rdm(layer_activation)
        
        return rdms

    def model_rdms_parallel(self, model_activations: Dict[str, np.ndarray], num_processes: int = None) -> np.ndarray:
        """
        Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for Artificial Neural Networks layers activations using parallel processing.

        Args:
            model_activations (dict): A dictionary containing the activations of the Artificial Neural Network layers.
            num_processes (int, optional): Number of parallel processes to use, default is None (auto-detect).

        Returns:
            np.ndarray: An array of RDMs for each layer. The shape is [num_layers, num_conditions, num_conditions].
        """
        num_layers = len(model_activations)
        rdms = np.zeros((num_layers, self.num_conditions, self.num_conditions), dtype=np.float64)
        
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        
        activations_list = [(layer_id, layer_activation) for layer_id, layer_activation in enumerate(model_activations.values())]

        with multiprocessing.Pool() as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        _calculate_rdm_parallel,
                        [(data, self.calculate_rdm) for data in activations_list],
                    ),
                    total=num_layers,
                    desc="Calculating ANN RDMs in parallel",
                )
            )
        
        for layer_id, rdm in results:
            rdms[layer_id] = rdm

        return rdms