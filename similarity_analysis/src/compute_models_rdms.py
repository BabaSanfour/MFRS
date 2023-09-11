import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
import multiprocessing
from typing import Dict


def calculate_rdm_parallel(layer_id, layer_activation, rdm_calculator):
    return rdm_calculator.calculate_rdm(layer_activation)


class RDM:
    def __init__(self, num_conditions, calculate_absolute=False):
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

    def brain_rdms(self):
        pass

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
        
        for layer_id, (_, layer_activation) in tqdm(enumerate(model_activations.items()), desc="Calculating RDMs", total=num_layers):
            rdms[layer_id] = self.calculate_rdm(layer_activation)
        
        return rdms

    def model_rdms_parallel(self, model_activations: Dict[str, np.ndarray], num_processes: int = None) -> np.ndarray:
        """
        Calculate the Representational Dissimilarity Matrix(Matrices) - RDM(s) for Artificial Neural Networks layers activations.

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

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        calculate_rdm_parallel,
                        [(layer_id, layer_activation, self) for layer_id, (_, layer_activation) in enumerate(model_activations.items())],
                    ),
                    total=num_layers,
                    desc="Calculating RDMs in parallel",
                )
            )

        rdms = np.array(results)

        return rdms