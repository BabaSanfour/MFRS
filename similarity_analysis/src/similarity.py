import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr, kendalltau
import multiprocessing


correlation_functions = {
        "spearman": spearmanr,
        "pearson": pearsonr,
        "kendalltau": kendalltau,
}

def _calculate_similarity_parallel_rsa_movie(i, j, k, meg_rdm, model_rdm, calculate_rsa, permutation, iter):
    return i, j, k, calculate_rsa(meg_rdm[i, j], model_rdm[k], permutation, iter)

class RSA:
    def __init__(self, similarity_measure: str, num_conditions: int):
        """
        Initialize the RSA class.

        Args:
        similarity_measure (str): The similarity measure to use, e.g., "spearman", "pearson".
        num_conditions (int): The number of conditions in the RDMs.
        """
        self.similarity_measure = similarity_measure
        self.num_conditions = num_conditions
        self.similarity_function = correlation_functions[self.similarity_measure]

    def save(self, sim, file_path):
        """
        Saves an similary scores to a NumPy file.

        Args:
            sim (np.ndarray): The similarity scores to save.
            file_path (str): The path to the NumPy file where the similarity scores will be saved.
        """
        np.save(file_path, sim)

    def load(self, file_path):
        """
        Loads an similary scores from a NumPy file.

        Args:
            file_path (str): The path to the NumPy file containing the similary scores.

        Returns:
            np.ndarray: The loaded similary scores.
        """
        return np.load(file_path)

    def permutation(self, rdm1_vector: np.array, rdm2_vector: np.array, r: float, iter: int = 1000) -> float:

        """
        Conduct Permutation test for correlation coefficients

        Args:
            rdm1_vector (np.array): The first RDM.  
            rdm2_vector (np.array): The second RDM.
            r (float): The correlation coefficient of the original data.
            iter (int): The number of iterations for permutation tests.

        Returns:
            p (float): The p-value of the permutation test.
        """

        ni = 0

        for _ in range(iter):
            rdm1_vector_shuffle = np.random.permutation(rdm1_vector)
            rdm2_vector_shuffle = np.random.permutation(rdm2_vector)
            rperm = self.similarity_function(rdm1_vector_shuffle, rdm2_vector_shuffle)[0]

            if rperm > r:
                ni += 1

        p = np.float64((ni + 1) / (iter + 1))
        return p

    def calculate_rsa(self, rdm1_vector: np.array, rdm2_vector: np.array, permutation: bool, iter: int = 1000) -> np.array:
        """
        Compute the similarity between two RDMs.

        Args:
            rdm1_vector (np.array): The first RDM.
            rdm2_vector (np.array): The second RDM.
            permutation (bool): Whether to use permutation tests or not.

        Returns:
            rp (np.array): The similarity measure and p-value.

        """
        if len(rdm1_vector) != len(rdm2_vector):
            raise ValueError("The two RDMs must have the same length.")
        rp = np.array(self.similarity_function(rdm1_vector, rdm2_vector))

        if permutation == True:

            rp[1] = self.permutation(rdm1_vector, rdm2_vector, r=rp[0], iter=iter)

        return rp

    def _rsa(self, meg_rdm: np.array, model_rdm: np.array, permutation: bool = False, iter: int = 1000) -> np.array:
        """
        Compute the similarity between MEG RDMs and model RDMs.

        Args:
            meg_rdm (np.array): A 4D MEG RDM movie. Shape: [num_brain_elements, num_rdms, num_conditions, num_conditions].
            model_rdm (np.array): The model RDMs. Shape: [num_layers, num_conditions, num_conditions]
            permutation (bool): Whether to use permutation tests or not.

        Returns:
            similarity (np.array): The similarity measure and p-value. Shape: [num_brain_elements, num_rdms, num_layers, 2].

        """
        if meg_rdm.shape[2] != model_rdm.shape[1]:
            raise ValueError("The two RDMs must have the same length.")

        num_brain_elements = meg_rdm.shape[0]
        rdm_movie_length = meg_rdm.shape[1]
        num_layers = model_rdm.shape[0]

        similarity = np.zeros((num_brain_elements, rdm_movie_length, num_layers, 2))

        for i in tqdm(range(meg_rdm.shape[0]), desc="Computing RSA"):
            for j in range(meg_rdm.shape[1]):
                for k in range(model_rdm.shape[0]):
                    similarity[i, j, k] = self.calculate_rsa(meg_rdm[i, j], model_rdm[k], permutation)
        return similarity

    def _rsa_parallel(self, meg_rdm: np.array, model_rdm: np.array, permutation: bool = False, iter: int = 1000) -> np.array:
        """
        Compute the similarity between MEG RDMs and model RDMs in parallel.
        """
        if meg_rdm.shape[2] != model_rdm.shape[1]:
            raise ValueError("The two RDMs must have the same length.")

        num_brain_elements = meg_rdm.shape[0]
        rdm_movie_length = meg_rdm.shape[1]
        num_layers = model_rdm.shape[0]

        similarity = np.zeros((num_brain_elements, rdm_movie_length, num_layers, 2))

        with multiprocessing.Pool() as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        _calculate_similarity_parallel_rsa_movie,
                        [(i, j, k, meg_rdm, model_rdm, self.calculate_rsa, permutation, iter) for i in range(num_brain_elements) for j in range(rdm_movie_length) for k in range(num_layers)],
                    ), 
                    total=num_brain_elements * rdm_movie_length * num_layers,
                    desc="Computing RSA in parallel",
                )
            )

        for i, j, k, result in results:
            similarity[i, j, k] = result

        return similarity

    def score(self, meg_rdm: np.array, model_rdm: np.array, parrallel: bool = True, permutation: bool = False, iter: int = 1000) -> np.array:
        """
        Compute the similarity between MEG RDMs and model RDMs.

        Args:
            meg_rdm (np.array): A 4D MEG RDM movie. Shape: [num_brain_elements, num_rdms, num_conditions, num_conditions].
            model_rdm (np.array): The model RDMs. Shape: [num_layers, num_conditions, num_conditions]
            parrallel (bool): Whether to use parallel computation or not.
            permutation (bool): Whether to use permutation tests or not.
            iter (int): The number of iterations for permutation tests.

        Returns:
            similarity (np.array): The similarity measure and p-value. Shape: [num_brain_elements, num_rdms, num_layers, 2].

        """
        if parrallel:
            return self._rsa_parallel(meg_rdm, model_rdm, permutation, iter)
        else:
            return self._rsa(meg_rdm, model_rdm, permutation, iter)
    
    def max_sim_scores(self, similarity: np.array, axis_name: str) -> np.array:
        """
        Calculate the maximum r values along the specified axis.

        Args:
            similarity_array (np.ndarray): An array of shape [num_brain_elements, num_rdms, num_layers, 2].
            axis_name (str): A string specifying the axis along which to calculate the maximum values
                            ('brain_elements', 'rdms', or 'layers').

        Returns:
            max_values (np.ndarray): An array containing the max r values along the specified axis.
        """
        axis_map = {
            'num_brain_elements': 0,
            'num_rdms': 1,
            'num_layers': 2
        }

        if axis_name not in axis_map:
            raise ValueError("Invalid axis_name. Use 'num_brain_elements', 'num_rdms', or 'num_layers'.")

        axis_index = axis_map[axis_name]
        max_values = np.max(similarity, axis=axis_index)

        return max_values