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
    def __init__(self, num_conditions: int, similarity_measure: str = "spearman"):
        """
        Initialize the RSA class.

        Args:
        num_conditions (int): The number of conditions in the RDMs.
        similarity_measure (str): The similarity measure to use, e.g., "spearman", "pearson".
        """
        self.similarity_measure = similarity_measure
        self.num_conditions = num_conditions
        self.similarity_function = correlation_functions[self.similarity_measure]

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

    def _get_rdm_vectors(self, rdms: np.array) -> np.array:
        """
        Transform 2D RDMs into 1D vectors using the upper triangle.

        Args:
            rdms (np.array): The RDMs. Shape: [num_rdms, num_conditions, num_conditions].

        Returns:
            rdm_vectors (np.array): The RDM vectors. Shape: [num_rdms, num_conditions * (num_conditions - 1) / 2].

        """
        num_rdms = rdms.shape[0]
        num_conditions = rdms.shape[-1]
        rdm_vectors = np.zeros((num_rdms, int(num_conditions * (num_conditions - 1) / 2)))
        for i in range(num_rdms):
            rdm_vectors[i] = rdms[i][np.triu_indices(num_conditions, k=1)]
        return rdm_vectors

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
        rdm_length = meg_rdm.shape[1]
        num_layers = model_rdm.shape[0]
        meg_rdm_vector = np.zeros((num_brain_elements, rdm_length, int(meg_rdm.shape[-1] * (meg_rdm.shape[-1] - 1) / 2)))
        for i in range(num_brain_elements):
            meg_rdm_vector[i] =  self._get_rdm_vectors(meg_rdm[i])
        model_rdm_vector = self._get_rdm_vectors(model_rdm) 
        del meg_rdm, model_rdm
        similarity = np.zeros((num_brain_elements, rdm_length, num_layers, 2))

        for i in tqdm(range(num_brain_elements), desc="Computing RSA"):
            for j in range(rdm_length):
                for k in range(num_layers):
                    similarity[i, j, k] = self.calculate_rsa(meg_rdm_vector[i, j], model_rdm_vector[k], permutation)
        return similarity

    def _rsa_parallel(self, meg_rdm: np.array, model_rdm: np.array, permutation: bool = False, iter: int = 1000) -> np.array:
        """
        Compute the similarity between MEG RDMs and model RDMs in parallel.
        """
        if meg_rdm.shape[2] != model_rdm.shape[1]:
            raise ValueError("The two RDMs must have the same length.")

        num_brain_elements = meg_rdm.shape[0]
        rdm_length = meg_rdm.shape[1]
        num_layers = model_rdm.shape[0]

        meg_rdm_vector = np.zeros((num_brain_elements, rdm_length,  int(meg_rdm.shape[-1] * (meg_rdm.shape[-1] - 1) / 2)))
        for i in range(num_brain_elements):
            meg_rdm_vector[i] =  self._get_rdm_vectors(meg_rdm[i])
        model_rdm_vector = self._get_rdm_vectors(model_rdm) 
        del meg_rdm, model_rdm
        similarity = np.zeros((num_brain_elements, rdm_length, num_layers, 2))

        with multiprocessing.Pool() as pool:
            results = list(
                tqdm(
                    pool.starmap(
                        _calculate_similarity_parallel_rsa_movie,
                        [(i, j, k, meg_rdm_vector, model_rdm_vector, self.calculate_rsa, permutation, iter) for i in range(num_brain_elements) for j in range(rdm_length) for k in range(num_layers)],
                    ), 
                    total=num_brain_elements * rdm_length * num_layers,
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
            'brain_elements': 0,
            'rdms': 1,
            'layers': 2
        }

        if axis_name not in axis_map:
            raise ValueError("Invalid axis_name. Use 'brain_elements', 'rdms', or 'layers'.")

        axis_index = axis_map[axis_name]
        max_values = np.max(similarity, axis=axis_index)

        return max_values

    def bootstrap(self, meg_rdm: np.array, model_rdm: np.array, parrallel: bool = True, N_bootstrap: int = 100) -> np.array:
        """
        Calculate the bootstrap error bars along the subjects.

        Args:
            meg_rdm (np.array): A 5D MEG RDM movie. Shape: [num_subjects, num_brain_elements, num_rdms, num_conditions, num_conditions].
            model_rdm (np.array): The model RDMs. Shape: [num_layers, num_conditions, num_conditions].
            parrallel (bool): Whether to use parallel computation or not.
            N_bootstrap (int): The number of bootstrap iterations.

        Returns:
            similarity (np.array): The similarity measure and p-value. Shape: [num_brain_elements, num_rdms, num_layers, N_bootstrap, 2].
        """

        num_subjects = meg_rdm.shape[0]
        num_brain_elements = meg_rdm.shape[1]
        rdm_length = meg_rdm.shape[2]
        num_layers = model_rdm.shape[0]
        num_subjects_to_sample = int(num_subjects * 0.5)
        similarity = np.zeros((num_brain_elements, rdm_length, num_layers, N_bootstrap, 2))
        for i in tqdm(range(N_bootstrap), desc="bootsrapping RSA"):
            meg_rdm_selected = similarity[np.random.choice(num_subjects, num_subjects_to_sample, replace=False)]
            meg_rdm_selected = np.mean(meg_rdm_selected, axis=0)
            similarity[:, :, :, i] = self.score(meg_rdm_selected, model_rdm, parrallel)
        
        return similarity

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
