import numpy as np
from tqdm import tqdm
from typing import Tuple

from .rsa import RSA


class noise_ceiling:
    def __init__(self, data: np.array, similarity_measure: str = "spearman", n_perm: int = 100):
        """
        Compute the noise ceiling for the data

        Args:
            data (numpy array): The data to compute the noise ceiling for. The shape of the data should be (num_subjects, num_brain_elements, num_rdms, num_conditions, num_conditions)
            similarity_measure (str, optional): The similarity measure to use. Defaults to "spearman".
            n_perm (int, optional): The number of permutations to use to compute the noise ceiling. Defaults to 100.
        """
        self.data = data
        self.similarity_measure = similarity_measure
        self.n_perm = n_perm
        self.num_subjects = data.shape[0]
        self.num_brain_elements = data.shape[1]
        self.num_rdms = data.shape[2]
        self.num_conditions = data.shape[3]
        self.num_permutations = n_perm
        self.num_subjects_to_sample = int(self.num_subjects*0.5)
        self.rsa_instance = RSA(self.num_conditions, self.similarity_measure)

    def _bootstrap_noise_ceiling(self) -> np.array:
        """
        Compute the upper noise ceiling for the data using a bootstrap procedure
        """

        noise_ceilings = np.zeros((self.num_brain_elements, self.num_rdms, self.num_permutations))
        for i in tqdm(range(self.num_permutations)):
            meg_rdm_selected1 = self.data[np.random.choice(self.num_subjects, self.num_subjects_to_sample, replace=False)]
            meg_rdm_selected2 = self.data[np.random.choice(self.num_subjects, self.num_subjects_to_sample, replace=False)]
            meg_rdm_selected1 = np.mean(meg_rdm_selected1, axis=0)
            meg_rdm_selected2 = np.mean(meg_rdm_selected2, axis=0)
            
            for j in range(self.num_brain_elements):
                for k in range(self.num_rdms):
                    rdm_vector1 = self.rsa_instance._get_rdm_vectors(meg_rdm_selected1[j, k, :, :][np.newaxis, :, :])[0]
                    rdm_vector2 = self.rsa_instance._get_rdm_vectors(meg_rdm_selected2[j, k, :, :][np.newaxis, :, :])[0]
                    noise_ceilings[j, k, i] = self.rsa_instance.calculate_rsa(rdm_vector1, rdm_vector2)[0]
        mean_noise_ceiling = np.mean(noise_ceilings, axis=2)
        std_noise_ceiling = np.std(noise_ceilings, axis=2)
        return mean_noise_ceiling, std_noise_ceiling
    
    def _loo_upper_noise_ceiling(self) -> np.array:
        """
        Compute the upper noise ceiling for the data using a leave-one-out procedure
        """
        upper_noise_ceiling = np.zeros((self.num_brain_elements, self.num_rdms))
        meg_rdm_selected2 = np.mean(self.data, axis=0)
        for i in tqdm(range(self.num_subjects)):
            meg_rdm_selected1 = self.data[i]
            
            for j in range(self.num_brain_elements):
                for k in range(self.num_rdms):
                    rdm_vector1 = self.rsa_instance._get_rdm_vectors(meg_rdm_selected1[j, k, :, :][np.newaxis, :, :])[0]
                    rdm_vector2 = self.rsa_instance._get_rdm_vectors(meg_rdm_selected2[j, k, :, :][np.newaxis, :, :])[0]
                    upper_noise_ceiling[j, k] += self.rsa_instance.calculate_rsa(rdm_vector1, rdm_vector2, False)[0]

        return upper_noise_ceiling/self.num_subjects


    def _loo_lower_noise_ceiling(self) -> np.array:
        """
        Compute the lower noise ceiling for the data using a leave-one-out procedure
        """
        lower_noise_ceiling = np.zeros((self.num_brain_elements, self.num_rdms))
        for i in tqdm(range(self.num_subjects)):
            meg_rdm_selected1 = self.data[i]
            meg_rdm_selected2 = np.delete(self.data, i, axis=0)
            meg_rdm_selected2 = np.mean(meg_rdm_selected2, axis=0)
            
            for j in range(self.num_brain_elements):
                for k in range(self.num_rdms):
                    rdm_vector1 = self.rsa_instance._get_rdm_vectors(meg_rdm_selected1[j, k, :, :][np.newaxis, :, :])[0]
                    rdm_vector2 = self.rsa_instance._get_rdm_vectors(meg_rdm_selected2[j, k, :, :][np.newaxis, :, :])[0]
                    lower_noise_ceiling[j, k] += self.rsa_instance.calculate_rsa(rdm_vector1, rdm_vector2, False)[0]
        
        return lower_noise_ceiling/self.num_subjects

    def __getitem__(self, type: str = "loo") -> Tuple[np.array, np.array]:
        """
        Get the noises ceiling for the data

        Args:
            type (str, optional): The type of noise ceiling to compute. Defaults to "loo".
        
        Returns:
            Tuple[np.array, np.array]: The upper and lower noise ceiling for the data for loo and mean noise ceiling and std for bootstrap
        """

        if type == "loo":
            upper_noise_ceiling = self._loo_upper_noise_ceiling()
            lower_noise_ceiling = self._loo_lower_noise_ceiling()
            return upper_noise_ceiling, lower_noise_ceiling

        elif type == "bootstrap":
            mean_noise_ceiling, std_noise_ceiling = self._bootstrap_upper_noise_ceiling()
            return mean_noise_ceiling, std_noise_ceiling
        
        else:
            raise ValueError("The type of noise ceiling should be either 'loo' or 'bootstrap'")


        