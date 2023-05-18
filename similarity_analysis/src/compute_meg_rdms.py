"""
Compute MEG RDMs
"""

import os
import mne
import numpy as np
import os.path as op
from neurora.rdm_cal import eegRDM
from tqdm import tqdm
import sys
sys.path.append('../../../MFRS/')
from utils.config import get_similarity_parser
from utils.library.config_bids import meg_dir
from utils.general import save_npy

def transform_data(sub_id: list, n_cons: int, n_chls: int, n_time_points: int, list_names: list, name: str = "FamUnfam", tsss: int = 10):
    """
    Reads epoched MEG data and transforms it into a matrix with the shape [n_sub, n_conditions, n_channels, n_time_points].

    Parameters:
    -----------
    sub_id : list
        List of subject IDs.
    n_cons : int
        Number of conditions.
    n_chls : int
        Number of channels.
    n_time_points : int
        Number of time points.
    list_names : list
        List of condition names.
    name : str, optional
        Name of the MEG data (default: "FamUnfam").
    tsss : int, optional
        TSSS value (default: 10).

    Returns:
    --------
    megdata : array
        The transformed MEG data matrix.
        Shape: [n_cons, n_subs, n_chls, n_ts].
    """

    megdata = np.zeros([len(sub_id), n_cons, n_chls, n_time_points], dtype=np.float32)
    subindex = 0
    for subject_id in sub_id:
        subject = f"sub-{subject_id:02d}"
        data_path = op.join(meg_dir, subject)
        file = op.join(data_path, f'{subject}-tsss_{tsss}_{name}-epo.fif')
        epochs = mne.read_epochs(file)
        subdata = np.zeros([n_cons, n_chls, n_time_points], dtype=np.float32)
        for j in range(n_cons):
            epoch = epochs[list_names[j]]
            subdata[j] = epoch.get_data()
        megdata[subindex] = subdata
        subindex += 1
    return megdata
def compute_RDMs(megdata: np.array, n_cons: int, n_subj: int, n_trials: int, n_chls: int, n_time_points: int,
                 name: str = "FamUnfam", sub_opt: int = 1, chl_opt: int = 1, save: bool = True) -> np.array:
    """
    Computes RDMs for MEG data and saves them in a numpy file.

    Parameters:
    -----------
    megdata : array
        The MEG data array.
        Shape: [n_cons, n_subs, n_chls, n_ts], where:
        - n_cons: number of conditions
        - n_subs: number of subjects
        - n_chls: number of channels
        - n_ts: number of time-points
    n_cons : int
        Number of conditions.
    n_subj : int
        Number of subjects.
    n_trials : int
        Number of trials.
    n_chls : int
        Number of channels.
    n_time_points : int
        Number of time points.
    name : str, optional
        Name of the RDM (default: "FamUnfam").
    sub_opt : int, optional
        Subject option (0 for single RDM, 1 for multiple RDMs) (default: 1).
    chl_opt : int, optional
        Channel option (0 for single RDM, 1 for multiple RDMs) (default: 1).
    save : bool, optional
        Flag indicating whether to save the RDMs to a file (default: True).

    Returns:
    --------
    RDMs : array
        The MEG RDM(s) based on the specified options.

        If sub_opt=0 & chl_opt=0, return only one RDM.
            Shape: [n_cons, n_cons].
        If sub_opt=0 & chl_opt=1, return n_chls RDMs.
            Shape: [n_chls, n_cons, n_cons].
        If sub_opt=1 & chl_opt=0, return n_subj RDMs.
            Shape: [n_subs, n_cons, n_cons].
        If sub_opt=1 & chl_opt=1, return n_subj*n_chls RDMs.
            Shape: [n_subs, n_chls, n_cons, n_cons].
    """
    megdata = np.transpose(megdata, (1, 0, 2, 3))
    megdata = np.reshape(megdata, [n_cons, n_subj, n_trials, n_chls, n_time_points])
    rdm = eegRDM(megdata, sub_opt, chl_opt)

    if save:
        out_file = op.join(meg_dir, f'RDMs_{name}_{n_subj}-subject_{sub_opt}-sub_opt{chl_opt}-chl_opt.npy')
        save_npy(out_file, rdm)
    return rdm

def compute_across_time_RDMs(megdata: np.array, n_cons: int, n_subj: int, n_trials: int, n_chls: int, n_time_points: int, 
            time_window: int, name: str = "FamUnfam", sub_opt: int = 1, chl_opt: int = 1, save: bool = True) -> np.array:
    """
    Computes MEG RDMs across time.

    Parameters:
    -----------
    megdata : array
        MEG data array of shape [n_cons, n_subs, n_chls, n_ts].
    n_cons : int
        Number of conditions.
    n_subj : int
        Number of subjects.
    n_trials : int
        Number of trials.
    n_chls : int
        Number of channels.
    n_time_points : int
        Total number of time points.
    t : int
        Length of each time window for RDM calculation.
    name : str, optional
        Name of the RDM (default: "FamUnfam").
    sub_opt : int, optional
        Subject option (0 for single RDM, 1 for multiple RDMs) (default: 1).
    chl_opt : int, optional
        Channel option (0 for single RDM, 1 for multiple RDMs) (default: 1).
    save : bool, optional
        Flag indicating whether to save the RDMs to a file (default: True).

    Returns:
    --------
    RDMs : array
        The MEG RDM(s) movie based on the specified options.
            Shape: [n_rdms, n_cons, n_cons].
    """
    n_rdms = n_time_points // time_window
    across_time_dir = op.join(meg_dir, "across_time", name)
    if not op.isdir(across_time_dir):
        os.makedirs(across_time_dir)
    for i in tqdm(range(n_rdms), desc="Calculating across time RDMs"):
        sub_megdata = megdata[..., i*time_window:i*time_window+time_window]
        out_file = op.join(across_time_dir, f'RDMs_{i}_{name}_{n_subj}-subject_{sub_opt}-sub_opt_{chl_opt}-chl_opt.npy')
        if not op.isfile(out_file):
            rdm = compute_RDMs(sub_megdata, n_cons, n_subj, n_trials, n_chls, time_window, name, sub_opt, chl_opt, False)
            np.save(out_file, rdm)

def get_list_names(stimuli_file_name: str) -> list:
    """
    Returns a list of condition names based on the given stimuli file name.

    Parameters:
    -----------
    stimuli_file_name : str
        Name of the stimuli file.

    Returns:
    --------
    list_names : list
        List of condition names.
    """

    list_names = []
    if stimuli_file_name == "FamUnfam":
        return [str(i) for i in range(1, 301)]
    elif stimuli_file_name == "Fam":
        return [str(i) for i in range(1, 151)]
    elif stimuli_file_name == "Unfam":
        return [str(i) for i in range(151, 301)]
    elif stimuli_file_name == "Scram":
        return [str(i) for i in range(301, 451)]
    elif stimuli_file_name == "FamScram":
        list_names = [str(i) for i in range(1, 151)]
        for i in range(301, 451):
            list_names.append(str(i))
        return list_names

if __name__ == '__main__':
    parser = get_similarity_parser()
    args = parser.parse_args()


    list_names = get_list_names(args.stimuli_file_name)
    sub_id = [i for i in range(1,17)]
    name = args.stimuli_file_name
    if args.band != None:
        name = f"{args.stimuli_file_name}_{args.band}"

    megdata = transform_data(sub_id, args.cons, 306, 881, list_names, name)

    if args.type_meg_rdm == "basic":
        rdm = compute_RDMs(megdata, args.cons, len(sub_id), 1, 306, 881, name)
    if args.type_meg_rdm == "across_time":
        compute_across_time_RDMs(megdata, args.cons, len(sub_id), 1, 306, 881, args.time_window, name)

