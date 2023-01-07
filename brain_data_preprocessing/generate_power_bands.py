import sys
sys.path.append('../../MFRS/')
from epoching_utils import run_epochs
from utils.library.config_bids import  N_JOBS
from mne.parallel import parallel_func

if __name__ == "__main__":

    FREQ_BANDS = {"delta": [1.5, 4.5],
              "theta": [4.5, 8.5],
              "alpha": [8.5, 15.5],
              "beta": [15.5, 30],
              "Gamma": [30, 70]}

    subjects_ids=[i for i in range(1,17)]
    # Create epochs
    for  frequency_band, f in FREQ_BANDS.items():
        parallel, run_func, _ = parallel_func(run_epochs, n_jobs=max(N_JOBS // 4, 1))
        parallel(run_func(subject_id, 'FamUnfam', 10, f[0], f[1], frequency_band) for subject_id in range(1, 17))