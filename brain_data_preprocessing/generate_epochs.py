import sys
sys.path.append('../../MFRS/')
from epoching_utils import run_events, run_epochs
from utils.library.config_bids import  N_JOBS
from mne.parallel import parallel_func

if __name__ == '__main__':

    conditions_mapping ={}
    for i in range (1,151):
        conditions_mapping["meg/f%03d.bmp"%i]=i
        conditions_mapping["meg/u%03d.bmp"%i]=i+150
        conditions_mapping["meg/s%03d.bmp"%i]=i+300

    # Get events
    parallel, run_func, _ = parallel_func(run_events, n_jobs=N_JOBS)
    parallel(run_func(subject_id, [[ 'FamUnfam'], [5,6,7, 13,14,15,]], 'FamUnfam', conditions_mapping) for subject_id in range(1, 17))

    # Create epochs
    parallel, run_func, _ = parallel_func(run_epochs, n_jobs=max(N_JOBS // 4, 1))
    parallel(run_func(subject_id, 'FamUnfam', 10) for subject_id in range(1, 17))
