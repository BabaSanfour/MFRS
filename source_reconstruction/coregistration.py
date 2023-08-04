import os
import sys
import mne
sys.path.append('../../MFRS/')
from utils.config import study_path, subjects_dir
from utils.arg_parser import source_rescontruction_parser


if __name__=="__main__":
    parser = source_rescontruction_parser()
    args = parser.parse_args()
    subject = f"sub-{args.subject:02d}"

    fname_raw = os.path.join(study_path, f"ds000117/{subject}/ses-meg/meg", f"{subject}_ses-meg_task-facerecognition_run-01_meg.fif")
    info = mne.io.read_info(fname_raw)
    fiducials = "estimated"  # get fiducials from fsaverage
    coreg = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials=fiducials)
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
    coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    mne.write_trans(os.path.join(subjects_dir,f'{subject}/{subject}-trans.fif'), coreg.trans)





