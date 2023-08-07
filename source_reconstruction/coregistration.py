import os
import sys
import pickle
import numpy as np

import mne

from sklearn.model_selection import KFold

sys.path.append('../../MFRS/')
from utils.config import study_path, subjects_dir, spacing, meg_dir, mindist
from utils.arg_parser import source_rescontruction_parser

def compute_coregistration(fname_trans, subject, overwrite):
    #Fitting coregistration matrix
    print(" ========> computing registration matrix")
    fname_raw = os.path.join(study_path, f"ds000117/{subject}/ses-meg/meg", f"{subject}_ses-meg_task-facerecognition_run-01_meg.fif")
    info = mne.io.read_info(fname_raw)
    coreg = mne.coreg.Coregistration(info, subject, subjects_dir, fiducials="estimated")
    coreg.fit_fiducials(verbose=True)
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
    coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    mne.write_trans(fname_trans, coreg.trans, overwrite=overwrite)
    return coreg

def compute_cov(fname_cov, epochs, overwrite, tmax=0, kfolds=3):
    print(" ========> computing cov matrix")
    cv = KFold(kfolds)
    cov = mne.compute_covariance(epochs, tmax=tmax, method='shrunk', cv=cv)
    cov.save(fname_cov, overwrite=overwrite)
    return cov

def compute_forward_solution(fname_fwd, fname_src, fname_epo, fname_trans, fname_bem, mindist, overwrite):
    print(" ========> computing forward solution")
    src = mne.read_source_spaces(fname_src)
    info = mne.io.read_info(fname_epo)
    fwd = mne.make_forward_solution(info, fname_trans, fname_src, fname_bem, meg=True, eeg=False, mindist=mindist)
    mne.write_forward_solution(fname_fwd, fwd, overwrite=overwrite)
    print(f"Before: {src}")
    print(f'After:  {fwd["src"]}')
    leadfield = fwd["sol"]["data"]
    print(f"Leadfield size : {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")
    return fwd

def compute_inverse_problem(fname_inv, epochs, fwd, cov, overwrite):
    print(" ========> computing inverse operator")
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
    mne.minimum_norm.write_inverse_operator(fname_inv, inverse_operator, overwrite=overwrite)
    return inverse_operator

def compute_source_estimates(fname_stc, epochs, inverse_operator, method):
    print(" ========> computing source estimates")
    snr = 3.0
    lambda2 = 1.0 / snr**2
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2,
        method=method,
        pick_ori='vector',
        verbose=True,
    )
    with open(fname_stc, 'wb') as f:
        pickle.dump(stcs, f)


if __name__=="__main__":
    parser = source_rescontruction_parser()
    args = parser.parse_args()
    subject = f"sub-{args.subject:02d}"

    fname_trans = os.path.join(subjects_dir, f'{subject}/{subject}-trans.fif')
    fname_src = os.path.join(subjects_dir, subject, 'bem', f'{subject}-{spacing}-src.fif')
    fname_bem = os.path.join(subjects_dir, subject, 'bem', f'{subject}-5120-bem-sol.fif')     # we use 1-layer BEM because the 3-layer is unreliable
    fname_epo = os.path.join(meg_dir, subject, f'{subject}-tsss_10_{args.stimuli_file_name}-epo.fif')
    fname_fwd = os.path.join(meg_dir, subject, f'{subject}-meg-eeg-{spacing}-fwd.fif' )
    fname_cov = os.path.join(meg_dir, subject, f'{subject}-tsss_10_{args.stimuli_file_name}-cov.fif')
    fname_inv = os.path.join(meg_dir, subject, f"{subject}-tsss_10_{args.stimuli_file_name}-meg-{spacing}-inv.fif")
    fname_stc = os.path.join(meg_dir, subject, f'{args.method}_src_tsss_10_{args.stimuli_file_name}.pkl')

    if not os.path.isfile(fname_trans) or args.overwrite:
        coreg = compute_coregistration(fname_trans, subject, args.overwrite)
    else:
        print(" ========> loading registration matrix")
        coreg = mne.read_trans(fname_trans)

    epochs = mne.read_epochs(fname_epo, preload=True)

    if not os.path.isfile(fname_cov) or args.overwrite:
        cov = compute_cov(fname_cov, epochs, args.overwrite)
    else:
        print(" ========> loading cov matrix")
        cov = mne.read_cov(fname_cov)

    if not os.path.isfile(fname_fwd) or args.overwrite:
        fwd = compute_forward_solution(fname_fwd, fname_src, fname_epo, fname_trans, fname_bem, mindist, args.overwrite)
    else:
        print(" ========> loading forward solution")
        fwd = mne.read_forward_solution(fname_fwd)

    if not os.path.isfile(fname_inv) or args.overwrite:
        inverse_operator = compute_inverse_problem(fname_inv, epochs, fwd, cov, args.overwrite)
    else:
        print(" ========> loading inverse operatpr")
        inverse_operator = mne.minimum_norm.read_inverse_operator(fname_inv)

    if not os.path.isfile(fname_stc) or args.overwrite:
        stcs = compute_source_estimates(fname_stc, epochs, inverse_operator, args.method)
    else:
        print(" ========> loading source estimates")
        with open(fname_stc, 'rb') as f:
            stcs = pickle.load(f)



