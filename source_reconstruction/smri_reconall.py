import os
import time
import glob
import shutil
import subprocess

import numpy as np
import sys
import mne
from mne.parallel import parallel_func
import nibabel as nib
sys.path.append('../../MFRS/')
from utils.config import study_path, subjects_dir, spacing
from utils.arg_parser import source_rescontruction_parser

def tee_output(command, log_file):
    to_print = " ".join(command)
    print("Running :\n")
    print(to_print)
    with open(log_file, 'wb') as fid:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            # print(line.decode('utf-8'))
            fid.write(line)
    if proc.wait() != 0:
        raise RuntimeError('Command failed')

def convert_to_mgz(input_file, output_file, fa, log_file=None):
    """
    Convert NIfTI file to MGZ format using mri_convert.

    Parameters:
        input_file (str): Path to the input NIfTI file.
        output_file (str): Path to the output MGZ file.
        log_file (str, optional): Path to the log file. If provided, the output will be logged to this file.
    """
    log_fname = os.path.join(study_path, 'ds000117', 'my-recon-all.txt')
    cmd = ['mri_convert', input_file, output_file, "-flip_angle", fa]
    tee_output(cmd, log_fname)

def process_subject_anat(subject_id, force_recon_all=False):
    subject = f"sub-{subject_id:02d}"
    print(f"Processing {subject}")

    t1_fname = os.path.join(study_path, 'ds000117', subject, 'ses-mri/anat',
                       f'{subject}_ses-mri_acq-mprage_T1w.nii.gz')
    log_fname = os.path.join(study_path, 'ds000117', subject, 'my-recon-all.txt')
    subject_dir = os.path.join(subjects_dir, subject)

    if os.path.isdir(subject_dir):
        print('  Skipping reconstruction (folder exists)')
    else:
        print('  Running reconstruction (usually takes hours)')
        t0 = time.time()
        tee_output(
            ['recon-all', '-all', '-s', subject, '-sd', subjects_dir,
             '-i', t1_fname], log_fname)
        print(f'  Recon for {subject} complete in {((time.time() - t0) / 60. / 60.):0.1f} hours')


    # Move flash data
    fnames = glob.glob(os.path.join(study_path, 'ds000117', subject, 'ses-mri/anat',
                               '*FLASH*'))
    dst_flash = os.path.join(subjects_dir, subject, 'mri', 'flash')
    if not os.path.isdir(dst_flash):
        print('  Converting FLASH files')
        os.makedirs(dst_flash)
        for f_src in fnames:
            # Extract run number and echo number from the source file name
            run_number = int(f_src.split('_run-')[1].split('_')[0])
            echo_number = int(f_src.split('echo-')[1].split('_')[0])

            # Define a dictionary to map run numbers to prefixes
            run_prefixes = {1: "mef05", 2: "mef30"}
            fa = {1: "05", 2: "30"}
            # Check if the run number is valid and construct the destination file name
            if run_number in run_prefixes:
                f_dst = os.path.join(dst_flash, f"{run_prefixes[run_number]}_{echo_number:03d}.mgz")
            else:
                raise ValueError("Invalid run number")

            convert_to_mgz(f_src,  f_dst, fa[run_number])

    # Fix the headers for subject 13
    if subject_id == 13:
        print('  Fixing FLASH files for %s' % (subject,))
        fnames = (['mef05_%d.mgz' % x for x in range(7)] +
                  ['mef30_%d.mgz' % x for x in range(7)])
        for fname in fnames:
            dest_fname = os.path.join(dst_flash, fname)
            dest_img = nib.load(os.path.splitext(dest_fname)[0] + '.nii.gz')

            # Copy the headers from subjects 1
            src_img = nib.load(os.path.join(
                subjects_dir, "sub001", "mri", "flash", fname))
            hdr = src_img.header
            fixed = nib.MGHImage(dest_img.get_data(), dest_img.affine, hdr)
            nib.save(fixed, dest_fname)

    # Make BEMs
    if not os.path.isfile(os.path.join(subject_dir, "mri/flash/parameter_maps/flash5.mgz")):
        print('  Converting flash MRIs')
        mne.bem.convert_flash_mris(subject,
                                   subjects_dir=subjects_dir, verbose=True)
    if not os.path.isfile(os.path.join(subject_dir, "bem/flash/outer_skin.surf")):
        print('  Making BEM')
        mne.bem.make_flash_bem(subject, subjects_dir=subjects_dir,
                               show=False, verbose=True)
    for n_layers in (1, 3):
        extra = '-'.join(['5120'] * n_layers)
        fname_bem_surfaces = os.path.join(subjects_dir, subject, 'bem',
                                     '%s-%s-bem.fif' % (subject, extra))
        if not os.path.isfile(fname_bem_surfaces):
            print('  Setting up %d-layer BEM' % (n_layers,))
            conductivity = (0.3, 0.006, 0.3)[:n_layers]
            try:
                bem_surfaces = mne.make_bem_model(
                    subject, ico=4, conductivity=conductivity,
                    subjects_dir=subjects_dir)
            except RuntimeError as exp:
                print('  FAILED to create %d-layer BEM for %s: %s'
                      % (n_layers, subject, exp.args[0]))
                continue
            mne.write_bem_surfaces(fname_bem_surfaces, bem_surfaces)
        fname_bem = os.path.join(subjects_dir, subject, 'bem',
                            '%s-%s-bem-sol.fif' % (subject, extra))
        if not os.path.isfile(fname_bem):
            print('  Computing  %d-layer BEM solution' % (n_layers,))
            bem_model = mne.read_bem_surfaces(fname_bem_surfaces)
            bem = mne.make_bem_solution(bem_model)
            mne.write_bem_solution(fname_bem, bem)

    # Create the surface source space
    fname_src = os.path.join(subjects_dir, subject, 'bem', '%s-%s-src.fif'
                        % (subject, spacing))
    if not os.path.isfile(fname_src):
        print('  Setting up source space')
        src = mne.setup_source_space(subject, spacing,
                                     subjects_dir=subjects_dir)
        mne.write_source_spaces(fname_src, src)

if __name__=="__main__":
    parser = source_rescontruction_parser()
    args = parser.parse_args()

    parallel, run_func, _ = parallel_func(process_subject_anat, n_jobs=1)
    # parallel(run_func(args.subject))
    #TODO: TEST THE REST AFTER RECON ALL FINISH RUNNING
 
    # now we do something special for fsaverage
    fsaverage_src_dir = os.path.join(os.environ['FREESURFER_HOME'], 'subjects', 'fsaverage')
    fsaverage_dst_dir = os.path.join(subjects_dir, 'fsaverage')

    print('Copying fsaverage into subjects directory')  # to allow writting in folder
    os.unlink(fsaverage_dst_dir)  # remove symlink
    shutil.copytree(fsaverage_src_dir, fsaverage_dst_dir)

    fsaverage_bem = os.path.join(fsaverage_dst_dir, 'bem')
    if not os.path.isdir(fsaverage_bem):
        os.mkdir(fsaverage_bem)

    fsaverage_src = os.path.join(fsaverage_bem, 'fsaverage-5-src.fif')
    if not os.path.isfile(fsaverage_src):
        print('Setting up source space for fsaverage')
        src = mne.setup_source_space('fsaverage', 'ico5',
                                    subjects_dir=subjects_dir)
        for s in src:
            assert np.array_equal(s['vertno'], np.arange(10242))
        mne.write_source_spaces(fsaverage_src, src)
