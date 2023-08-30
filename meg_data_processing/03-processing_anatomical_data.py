import os
import time
import shutil
import subprocess
import numpy as np
from typing import List

import mne
import nibabel as nib

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path, subjects_dir, spacing
from utils.arg_parser import source_rescontruction_parser

# Import the logging module and configure it
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tee_output(command: List[str], log_file: str) -> None:
    """
    Run a command and log the output to a file.

    Parameters:
    -----------
    command : List[str]
        List of strings representing the command and its arguments to be executed.
    log_file : str
        File path to log the output of the command.

    Raises:
    -------
    RuntimeError
        If the command execution returns a non-zero exit code.

    """
    # Join the command elements to create a readable command string
    to_print = " ".join(command)
    logger.info("Running:\n")
    logger.info(to_print)

    # Open the log_file for writing the output of the command
    with open(log_file, 'wb') as fid:
        # Start a subprocess and capture its output
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            # Write the captured output to the log_file
            fid.write(line)

    # Check the command's exit code, raise an error if it's non-zero
    if proc.wait() != 0:
        raise RuntimeError('Command failed')


def process_subject_anat(subject_id: int) -> None:
    """
    Process the anatomical data for a subject using recon-all and create BEM and source space.

    Parameters:
    -----------
    subject_id : int
        Subject ID (integer).

    Raises:
    -------
    RuntimeError
        If the reconstruction or the FLASH files conversion fails.
    """

    subject = f"sub-{subject_id:02d}"
    logger.info(f"Processing {subject}")

    t1_fname = os.path.join(study_path, 'ds000117', subject, 'ses-mri/anat',
                       f'{subject}_ses-mri_acq-mprage_T1w.nii.gz')
    log_fname = os.path.join(study_path, 'ds000117', subject, 'my-recon-all.txt')
    subject_dir = os.path.join(subjects_dir, subject)

    if os.path.isdir(subject_dir):
        logger.info('Skipping reconstruction (folder exists)')
    else:
        logger.info('Running reconstruction (usually takes hours)')
        t0 = time.time()
        tee_output(
            ['recon-all', '-all', '-s', subject, '-sd', subjects_dir,
             '-i', t1_fname], log_fname)
        logger.info(f'Recon for {subject} complete in {((time.time() - t0) / 60. / 60.):0.1f} hours')

    bem_dir = os.path.join(subjects_dir, subject, 'bem')

    surf_name = 'inner_skull.surf'
    sbj_inner_skull_fname = os.path.join(bem_dir, subject + '-' + surf_name)
    inner_skull_fname = os.path.join(bem_dir, surf_name)


    # check if inner_skull surf exists, if not BEM computation is
    # performed by MNE python functions mne.bem.make_watershed_bem
    if not (os.path.isfile(sbj_inner_skull_fname) or
            os.path.isfile(inner_skull_fname)):
        logger.info(f"{inner_skull_fname} ---> FILE NOT FOUND!!!---> BEM computed")
        mne.bem.make_watershed_bem(subject, subjects_dir, overwrite=True, verbose=True)
    else:
        logger.info(f"\n inner skull {inner_skull_fname} surface exists!!!\n")

    # Create a BEM model for a subject        
    fname_bem_surfaces = os.path.join(bem_dir, f'{subject}-5120-bem.fif')
    if not os.path.isfile(fname_bem_surfaces):
        logger.info(f'Setting up 1-layer BEM')
        conductivity = (0.3, 0.006, 0.3)[:1]
        try:
            bem_surfaces = mne.make_bem_model(
                subject, ico=4, conductivity=conductivity,
                subjects_dir=subjects_dir)
        except RuntimeError as exp:
            logger.info(f'FAILED to create 1-layer BEM for {subject}: {exp.args[0]}')
        # Write BEM surfaces to a fiff file
        mne.write_bem_surfaces(fname_bem_surfaces, bem_surfaces)

    # Create a BEM solution using the linear collocation approach
    fname_bem = os.path.join(bem_dir, f'{subject}-5120-bem-sol.fif')
    if not os.path.isfile(fname_bem):
        logger.info('Computing 1-layer BEM solution')
        bem_model = mne.read_bem_surfaces(fname_bem_surfaces)
        bem = mne.make_bem_solution(bem_model)
        mne.write_bem_solution(fname_bem, bem)

    # Create the surface source space
    fname_src = os.path.join(subjects_dir, subject, 'bem', f'{subject}-{spacing}-src.fif')
    if not os.path.isfile(fname_src):
        logger.info('Setting up source space')
        src = mne.setup_source_space(subject, spacing, subjects_dir=subjects_dir)
        mne.write_source_spaces(fname_src, src)

if __name__=="__main__":
    parser = source_rescontruction_parser()
    args = parser.parse_args()
    
    # Run the process_subject_anat function in parallel for all subjects
    subject_id = args.subject
    process_subject_anat(subject_id)
 
    # Now we do something special for fsaverage
    fsaverage_src_dir = os.path.join(os.environ['FREESURFER_HOME'], 'subjects', 'fsaverage')
    fsaverage_dst_dir = os.path.join(subjects_dir, 'fsaverage')

    if os.path.exists(fsaverage_dst_dir):
        logger.info('Copying fsaverage into subjects directory')  # to allow writing in folder
        os.unlink(fsaverage_dst_dir)  # remove symlink
        shutil.copytree(fsaverage_src_dir, fsaverage_dst_dir)

        fsaverage_bem = os.path.join(fsaverage_dst_dir, 'bem')
        if not os.path.isdir(fsaverage_bem):
            os.mkdir(fsaverage_bem)

        fsaverage_src = os.path.join(fsaverage_bem, 'fsaverage-5-src.fif')
        if not os.path.isfile(fsaverage_src):
            logger.info('Setting up source space for fsaverage')
            src = mne.setup_source_space('fsaverage', 'ico5', subjects_dir=subjects_dir)
            for s in src:
                assert np.array_equal(s['vertno'], np.arange(10242))
            mne.write_source_spaces(fsaverage_src, src)
    else:
        logger.warning("fsaverage_dst_dir does not exist, skipping copying of fsaverage")
