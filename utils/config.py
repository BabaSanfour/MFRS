"""
============================
Config file the whole study
============================
"""

import os
import sys
import mne
import numpy as np
from distutils.version import LooseVersion


###############################################################################
# Let's set the path where the data is downloaded and stored.
sys.path.append('../../MFRS')
user = os.path.expanduser('~')
scratch_folder = os.path.join(user, "scratch")
if not os.path.isdir(scratch_folder):
    print("Scratch folder does not exists!!!")
    print("Creating Scratch folder.")
    os.makedirs(scratch_folder)

proj_path = os.path.join(user, "MFRS/")
sys.path.append(proj_path)

study_path = os.path.join(scratch_folder, "data/MFRS_data")
if not os.path.isdir(study_path):
    print("Data folder does not exists!!!")
    print("Creating Data folder.")
    print("Please make sure to download the data before running any experiment.")
    os.makedirs(study_path)

weights_path = os.path.join(scratch_folder, "net_weights")
if not os.path.isdir(weights_path):
    print("Weights folder does not exists!!!")
    print("Creating Weights folder.")
    os.makedirs(weights_path)

results_path = os.path.join(study_path, "results")
if not os.path.isdir(results_path):
    print("Results folder does not exists!!!")
    print("Creating Results folder.")
    os.makedirs(results_path)

similarity_folder = os.path.join(study_path, 'similarity_scores')
if not os.path.isdir(similarity_folder):
    os.makedirs(similarity_folder)

activations_folder = os.path.join(study_path, 'activations')
if not os.path.isdir(activations_folder):
    os.makedirs(activations_folder)

rdms_folder = os.path.join(study_path, 'networks_rdms')
if not os.path.isdir(rdms_folder):
    os.makedirs(rdms_folder)

plots_path = os.path.join(scratch_folder, "plots")
if not os.path.isdir(plots_path):
    print("Plots folder does not exists!!!")
    print("Creating Plots folder.")
    os.makedirs(plots_path)

subjects_dir = os.path.join(study_path, 'subjects')
if not os.path.isdir(subjects_dir):
    os.makedirs(subjects_dir)

meg_dir = os.path.join(study_path, 'MEG')
if not os.path.isdir(meg_dir):
    os.makedirs(meg_dir)


os.environ["SUBJECTS_DIR"] = subjects_dir

spacing = 'oct6'
mindist = 5
N_JOBS = 1
###############################################################################
# Some mapping betwen filenames for bad sensors and subjects

map_subjects = {1: 'subject_02', 2: 'subject_03', 3: 'subject_06',
                4: 'subject_08', 5: 'subject_09', 6: 'subject_10',
                7: 'subject_11', 8: 'subject_12', 9: 'subject_14',
                10: 'subject_15', 11: 'subject_17', 12: 'subject_18',
                13: 'subject_19', 14: 'subject_23', 15: 'subject_24',
                16: 'subject_25'}


###############################################################################
# Subjects that are known to be bad from the publication

exclude_subjects = [1, 5, 16]  # Excluded subjects

###############################################################################
# The `cross talk file <https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/results/library/ct_sparse.fif>`_
# and `calibration file <https://github.com/mne-tools/mne-biomag-group-demo/blob/master/scripts/results/library/sss_cal.dat>`_
# are placed in the same folder.

ctc = os.path.join(os.path.dirname(__file__), 'ct_sparse.fif')
cal = os.path.join(os.path.dirname(__file__), 'sss_cal.dat')

ylim = {'eeg': [-10, 10], 'mag': [-300, 300], 'grad': [-80, 80]}


def set_matplotlib_defaults():
    import matplotlib.pyplot as plt
    fontsize = 8
    params = {'axes.labelsize': fontsize,
              'legend.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize + 2,
              'figure.max_open_warning': 200,
              'axes.spines.top': False,
              'axes.spines.right': False,
              'axes.grid': True,
              'lines.linewidth': 1,
              }
    import matplotlib
    if LooseVersion(matplotlib.__version__) >= '2':
        params['font.size'] = fontsize
    else:
        params['text.fontsize'] = fontsize
    plt.rcParams.update(params)


annot_kwargs = dict(fontsize=12, fontweight='bold',
                    xycoords="axes fraction", ha='right', va='center')
l_freq = None

tmin = -0.2
tmax = 2.9  # min duration between onsets: (400 fix + 800 stim + 1700 ISI) ms
reject_tmax = 0.8  # duration we really care about
random_state = 42

smooth = 10

fsaverage_vertices = [np.arange(10242), np.arange(10242)]

# Get whole sensors list + sensors lists per type + sesnsor positions for topomaps + meg file
subject_id, run=1, 1
subject = "sub-%02d" % subject_id
in_path = os.path.join(study_path, 'ds000117', subject, 'ses-meg/meg')
if not os.path.isdir(in_path):
    print("Please make sure to download the ds000117 dataset.")

run_fname = os.path.join(in_path, 'sub-%02d_ses-meg_task-facerecognition_run-%02d_meg.fif' % (
            subject_id,run))
raw = mne.io.read_raw_fif(run_fname)
meg_sensors=raw.pick_types(meg=True).info.ch_names
info = mne.io.read_info(run_fname)
channels_mag=raw.copy().pick_types(meg='mag').info.ch_names
channels_grad1=raw.copy().pick_types(meg='planar1').info.ch_names
channels_grad2=raw.copy().pick_types(meg='planar2').info.ch_names
sensors_position=raw.copy().pick_types(meg='mag').info
out_file = os.path.join(meg_dir, "RDMs_FamUnfam_16-subject_1-sub_opt1-chl_opt.npy")
meg_rdm = np.load(out_file)

# for plotting
mask_params=dict(marker='*', markerfacecolor='w', markeredgecolor='k',
        linewidth=0, markersize=15)


# Main layers for networks // extracted manually
resnet_layers= [
 'model.relu',  'model.block1.0.bn1', 'model.block1.0.bn2', 'model.block1.0.relu', 'model.block1.1.bn1', 'model.block1.1.bn2', 'model.block1.1.relu', 'model.block1.2.bn1',
 'model.block1.2.bn2', 'model.block1.2.relu', 'model.block2.0.bn1', 'model.block2.0.bn2', 'model.block2.0.relu', 'model.block2.1.bn1', 'model.block2.1.bn2', 'model.block2.1.relu',
 'model.block2.2.bn1', 'model.block2.2.bn2', 'model.block2.2.relu', 'model.block2.3.bn1', 'model.block2.3.bn2', 'model.block2.3.relu', 'model.block3.0.bn1', 'model.block3.0.bn2',
 'model.block3.0.relu', 'model.block3.1.bn1', 'model.block3.1.bn2', 'model.block3.1.relu', 'model.block3.2.bn1', 'model.block3.2.bn2', 'model.block3.2.relu', 'model.block3.3.bn1',
 'model.block3.3.bn2', 'model.block3.3.relu', 'model.block3.4.bn1', 'model.block3.4.bn2', 'model.block3.4.relu', 'model.block3.5.bn1', 'model.block3.5.bn2',  'model.block3.5.relu',
 'model.block4.0.bn1',  'model.block4.0.bn2', 'model.block4.0.relu', 'model.block4.1.bn1', 'model.block4.1.bn2', 'model.block4.1.relu',  'model.block4.2.bn1',  'model.block4.2.bn2',
 'model.block4.2.relu']

cornet_s_layers = [
 'model.V1.nonlin1', 'model.V1.nonlin2', 'model.V2.conv_input', 'model.V2.norm1_0', 'model.V2.norm2_0', 'model.V2.norm3_0', 'model.V2.norm1_1',  'model.V2.norm2_1',
 'model.V2.norm3_1', 'model.V4.conv_input', 'model.V4.norm1_0', 'model.V4.norm2_0', 'model.V4.norm3_0', 'model.V4.norm1_1', 'model.V4.norm2_1', 'model.V4.norm3_1',
 'model.V4.norm1_2', 'model.V4.norm2_2', 'model.V4.norm3_2', 'model.V4.norm1_3', 'model.V4.norm2_3', 'model.V4.norm3_3', 'model.IT.conv_input', 'model.IT.norm1_0',
 'model.IT.norm2_0', 'model.IT.norm3_0', 'model.IT.norm1_1', 'model.IT.norm2_1', 'model.IT.norm3_1']

facenet_layers = [
 'conv2d_1a.relu', 'conv2d_2a.relu', 'conv2d_2b.relu', 'conv2d_3b.relu', 'conv2d_4a.relu', 'conv2d_4b.relu', 'repeat_1.0.branch0.relu', 'repeat_1.0.branch1.0.relu',
 'repeat_1.0.branch1.1.relu', 'repeat_1.0.branch2.0.relu', 'repeat_1.0.branch2.1.relu', 'repeat_1.0.branch2.2.relu', 'repeat_1.0.relu', 'repeat_1.1.branch0.relu',
 'repeat_1.1.branch1.0.relu', 'repeat_1.1.branch1.1.relu', 'repeat_1.1.branch2.0.relu', 'repeat_1.1.branch2.1.relu', 'repeat_1.1.branch2.2.relu', 'repeat_1.1.relu',
 'repeat_1.2.branch0.relu', 'repeat_1.2.branch1.0.relu', 'repeat_1.2.branch1.1.relu', 'repeat_1.2.branch2.0.relu', 'repeat_1.2.branch2.1.relu', 'repeat_1.2.branch2.2.relu',
 'repeat_1.2.relu', 'repeat_1.3.branch0.relu', 'repeat_1.3.branch1.0.relu', 'repeat_1.3.branch1.1.relu', 'repeat_1.3.branch2.0.relu', 'repeat_1.3.branch2.1.relu',
 'repeat_1.3.branch2.2.relu', 'repeat_1.3.relu', 'repeat_1.4.branch0.relu', 'repeat_1.4.branch1.0.relu', 'repeat_1.4.branch1.1.relu', 'repeat_1.4.branch2.0.relu',
 'repeat_1.4.branch2.1.relu', 'repeat_1.4.branch2.2.relu', 'repeat_1.4.relu', 'mixed_6a.branch0.relu', 'mixed_6a.branch1.0.relu', 'mixed_6a.branch1.1.relu',
 'mixed_6a.branch1.2.relu', 'repeat_2.0.branch0.relu', 'repeat_2.0.branch1.0.relu', 'repeat_2.0.branch1.1.relu', 'repeat_2.0.branch1.2.relu', 'repeat_2.0.relu',
 'repeat_2.1.branch0.relu', 'repeat_2.1.branch1.0.relu', 'repeat_2.1.branch1.1.relu', 'repeat_2.1.branch1.2.relu', 'repeat_2.1.relu', 'repeat_2.2.branch0.relu',
 'repeat_2.2.branch1.0.relu', 'repeat_2.2.branch1.1.relu', 'repeat_2.2.branch1.2.relu', 'repeat_2.2.relu', 'repeat_2.3.branch0.relu', 'repeat_2.3.branch1.0.relu',
 'repeat_2.3.branch1.1.relu', 'repeat_2.3.branch1.2.relu', 'repeat_2.3.relu', 'repeat_2.4.branch0.relu', 'repeat_2.4.branch1.0.relu', 'repeat_2.4.branch1.1.relu',
 'repeat_2.4.branch1.2.relu', 'repeat_2.4.relu', 'repeat_2.5.branch0.relu', 'repeat_2.5.branch1.0.relu', 'repeat_2.5.branch1.1.relu', 'repeat_2.5.branch1.2.relu',
 'repeat_2.5.relu', 'repeat_2.6.branch0.relu', 'repeat_2.6.branch1.0.relu', 'repeat_2.6.branch1.1.relu', 'repeat_2.6.branch1.2.relu', 'repeat_2.6.relu', 'repeat_2.7.branch0.relu',
 'repeat_2.7.branch1.0.relu', 'repeat_2.7.branch1.1.relu', 'repeat_2.7.branch1.2.relu', 'repeat_2.7.relu', 'repeat_2.8.branch0.relu', 'repeat_2.8.branch1.0.relu',
 'repeat_2.8.branch1.1.relu', 'repeat_2.8.branch1.2.relu', 'repeat_2.8.relu', 'repeat_2.9.branch0.relu', 'repeat_2.9.branch1.0.relu', 'repeat_2.9.branch1.1.relu',
 'repeat_2.9.branch1.2.relu', 'repeat_2.9.relu', 'mixed_7a.branch0.0.relu', 'mixed_7a.branch0.1.relu', 'mixed_7a.branch1.0.relu', 'mixed_7a.branch1.1.relu',
 'mixed_7a.branch2.0.relu', 'mixed_7a.branch2.1.relu', 'mixed_7a.branch2.2.relu', 'repeat_3.0.branch0.relu', 'repeat_3.0.branch1.0.relu', 'repeat_3.0.branch1.1.relu',
 'repeat_3.0.branch1.2.relu', 'repeat_3.0.relu',  'repeat_3.1.branch0.relu', 'repeat_3.1.branch1.0.relu', 'repeat_3.1.branch1.1.relu', 'repeat_3.1.branch1.2.relu',
 'repeat_3.1.relu', 'repeat_3.2.branch0.relu', 'repeat_3.2.branch1.0.relu', 'repeat_3.2.branch1.1.relu', 'repeat_3.2.branch1.2.relu', 'repeat_3.2.relu',
 'repeat_3.3.branch0.relu', 'repeat_3.3.branch1.0.relu', 'repeat_3.3.branch1.1.relu', 'repeat_3.3.branch1.2.relu', 'repeat_3.3.relu', 'repeat_3.4.branch0.relu',
 'repeat_3.4.branch1.0.relu', 'repeat_3.4.branch1.1.relu', 'repeat_3.4.branch1.2.relu', 'repeat_3.4.relu', 'block8.branch0.relu', 'block8.branch1.0.relu',
 'block8.branch1.1.relu', 'block8.branch1.2.relu', 'block8.conv2d']

mobilenet_layers = ['model.0.2', 'model.1.conv.0.2','model.1.conv.2', 'model.2.conv.0.2', 'model.2.conv.1.2', 'model.2.conv.3', 'model.3.conv.0.2',
'model.3.conv.1.2', 'model.3.conv.3', 'model.4.conv.0.2', 'model.4.conv.1.2', 'model.4.conv.3', 'model.5.conv.0.2', 'model.5.conv.1.2', 'model.5.conv.3',
'model.6.conv.0.2', 'model.6.conv.1.2', 'model.6.conv.3', 'model.7.conv.0.2', 'model.7.conv.1.2', 'model.7.conv.3', 'model.8.conv.0.2', 'model.8.conv.1.2',
'model.8.conv.3', 'model.9.conv.0.2', 'model.9.conv.1.2', 'model.9.conv.3', 'model.10.conv.0.2', 'model.10.conv.1.2', 'model.10.conv.3', 'model.11.conv.0.2',
'model.11.conv.1.2', 'model.11.conv.3', 'model.12.conv.0.2', 'model.12.conv.1.2', 'model.12.conv.3', 'model.13.conv.0.2', 'model.13.conv.1.2',
'model.13.conv.3', 'model.14.conv.0.2', 'model.14.conv.1.2', 'model.14.conv.3', 'model.15.conv.0.2', 'model.15.conv.1.2', 'model.15.conv.3',
  'model.16.conv.0.2', 'model.16.conv.1.2', 'model.16.conv.3', 'model.17.conv.0.2', 'model.17.conv.1.2', 'model.17.conv.3', 'model.18.2' ]

SphereFace_layers=['relu1_1', 'relu1_2', 'relu1_3', 'relu2_1', 'relu2_2', 'relu2_3', 'relu2_4', 'relu2_5', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4',
'relu3_5', 'relu3_6', 'relu3_7', 'relu3_8', 'relu3_9', 'relu4_1', 'relu4_2', 'relu4_3']

vgg_layers = ['model.2', 'model.5', 'model.9', 'model.12', 'model.16', 'model.19', 'model.22', 'model.26', 'model.29', 'model.32', 'model.36', 'model.39',
 'model.42', 'classifier.1',  'classifier.4']

inception_layers=['Conv2d_1a_3x3.bn', 'Conv2d_2a_3x3.bn', 'Conv2d_2b_3x3.bn', 'Conv2d_3b_1x1.bn', 'Conv2d_4a_3x3.bn', 'Mixed_5b.branch1x1.bn',
'Mixed_5b.branch5x5_1.bn', 'Mixed_5b.branch5x5_2.bn', 'Mixed_5b.branch3x3dbl_1.bn', 'Mixed_5b.branch3x3dbl_2.bn', 'Mixed_5b.branch3x3dbl_3.bn',
'Mixed_5b.branch_pool.bn', 'Mixed_5c.branch1x1.bn', 'Mixed_5c.branch5x5_1.bn', 'Mixed_5c.branch5x5_2.bn', 'Mixed_5c.branch3x3dbl_1.bn',
 'Mixed_5c.branch3x3dbl_2.bn', 'Mixed_5c.branch3x3dbl_3.bn', 'Mixed_5c.branch_pool.bn','Mixed_5d.branch1x1.bn', 'Mixed_5d.branch5x5_1.bn',
'Mixed_5d.branch5x5_2.bn', 'Mixed_5d.branch3x3dbl_1.bn', 'Mixed_5d.branch3x3dbl_2.bn', 'Mixed_5d.branch3x3dbl_3.bn', 'Mixed_5d.branch_pool.bn',
'Mixed_6a.branch3x3.bn', 'Mixed_6a.branch3x3dbl_1.bn', 'Mixed_6a.branch3x3dbl_2.bn', 'Mixed_6a.branch3x3dbl_3.bn', 'Mixed_6b.branch1x1.bn',
'Mixed_6b.branch7x7_1.bn', 'Mixed_6b.branch7x7_2.bn', 'Mixed_6b.branch7x7_3.bn','Mixed_6b.branch7x7dbl_1.bn', 'Mixed_6b.branch7x7dbl_2.bn',
'Mixed_6b.branch7x7dbl_3.bn', 'Mixed_6b.branch7x7dbl_4.bn', 'Mixed_6b.branch7x7dbl_5.bn', 'Mixed_6b.branch_pool.bn', 'Mixed_6c.branch1x1.bn',
'Mixed_6c.branch7x7_1.bn', 'Mixed_6c.branch7x7_2.bn', 'Mixed_6c.branch7x7_3.bn', 'Mixed_6c.branch7x7dbl_1.bn', 'Mixed_6c.branch7x7dbl_2.bn',
'Mixed_6c.branch7x7dbl_3.bn', 'Mixed_6c.branch7x7dbl_4.bn', 'Mixed_6c.branch7x7dbl_5.bn', 'Mixed_6c.branch_pool.bn', 'Mixed_6d.branch1x1.bn',
'Mixed_6d.branch7x7_1.bn', 'Mixed_6d.branch7x7_2.bn', 'Mixed_6d.branch7x7_3.bn', 'Mixed_6d.branch7x7dbl_1.bn', 'Mixed_6d.branch7x7dbl_2.bn',
'Mixed_6d.branch7x7dbl_3.bn', 'Mixed_6d.branch7x7dbl_4.bn', 'Mixed_6d.branch7x7dbl_5.bn', 'Mixed_6d.branch_pool.bn','Mixed_6e.branch1x1.bn',
'Mixed_6e.branch7x7_1.bn', 'Mixed_6e.branch7x7_2.bn', 'Mixed_6e.branch7x7_3.bn', 'Mixed_6e.branch7x7dbl_1.bn', 'Mixed_6e.branch7x7dbl_2.bn',
'Mixed_6e.branch7x7dbl_3.bn', 'Mixed_6e.branch7x7dbl_4.bn', 'Mixed_6e.branch7x7dbl_5.bn', 'Mixed_6e.branch_pool.bn', 'Mixed_7a.branch3x3_1.bn',
'Mixed_7a.branch3x3_2.bn', 'Mixed_7a.branch7x7x3_2.bn', 'Mixed_7a.branch7x7x3_3.bn', 'Mixed_7a.branch7x7x3_4.bn','Mixed_7b.branch1x1.bn',
'Mixed_7b.branch3x3_1.bn', 'Mixed_7b.branch3x3_2a.bn', 'Mixed_7b.branch3x3_2b.bn','Mixed_7b.branch3x3dbl_1.bn', 'Mixed_7b.branch3x3dbl_2.bn',
'Mixed_7b.branch3x3dbl_3a.bn', 'Mixed_7b.branch3x3dbl_3b.bn', 'Mixed_7b.branch_pool.bn','Mixed_7c.branch1x1.bn', 'Mixed_7c.branch3x3_1.bn',
'Mixed_7c.branch3x3_2a.bn','Mixed_7c.branch3x3_2b.bn', 'Mixed_7c.branch3x3dbl_1.bn', 'Mixed_7c.branch3x3dbl_2.bn', 'Mixed_7c.branch3x3dbl_3a.bn',
 'Mixed_7c.branch3x3dbl_3b.bn', 'Mixed_7c.branch_pool.bn']

# Network name : main layers
networks= {"inception_v3": inception_layers, "mobilenet": mobilenet_layers, "SphereFace": SphereFace_layers, "resnet50": resnet_layers, "cornet_s": cornet_s_layers, "FaceNet": facenet_layers, "vgg16_bn": vgg_layers, }