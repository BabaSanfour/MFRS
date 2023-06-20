"""
    `A script to save stimuli pictures (used in recording the MEG data) in H5 files `_.
    `We defined 9 files that we will use:
        Fam.h5:             (150, 224, 224, 1), 150 Famous Face Pictures.
        Scram.h5:           (150, 224, 224, 1), 150 Scrambled Face Pictures.
        FamUnfam.h5:        (300, 224, 224, 1), 150 Famous then Unfamiliar Face Pictures.
        FamScram.h5:        (300, 224, 224, 1), 150 Famous then Scrambled Face Pictures.
    `_.

    For each file:
    - Create a HDF5 file with pictures and their IDs.
    ---------------
    Output Files:
    ---------------
    9 HDF5 files as described above
    Parameters:
    images       images array, (N, 224, 224, 3) to be stored

"""

import os
import sys
import h5py
import datetime

import numpy as np
from tqdm import tqdm
from PIL import Image

import torchvision
sys.path.append("../../MFRS")
from utils.config import study_path
data_path =  os.path.join(study_path, "ds000117/stimuli/meg/")

def store_many_hdf5(images, name):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 224, 224, 1) to be stored
    """
    hdf5_dir = "/home/hamza97/scratch/data/MFRS_data/hdf5/"
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    # Create a new HDF5 file
    file = h5py.File("%s%s.h5"%(hdf5_dir,name), "w")
    print("{} h5 file created".format(name))

    # Create a dataset in the file
    dataset = file.create_dataset("images", np.shape(images), data=images) #h5py.h5t.STD_U8BE,
    file.close()
    print("{} h5 is ready".format(name))

def make_array(item):
    name, list_stimuli = item[0], item[1]
    # Concatenate array of images
    img_array = []

    # resize images to 224 224
    resize = torchvision.transforms.Resize((224, 224))
    # extract all the folders names: ids
    samples_pathes = sorted(
        [
            os.path.join(data_path, sname)
            for sname in list_stimuli
        ]
    )
    # run through the pictures list
    loop_generator = tqdm(samples_pathes)
    for  pic_name in loop_generator:
        # run through the pictures list
        img_sample = Image.open(pic_name) # read picture
        if img_sample is None:
            print('Picture not found:', pic_name)
            continue
        # transform the image
        sample = np.asarray(resize(img_sample), dtype=np.uint8)
        # append image to image and labels list
        img_array.append(sample)

    # print image array shape
    print(np.asarray(img_array).shape)
    # return image and label arrays
    return np.asarray(img_array)

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    # Generate lists with the names of pictures for each set_label
    Fam, Unfam, Scram = [], [], []
    for i in range (1,151):
        Fam.append('f%03d.bmp'%i)
        Unfam.append('u%03d.bmp'%i)
        Scram.append('s%03d.bmp'%i)
    FamUnfam = Fam + Unfam
    FamScram = Fam + Scram
    all = {'Fam': Fam,
            'Scram': Scram,
            'FamUnfam': FamUnfam,
            'FamScram': FamScram,
            }
    for item in all.items() :
        img_array = make_array(item)
        store_many_hdf5(img_array, item[0])
    print(datetime.datetime.now()-begin_time)
