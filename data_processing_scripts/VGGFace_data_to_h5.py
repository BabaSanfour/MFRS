"""
    For each set file:
    - Create a HDF5 file with pictures and their IDs.
    ---------------
    Output Files:
    ---------------
    3 HDF5 files for each data repartition set: in data/MFRS_data/HDF5_files:Train, valid and test sets
    Parameters:
    images       images array, (N, 224, 224, 3) to be stored
    labels       labels array, (N, ) to be stored

"""

import os
import sys
import cv2
import h5py

import datetime
import numpy as np
from tqdm import tqdm
import torchvision
from PIL import Image
sys.path.append("../../MFRS")
from utils.config import data_path
data_path =  os.path.join(data_path, "VGGface2/")

def store_many_hdf5(images, labels, folder):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 224, 224, 1) to be stored
        labels       labels array, (N, ) to be stored
    """
    hdf5_dir = "/home/hamza97/scratch/data/MFRS_data/hdf5/"
    if not os.path.exists(hdf5_dir):
        os.makedirs(hdf5_dir)
    # Create a new HDF5 file
    file = h5py.File("%s%s.h5"%(hdf5_dir,folder), "w")
    print("{} h5 file created".format(folder))

    # Create a dataset in the file
    dataset = file.create_dataset("images", np.shape(images), data=images) #h5py.h5t.STD_U8BE,
    metaset = file.create_dataset("meta", np.shape(labels), data=labels)
    file.close()
    print("{} h5 is ready".format(folder))

def make_array(data_folder):
    dir = data_path+data_folder+"/"
    # Concatenate array of images
    img_array = []
    label_array = []

    # resize images to 224 224
    resize = torchvision.transforms.Resize((224, 224))
    # extract all the folders names: ids
    samples_pathes = sorted(
        [
            os.path.join(dir, sname)
            for sname in os.listdir(dir)
        ]
    )
    # run through the ids list / folders list
    loop_generator = tqdm(samples_pathes)
    for  id in loop_generator:
        # extract all the pictures of a single id / in a folder
        pictures_pathes = sorted(
            [
                os.path.join(id, sname)
                for sname in os.listdir(id)
            ]
        )

        # get pictures id ( all pictures in the same folder has the same id)
        pic_id=os.path.basename(id)
        # run through the pictures list
        pictures_loop_generator = tqdm(pictures_pathes)
        for  picture in pictures_loop_generator:
            img_sample = cv2.imread(picture) # read picture
            if img_sample is None:
                continue
            img_sample = img_sample[:,:,::-1] # transform from BGR 2 RGB
            # transform the image
            PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')
            sample = np.asarray(resize(PIL_image), dtype=np.uint8)
            # append image and label to image and labels list
            img_array.append(sample)
            label_array.append(int(pic_id))

    # print label and image array shapes
    print(np.asarray(label_array).shape)
    print(np.asarray(img_array).shape)
    # return image and label arrays
    return np.asarray(img_array), np.asarray(label_array)

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    for folder in ["valid","test","train"] :
        img_array, label_array = make_array(folder)
        store_many_hdf5(img_array,label_array, folder)
    print(datetime.datetime.now()-begin_time)
