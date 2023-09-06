import os
import h5py
import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

data_path = os.path.join(study_path, "ds000117/stimuli/meg")

def store_many_hdf5(images: np.array) -> None:
    """
    Stores an array of images to HDF5.

    Args:
    ---------------
    images : numpy.ndarray
        Images array with shape (N, 224, 224, 3) to be stored.
    name : str
        Name of the HDF5 file to be created.
    """
    hdf5_path = os.path.join(study_path, "hdf5", f"Stimuli.h5")
    with h5py.File(hdf5_path, "w") as file:
        file.create_dataset("images", data=images)
    logger.info(f"Stimuli HDF5 file created at {hdf5_path}")

def make_array(list_stimuli: list) -> np.array:
    """
    Create an array of images from a list of image filenames.

    Args:
    ---------------
    list_stimuli : list
        List containing the name of stimuli filenames.

    Returns:
    ---------------
    numpy.ndarray
        Array of resized images.
    """

    resize = torchvision.transforms.Resize((224, 224))

    samples_paths = sorted([
        os.path.join(data_path, sname) for sname in list_stimuli
    ])

    img_array = []

    loop_generator = tqdm(samples_paths, desc="Loading images")
    for pic_name in loop_generator:
        if pic_name.lower().endswith(".bmp"):
            img_sample = Image.open(pic_name)
        else:
            logger.warning(f"Picture not found: {pic_name}")
            continue
        if img_sample is None:
            logger.warning(f"Picture not found: {pic_name}")
            continue
        sample = np.asarray(resize(img_sample), dtype=np.uint8)
        img_array.append(sample)

    img_array = np.asarray(img_array)
    logger.info(f"Image array shape: {img_array.shape}")
    return img_array

if __name__ == '__main__':
    begin_time = datetime.datetime.now()

    Fam, Unfam, Scram = [], [], []
    for i in range(1, 151):
        Fam.append(f'f{i:03d}.bmp')
        Unfam.append(f'u{i:03d}.bmp')
        Scram.append(f's{i:03d}.bmp')

    stimuli = Fam + Unfam + Scram
    img_array = make_array(sorted(stimuli))
    store_many_hdf5(img_array)

    elapsed_time = datetime.datetime.now() - begin_time
    logger.info(f"Time taken: {elapsed_time}")
