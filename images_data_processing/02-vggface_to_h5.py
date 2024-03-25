import os
import cv2
import datetime
import numpy as np
from tqdm import tqdm
import torchvision
from PIL import Image
import logging

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path
from utils.utils import store_many_hdf5

data_path = os.path.join(study_path, "VGGface2/")

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def make_array(data_folder: str) -> tuple:
    """
    Creates arrays of images and labels from a data folder.

    Args:
    ----------
    data_folder : str
        Name of the data folder (e.g., 'train', 'test', 'valid').

    Returns:
    -------
    tuple
        Tuple containing the image array and label array.
    """
    dir = os.path.join(data_path, data_folder)
    img_array = []
    label_array = []

    # Resize images to 224x224
    resize = torchvision.transforms.Resize((224, 224))

    # Extract all the folder names (IDs)
    samples_pathes = sorted(
        [
            os.path.join(data_path, sname)
            for sname in os.listdir(data_path)
        ]
    )

    loop_generator = tqdm(samples_pathes, desc=f"Processing {data_folder}")
    for id_folder in loop_generator:
        pictures_pathes = sorted(
            [
                os.path.join(id_folder, data_folder, sname)
                for sname in os.listdir(os.path.join(id_folder, data_folder))
            ]
        )

        pic_id = os.path.basename(id_folder)
        pictures_loop_generator = tqdm(pictures_pathes, desc="Processing pictures")
        for picture in pictures_loop_generator:
            img_sample = cv2.imread(picture)
            if img_sample is None:
                continue
            img_sample = img_sample[:, :, ::-1]
            PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')
            sample = np.asarray(resize(PIL_image), dtype=np.uint8)
            img_array.append(sample)
            label_array.append(int(pic_id))

    img_array = np.asarray(img_array)
    label_array = np.asarray(label_array)
    logger.info(f"{data_folder} - Images shape: {img_array.shape}, Labels shape: {label_array.shape}")
    return img_array, label_array

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    for folder in ["valid", "test", "train"]:
        img_array, label_array = make_array(folder)
        store_many_hdf5(img_array, label_array, folder)
    logger.info(f"Total time: {datetime.datetime.now() - begin_time}")