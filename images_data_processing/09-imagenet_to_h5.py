import os
import re
import cv2
import json
import numpy as np
from tqdm import tqdm
import torchvision.transforms
from PIL import Image
import pandas as pd
import logging
import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import proj_path, study_path
from utils.utils import store_many_hdf5

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load label matching data
with open(os.path.join(proj_path, "images_data_processing", "files", "match_labels.json"), 'r') as f:
    match_labels = json.load(f)

# Load CSV data
csv_data = pd.read_csv(os.path.join(proj_path, "images_data_processing", "files", "new_LOC_val_solution.csv"))

# Define the path for the ImageNet training and validation data
folder_paths = {
    "train": 'imagenet_train_subset',
    "valid": os.path.join('ILSVRC', 'Data', 'CLS-LOC', 'val'),
}

def make_array(analysis_type: str, data_dir: str, folder: str):
    """
    Create arrays of images and labels based on the specified analysis type and folder.

    Args:
        analysis_type (str): The type of analysis.
        data_dir (str): The path to the data directory.
        folder (str): The folder type ('train' or 'valid').

    Returns:
        tuple: A tuple containing image and label arrays.
    """
    # Concatenate array of images
    img_array = []
    label_array = []

    # Resize images to 224x224
    resize = torchvision.transforms.Resize((224, 224))

    # Mapping of analysis types to labels to drop
    drop_labels_map = {
        "meg_stimuli_faces": ["random_faces"],
        "random_faces": ["meg_stimuli_faces"],
    }

    # Get the labels to drop based on the analysis type (default to ["random_faces", "meg_stimuli_faces"])
    drop_labels = drop_labels_map.get(analysis_type, ["random_faces", "meg_stimuli_faces"])

    if folder == 'train':
        label_folders = sorted( [ label for label in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label)) ])
        for label_folder in label_folders:
            print(label_folder)
            if label_folder in drop_labels:
                logger.info(f"Skipping label: {label_folder}")
                continue
            label = match_labels.get(label_folder, 1000)
            pictures_paths = sorted(
                [
                    os.path.join(data_dir, label_folder, sname)
                    for sname in os.listdir(os.path.join(data_dir, label_folder))
                ]
            )

            for picture_path in tqdm(pictures_paths):
                img_sample = cv2.imread(picture_path)  # Read picture
                if img_sample is None:
                    continue
                img_sample = img_sample[:, :, ::-1]  # Transform from BGR to RGB
                PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')
                sample = np.asarray(resize(PIL_image), dtype=np.uint8)
                img_array.append(sample)
                label_array.append(label)
    
    elif folder == 'valid':
        pictures_paths = sorted(
            [
                os.path.join(data_dir, sname)
                for sname in os.listdir(data_dir)
            ]
        )

        for picture_path in pictures_paths:
            img_sample = cv2.imread(picture_path)  # Read picture
            if img_sample is None:
                continue
            # Extract labels from a CSV file (assuming a specific format)
            name = os.path.splitext(os.path.basename(picture_path))[0]
            if name[0]!="I":
                name = name + ".jpg"
            label = csv_data.loc[csv_data['ImageId'] == name, 'PredictionString'].iloc[0].split()[0]
            if label in drop_labels:
                logger.info(f"Skipping label: {label}")
                continue
            label = match_labels.get(label, 1000)  # Use match_labels if needed
            img_sample = img_sample[:, :, ::-1]  # Transform from BGR to RGB
            PIL_image = Image.fromarray(img_sample.astype('uint8'), 'RGB')
            sample = np.asarray(resize(PIL_image), dtype=np.uint8)
            img_array.append(sample)
            label_array.append(label)

    logger.info(f"Image array shape: {np.asarray(img_array).shape}")
    logger.info(f"Label array shape: {np.asarray(label_array).shape}")
    return np.asarray(img_array), np.asarray(label_array)

if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    for folder in ["train", "valid"]:
        for analysis_type in ["random_faces", "meg_stimuli_faces", None]:
            logger.info(f"Starting {folder} set of {analysis_type} analysis.")
            img_array, label_array = make_array(analysis_type, folder_paths[folder], folder)
            hdf5_filename = f"imagenet_subset_{folder}_{analysis_type}"
            store_many_hdf5(img_array, label_array, hdf5_filename)
    elapsed_time = datetime.datetime.now() - begin_time
    logger.info(f"Total execution time: {elapsed_time}")
