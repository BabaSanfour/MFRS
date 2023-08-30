import os
import json
import time
import logging
import random
import pickle 

import pandas as pd

# Set up logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add necessary paths
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import proj_path, study_path

# Define the path for the imagenet training data
train_path = os.path.join('ILSVRC', 'Data', 'CLS-LOC', 'train')
valid_path = os.path.join('ILSVRC', 'Data', 'CLS-LOC', 'val')

def get_all_image_files(directory: dict) -> list:
    """
    Get a list of all image filenames in a directory.
    
    Args:
        directory (str): Path to the directory containing image files.
        
    Returns:
        list: List of filenames of image files.
    """
    filenames = []
    class_name = directory.split('/')[-1]
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
            part_name = filename.split('_')[0]
            assert part_name == class_name
    return filenames

def copy_selected_data(dirs: list, new_train_path: str) -> None:
    """
    Copy selected data from specified directories to a new destination path.
    
    Args:
        dirs (list): List of directory names containing data to be copied.
        new_train_path (str): Path to the new destination directory.
    """
    total_images = 0
    for directory in dirs:
        directory_path = os.path.join(train_path, directory)
        filenames = get_all_image_files(directory_path)
        new_number_images_per_class = 500
        if len(filenames) > new_number_images_per_class:
            filenames = random.sample(filenames, new_number_images_per_class)
        total_images += len(filenames)
        
        new_directory_path = os.path.join(new_train_path, directory)
        os.makedirs(new_directory_path, exist_ok=True)
        
        for filename in filenames:
            src_file_path = os.path.join(train_path, directory, filename)
            tgt_file_path = os.path.join(new_directory_path, filename)
            os.system(f'cp {src_file_path} {tgt_file_path}')
    
    logger.info(f'Total number of images selected: {total_images}')

def add_faces_class(train_face_pictures_list: list, new_train_path: str, directory: str) -> None:
    """
    Add face images from a list to a specific directory in the new training path.
    
    Args:
        train_face_pictures_list (list): List of filenames of face images to be added to train set.
        new_train_path (str): Path to the new destination directory.
        directory (str): Directory name to add the face images.
    """
    logger.info(f"Adding face images to {directory} directory.")
    
    new_directory_path = os.path.join(new_train_path, directory)
    os.makedirs(new_directory_path, exist_ok=True)
    
    for filename in train_face_pictures_list:
        src_file_path = os.path.join(study_path, 'img_align_celeba/img_align_celeba', filename)
        tgt_file_path = os.path.join(new_directory_path, filename)
        os.system(f'cp {src_file_path} {tgt_file_path}')

    logger.info(f'{directory} face images added.')

def add_faces_class_valid(valid_meg_faces: list, valid_random_faces: list) -> None:
    """
    Add face images from to the validation set and update the validation labels csv file.
    
    Args:
        valid_meg_faces (list): List of filenames of stimuli face images to be added to valid set.
        valid_random_faces (list): List of filenames of random face images to be added to valid set.
    """
    logger.info(f"Adding face images to validation set.")

    csv_data = pd.read_csv(os.path.join(study_path, "LOC_val_solution.csv"))

    for filename in valid_meg_faces:
        src_file_path = os.path.join(study_path, 'img_align_celeba/img_align_celeba', filename)
        tgt_file_path = os.path.join(valid_path, filename)
        os.system(f'cp {src_file_path} {tgt_file_path}')
        new_row = {'ImageId': filename, 'PredictionString': 'meg_stimuli_faces'}
        csv_data = csv_data.append(new_row, ignore_index=True)

    for filename in valid_random_faces:
        src_file_path = os.path.join(study_path, 'img_align_celeba/img_align_celeba', filename)
        tgt_file_path = os.path.join(valid_path, filename)
        os.system(f'cp {src_file_path} {tgt_file_path}')
        new_row = {'ImageId': filename, 'PredictionString': 'random_faces'}
        csv_data = csv_data.append(new_row, ignore_index=True)

    csv_data.to_csv(os.path.join(proj_path, "images_data_processing", "files", "new_LOC_val_solution.csv"), index=False)

    logger.info('validation face images added.')


if __name__ == '__main__':
    logger.info("Starting the data processing.")
    process_start_time = time.time()
    
    # Get the list of class directories in the training path
    class_directories = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    num_classes = len(class_directories)
    logger.info(f'Number of classes: {num_classes}')
    
    # Define the new training path
    new_train_path = 'imagenet_train_subset/'
    if not os.path.exists(new_train_path):
        os.makedirs(new_train_path)
    
    # Copy a subset of images from each class to the new training path
    copy_selected_data(class_directories, new_train_path)

    # Load CSV and pickle files
    csv_path = os.path.join(proj_path, "images_data_processing", "files", "CelebA_with_stimuli.csv")

    dir_txt = os.path.join(proj_path, "images_data_processing", "files", "identity_CelebA_with_meg_stimuli.txt")
    data = pd.read_csv(dir_txt, sep=" ", header=None)
    data.columns = ["name", "id"]

    with open(os.path.join(proj_path, "images_data_processing", "files", "mapping_dict.pickle"), "rb") as pickle_file:
        loaded_data = pickle.load(pickle_file)
    
    id_values = [value[0] for value in loaded_data.values()]

    # Filter the DataFrame to include only rows where the ID exists in id_values
    filtered_df = data[data["id"].isin(id_values)]

    # Randomly select 500 names from the filtered data
    selected_train_names = random.sample(filtered_df["name"].tolist(), 500)
    # Generate a list of names that are not in selected_train_names
    remaining_names = [name for name in filtered_df["name"].tolist() if name not in selected_train_names]
    # Select 100 names from the remaining_names list
    additional_valid_names_meg = random.sample(remaining_names, 100)

    # Add face images to the new train set
    add_faces_class(selected_train_names, new_train_path, "meg_stimuli_faces")

    # Filter the DataFrame for IDs not in id_values
    filtered_df = data[~data['id'].isin(id_values)]
    selected_train_names = random.sample(filtered_df['name'].tolist(), 500)
    remaining_names = [name for name in filtered_df["name"].tolist() if name not in selected_train_names]
    additional_valid_names_random = random.sample(remaining_names, 100)

    # Add random face images to the new train set
    add_faces_class(selected_train_names, new_train_path, "random_faces")

    add_faces_class_valid(additional_valid_names_meg, additional_valid_names_random)

    logger.info(f'Finished processing, total time taken: {time.time() - process_start_time:.2f} sec')
