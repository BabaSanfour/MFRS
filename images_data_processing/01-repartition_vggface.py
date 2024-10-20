import os
import glob
import shutil
import numpy as np
import logging

# Importing from the utils folder
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path
from utils.utils import split

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
original_folders = os.path.join(study_path, "VGGface2_HQ_cropped/VGGface2_HQ_cropped/*")
new_folder_base = os.path.join(study_path, "VGGface2/")

def assign_class(num_files: int) -> dict:
    """
    Assign a class to each picture based on repartition counts.

    Args:
    ----------
    num_files : int
        Number of files to assign classes to.

    Returns:
    -------
    class_dic : dict
        Dictionary containing index as key and assigned class as value.
    """
    train_count = int(np.around(num_files * 0.7))
    test_count = int(np.around(num_files * 0.15))
    valid_count = num_files - test_count - train_count
    counts = [train_count, valid_count, test_count]

    class_dic = {}
    for index in range(num_files):
        picture_class = split(counts[0], counts[1], counts[2])
        counts[picture_class] -= 1
        class_dic[index] = picture_class
    return class_dic

def create_repartitions(src_folder: str, dest_folder: str, class_dic: dict) -> None:
    """
    Create data repartitions folders and copy pictures.

    Args:
    ----------
    src_folder : str
        Source folder containing pictures to be repartitioned.
    dest_folder : str
        Destination folder where repartitioned data will be stored.
    class_dic : dict
        Dictionary containing index as key and assigned class as value.

    Returns:
    -------
    None
    """
    train_folder = os.path.join(dest_folder, 'train')
    test_folder = os.path.join(dest_folder, 'test')
    valid_folder = os.path.join(dest_folder, 'valid')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    files_list = os.listdir(src_folder)
    for index, cl in class_dic.items():
        src_picture = os.path.join(src_folder, files_list[index])
        if cl == 0:
            dest_picture = os.path.join(train_folder, files_list[index])
        elif cl == 1:
            dest_picture = os.path.join(valid_folder, files_list[index])
        else:
            dest_picture = os.path.join(test_folder, files_list[index])
        
        shutil.move(src_picture, dest_picture)

if __name__ == '__main__':
    i = 0
    for folder in glob.glob(original_folders):
        logging.info(f"Processing folder {folder}")
        picture_list = os.listdir(folder)
        num_files = len(picture_list)

        class_dic = assign_class(num_files)
        new_folder_name = '0' * (4 - len(str(i))) + str(i)
        new_folder_path = os.path.join(new_folder_base, new_folder_name)

        create_repartitions(folder, new_folder_path, class_dic)
        i += 1
        logging.info(f"Folder {folder} processed successfully!")

