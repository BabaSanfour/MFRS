import os
import pickle
import shutil
import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import study_path
from utils.utils import split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assign_class(file_csv: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a class to each picture based on repartition counts.

    Args:
        file_csv (pd.DataFrame): DataFrame containing picture information.

    Returns:
        pd.DataFrame: DataFrame with assigned class for each picture.
    """
    for id in file_csv["new_id"].unique():
        # Get the list of indexes for each id (label)
        id_index = list(file_csv[file_csv['new_id'] == id].index)
        # Calculate the number of occurrences for each class
        train_count = int(np.around(len(id_index) * 0.7))
        test_count = int(np.around(len(id_index) * 0.15))
        valid_count = len(id_index) - test_count - train_count
        counts = [train_count, valid_count, test_count]
        
        # Assign the class randomly and add it to the CSV file
        for ind in id_index:
            id_class = split(counts[0], counts[1], counts[2])
            counts[id_class] = counts[id_class] - 1
            file_csv.loc[ind, 'class'] = int(id_class)
    return file_csv

def create_txt(file_csv: pd.DataFrame, files: list) -> None:
    """
    Create train, test, and valid text files and write picture names with IDs to each file.

    Args:
        file_csv (pd.DataFrame): DataFrame containing picture information.
        files (list): List of file paths for train, test, and valid text files.
    """
    try:
        # Open train, test, and valid text files
        with open(files[0], "w+") as train, open(files[1], "w+") as test, open(files[2], "w+") as valid:
            for i in range(len(file_csv)):
                name = f"{file_csv.loc[i,'name']} {int(file_csv.loc[i,'new_id'])}\n"
                if int(file_csv.loc[i,'class']) == 0:
                    train.write(name)
                elif int(file_csv.loc[i,'class']) == 1:
                    valid.write(name)
                else:
                    test.write(name)
        logger.info("Train, test, and valid text files created successfully.")
    except Exception as e:
        logger.error(f"An error occurred while creating text files: {e}")


def correct_gender(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects mixed gender designation for the attribute in the dataset.

    Args:
        merged (pd.DataFrame): Merged DataFrame containing picture and attribute information.

    Returns:
        pd.DataFrame: DataFrame with corrected gender attributions.
    """
    try:
        # Calculate the count of Male attribute designation for each id
        test = merged.groupby(['id', 'Male'])['Male'].count()
        
        # List of unique ids
        id_list = merged['id'].unique()
        
        # Iterate through each id
        for i in id_list:
            if len(test[i]) == 1:
                # If the id has only one type of designation (-1 or 1), skip to the next one
                continue
            else:
                # If it has both designations: -1 and 1
                ids = list(merged[merged['id'] == i]['Male'].index)
                
                # Determine the designation with the maximum occurrence
                if test[i][1] > test[i][-1]:
                    merged.loc[ids, 'Male'] = 1
                else:
                    merged.loc[ids, 'Male'] = -1
        logger.info("Gender attributions corrected successfully.")
        return merged
    except Exception as e:
        logger.error(f"An error occurred while correcting gender attributions: {e}")


def drop_unwanted_columns(selected: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unwanted columns from a DataFrame.

    Args:
        selected (list): List of selected columns to keep.
        df (pd.DataFrame): DataFrame containing columns to be filtered.

    Returns:
        pd.DataFrame: DataFrame with unwanted columns dropped.
    """
    try:
        # Generate a list of columns in the attribute file
        col = list(df.columns)
        
        # List of unwanted columns
        removed = list(set(col) - set(selected))
        
        # Drop unwanted columns
        filtered_df = df.drop(removed, axis=1)
        logger.info("Unwanted columns dropped successfully.")
        return filtered_df
    except Exception as e:
        logger.error(f"An error occurred while dropping unwanted columns: {e}")


def clean_csv(identity_file: pd.DataFrame, list_attr_celeba: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans the original dataset, selects relevant columns, merges files, and corrects gender attributions.

    Args:
        identity_file (pd.DataFrame): DataFrame containing picture names and IDs.
        list_attr_celeba (pd.DataFrame): DataFrame containing attribute information for pictures.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Three DataFrames representing male, female, and merged data.
    """
    try:
        # Rename the columns in identity_file
        identity_file.columns = ['name', 'id']

        # Clean the attribute file
        list_attr_celeba = list_attr_celeba.drop('Unnamed: 0', axis=1)

        # Define the list of attributes (columns) to keep
        selected = ['Eyeglasses', 'Male', 'Wearing_Hat', 'name', 'Wearing_Earrings']

        # Drop unwanted columns from the attribute file
        list_attr_celeba = drop_unwanted_columns(selected, list_attr_celeba)

        # Merge the two dataframes on the 'name' column
        merged = pd.merge(list_attr_celeba, identity_file, on="name")
        merged = merged.reset_index(drop=True)

        # Drop pictures with Eyeglasses, Hats, and Earrings
        merged = merged[merged['Eyeglasses'] == -1]
        merged = merged[merged['Wearing_Hat'] == -1]
        merged = merged[merged['Wearing_Earrings'] == -1]

        # Count the occurrence of each id in the dataset
        id_occurence = merged['id'].value_counts()
        id_occurence = id_occurence.to_frame().reset_index()
        id_occurence.columns = ['id', 'values']

        # Merge the id_occurrence dataframe and merged dataframe
        merged = pd.merge(merged, id_occurence, on="id")

        # Correct gender attributions
        merged = correct_gender(merged)

        # Separate male and female dataframes
        df_male = merged[merged['Male'] == 1]
        df_female = merged[merged['Male'] == -1]

        logger.info("Dataset cleaned and attributes corrected successfully.")
        return df_male, df_female, merged
    except Exception as e:
        logger.error(f"An error occurred while cleaning the dataset: {e}")


def drop_unwanted_rows(key: int, selected: List[int], df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops unwanted rows from the DataFrame based on the selected number of pictures for each ID.

    Args:
        key (int): Number of pictures to keep for each ID.
        selected (List[int]): List of IDs to process.
        df (pd.DataFrame): DataFrame to drop rows from.

    Returns:
        pd.DataFrame: DataFrame with unwanted rows dropped.
    """
    try:
        # Create a column 'drop' and initialize it with 1
        df['drop'] = 1

        # For each selected ID, select the desired number of pictures (key) and mark the rest to be dropped
        for i in selected:
            ids = list(df[df['id'] == i].index)
            df.loc[ids[:key], 'drop'] = -1

        # Return the DataFrame with only rows that were marked to be dropped (-1)
        return df[df['drop'] == -1]
    except Exception as e:
        logger.error(f"An error occurred while dropping unwanted rows: {e}")



def create_csv(key: int, value: int, df_male: pd.DataFrame, df_female: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new DataFrame with selected male and female IDs based on occurrence and key.

    Args:
        key (int): Minimum occurrence count for each ID.
        value (int): Number of IDs to select.
        df_male (pd.DataFrame): DataFrame containing male data.
        df_female (pd.DataFrame): DataFrame containing female data.
        merged (pd.DataFrame): Merged DataFrame containing attributes and IDs.

    Returns:
        pd.DataFrame: New DataFrame with selected male and female IDs.
    """
    try:
        # Select male IDs with occurrence greater than or equal to key
        labels_male = df_male[df_male['values'] >= key]['id'].unique()[:value]

        # Select female IDs with occurrence greater than or equal to key
        labels_female = df_female[df_female['values'] >= key]['id'].unique()[:value]

        # Combine selected male and female IDs
        selected = list(labels_female) + list(labels_male)

        # Select only desired IDs' pictures
        selected_merged = merged.loc[merged['id'].isin(selected)].reset_index(drop=True)

        # Drop rows for IDs with occurrence > key
        selected_merged = drop_unwanted_rows(key, selected, selected_merged)

        selected_columns = ['name', 'id']
        return drop_unwanted_columns(selected_columns, selected_merged)
    except Exception as e:
        logger.error(f"An error occurred while creating the new DataFrame: {e}")


def generate_new_ids(final: pd.DataFrame) -> pd.DataFrame:
    """
    Generates new IDs for the given DataFrame.

    Args:
        final (pd.DataFrame): DataFrame containing original IDs.

    Returns:
        pd.DataFrame: DataFrame with new IDs assigned.
    """
    try:
        final = final.reset_index(drop=True)

        # Extract unique labels (IDs) from the DataFrame
        labels = list(final['id'].unique())

        # Assign new IDs to each old label
        id_mapping = {}
        for i, label in enumerate(labels):
            id_mapping[label] = i

        # Create a new column 'new_id' and assign new labels
        final['new_id'] = final['id'].map(id_mapping)

        selected_columns = ['name', 'new_id']
        return drop_unwanted_columns(selected_columns, final)
    except Exception as e:
        logger.error(f"An error occurred while generating new IDs: {e}")


def create_repartitions(file_csv: pd.DataFrame, data_folders: Tuple[str, str, str]) -> None:
    """
    Create data repartition folders and copy pictures to respective folders.

    Args:
        file_csv (pd.DataFrame): DataFrame containing picture information and classes.
        data_folders (Tuple[str, str, str]): Tuple containing names of train, test, and valid folders.

    Returns:
        None
    """
    try:
        train_folder = os.path.join(study_path, data_folders[0])
        test_folder = os.path.join(study_path, data_folders[1])
        valid_folder = os.path.join(study_path, data_folders[2])

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        os.makedirs(valid_folder, exist_ok=True)

        for i in range(len(file_csv)):
            im = os.path.join(study_path, "img_align_celeba/img_align_celeba", file_csv.loc[i, 'name'])
            if file_csv.loc[i, 'class'] == 0:
                shutil.copy(im, train_folder)
            elif file_csv.loc[i, 'class'] == 1:
                shutil.copy(im, valid_folder)
            else:
                shutil.copy(im, test_folder)
    except Exception as e:
        logger.error(f"An error occurred while creating data repartitions: {e}")


if __name__ == '__main__':
    # Load 'identity_CelebA_with_meg_stimuli.txt' file
    new_data = pd.read_csv('/files/identity_CelebA_with_meg_stimuli.txt', sep=" ", header=None)

    # Load 'mapping_dict.pickle' file
    with open("/files/mapping_dict.pickle", "rb") as pickle_file:
        loaded_data = pickle.load(pickle_file)
    stimuli_ids = [value[0] for key, value in loaded_data.items()]
    stimuli_ids.extend([9558, 5183])

    # Load 'identity_CelebA.txt' file
    old_data = pd.read_csv('/files/identity_CelebA.txt', sep=" ", header=None)
    old_data.columns = ["name", "id"]

    # Load 'list_attr_celeba.csv' file
    list_attr_celeba = pd.read_csv('/files/list_attr_celeba.csv')
    # Merge 'old_data' and 'list_attr_celeba' files based on 'name' column
    list_attr_celeba = pd.merge(old_data, list_attr_celeba, on='name', how='outer')
    # Remove rows with IDs present in 'stimuli_ids'
    list_attr_celeba = list_attr_celeba[~list_attr_celeba["id"].isin(stimuli_ids)].reset_index(drop=True)

    # Count the occurrence of each ID in the dataset
    id_occurence = list_attr_celeba['id'].value_counts()
    id_occurence = id_occurence.to_frame().reset_index()
    id_occurence.columns = ['id', 'values']

    # Merge 'id_occurence' and 'list_attr_celeba' dataframes
    merged = pd.merge(list_attr_celeba, id_occurence, on="id")

    # Correct some mistakes in gender attribute
    merged = correct_gender(merged)
    df_male = merged[merged['Male'] == 1]
    df_female = merged[merged['Male'] == -1]

    # Create CSV with specific parameters
    final = create_csv(30, 437, df_male, df_female, merged)
    final = final[final["id"] != 59]

    # Concatenate new_data and final dataframes
    not_included = new_data[new_data[1].isin(stimuli_ids)].reset_index(drop=True)
    not_included.columns = ["name", "id"]
    final = pd.concat([final, not_included])

    # Generate new IDs for the final dataframe
    final = generate_new_ids(final)

    # Save final dataframe to 'celebA_with_stimuli.csv'
    final.to_csv("/files/celebA_with_stimuli.csv")

    # Assign classes to the final dataframe
    file_csv = assign_class(final)
    # Save class distribution to 'celebA_with_stimuli_class_dist.csv'
    file_csv.to_csv("/files/celebA_with_stimuli_class_dist.csv")

    # Define file paths for txt files
    files = ["identity_CelebA_train_with_meg_stimuli.txt", "identity_CelebA_test_with_meg_stimuli.txt", "identity_CelebA_valid_with_meg_stimuli.txt"]

    # Create txt files
    create_txt(file_csv, files)

    # Define data folders
    data_folders = ['train_with_meg_stimuli', 'test_with_meg_stimuli', 'valid_with_meg_stimuli/']

    # Create data repartitions using defined data folders
    create_repartitions(file_csv, data_folders)

    # Create CSV without MEG_stimuli
    final_without = create_csv(30, 500, df_male, df_female, merged)
    final_without = generate_new_ids(final_without)

    # Save final_without dataframe to 'celebA_without_stimuli.csv'
    final_without.to_csv("/files/celebA_without_stimuli.csv")

    # Assign classes to the final_without dataframe
    final_without_csv = assign_class(final_without)
    # Save class distribution to 'celebA_without_stimuli_class_dist.csv'
    final_without_csv.to_csv("/files/celebA_without_stimuli_class_dist.csv")

    # Define file paths for txt files without MEG_stimuli
    files = ["identity_CelebA_train_without_meg_stimuli.txt", "identity_CelebA_test_without_meg_stimuli.txt", "identity_CelebA_valid_without_meg_stimuli.txt"]

    # Create txt files without MEG_stimuli
    create_txt(final_without_csv, files)

    # Define data folders without MEG_stimuli
    data_folders = ['train_without_meg_stimuli', 'test_without_meg_stimuli', 'valid_without_meg_stimuli/']

    # Create data repartitions without MEG_stimuli using defined data folders
    create_repartitions(final_without_csv, data_folders)
