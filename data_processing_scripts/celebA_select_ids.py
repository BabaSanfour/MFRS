"""
    Clean original CSV file
    Create multiple CSV file with different number of ids.
    ---------------
    Input Files:
    ---------------
    identity_CelebA.txt        Text contraining picture names and original Ids
    list_attr_celeba.csv       CSV file with 40 attribute for each pictures
    ---------------
    Output Files:
    ---------------
    8 CSV files with different number of ids and pictures per id.
"""
import os
import sys
import pickle
import shutil 
import pandas as pd
sys.path.append("../../MFRS")
from utils.config import study_path
from utils.utils import split

def assign_class(file_csv) -> dict:

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

    for id in (list(file_csv["new_id"].unique())):
        # get the list of indexes for each id (label)
        id_index=list(file_csv[file_csv['new_id']==id].index)
        # get the list with number of occurence for a picture in each class
        train_count = int(np.around(len(id_index) * 0.7))
        test_count = int(np.around(len(id_index) * 0.15))
        valid_count = len(id_index) - test_count - train_count
        counts = [train_count, valid_count, test_count]
        # assign the class randomly and add it to csv file
        for ind in (id_index):
            id_class=split(counts[0],counts[1],counts[2])
            counts[id_class]=counts[id_class]-1
            file_csv.loc[ind,'class']=int(id_class)
    return file_csv

def create_txt(file_csv, files):

    # read train i file
    train= open(files[0],"w+")
    # read test i file
    test= open(files[1],"w+")
    # read valid i file
    valid= open(files[2],"w+")

    # add each picture to the class it belongs to
    for i in range(len(file_csv)):
        name=file_csv.loc[i,'name']+' '+str(int(file_csv.loc[i,'new_id']))+'\n'
        if int(file_csv.loc[i,'class']) == 0:
            train.write(name)
        elif int(file_csv.loc[i,'class']) == 1:
            valid.write(name)
        else :
            test.write(name)

def correct_gender(merged):
    """
        Some of the ids have mixed designation for the Male attribue
        In this function we correct this problem.
    """
    # test is a Series with the count of Male attribute designation for every id
    test= merged.groupby(['id','Male'])['Male'].count()
    # list of unique ids
    id_list= merged['id'].unique()
    for i in id_list:
        if len(test[i])==1:
            #if the id has 1 type of designation (-1 or 1) then we go the next one
            continue
        else:
            # if it has 2: -1 and 1
            ids=list(merged[merged['id']==i]['Male'].index)
            # we take the designation that has the maximum of occurence
            # and attribue it to the rest of pictures of this id
            if test[i][1]>test[i][-1]:
                merged.loc[ids,'Male']=1
            else:
                merged.loc[ids,'Male']=-1
    return merged

def drop_unwanted_columns(selected, df):
    # Generate list of columns in attribue file:
    col=list(df.columns)
    # list of unwanted columns
    removed = list(set(col) - set(selected))
    # drop unwanted columns
    return df.drop(removed, axis=1)

def clean_csv(identity_file, list_attr_celeba):
    """
        Takes the original dataset (txt and csv file)
        Select the columns we need
        Merge the two Files
        Remove pictures with Eyeglasses, Hats, and Earrings
        Count occurence of each id
        Correct some mistakes in the Male attribue
        return a male and female dataframes
    """
    # rename the columns
    identity_file.columns=['name', 'id']

    # Clean attribute file
    list_attr_celeba=list_attr_celeba.drop('Unnamed: 0', axis=1)
    # define the list of attribues (columns) we need
    selected = ['Eyeglasses','Male' ,'Wearing_Hat', 'name', 'Wearing_Earrings' ]

    # drop columns
    list_attr_celeba=drop_unwanted_columns(selected, list_attr_celeba)

    # Merge the two dataframes on the name column
    merged = pd.merge(list_attr_celeba, identity_file, on="name")
    merged=merged.reset_index(drop=True)

    # Drop pictures with eyeglasses, Hats, and Earrings.
    merged=merged[merged['Eyeglasses']==-1]
    merged=merged[merged['Wearing_Hat']==-1]
    merged=merged[merged['Wearing_Earrings']==-1]

    # Count the occurence of each id in the dataset
    id_occurence=merged['id'].value_counts()
    id_occurence=id_occurence.to_frame().reset_index()
    id_occurence.columns=['id', 'values']

    # Merge the id_occurence dataframe and merged dataframe
    merged = pd.merge(merged, id_occurence, on="id")

    # Corret some mistakes in gender attribue
    merged=correct_gender(merged)

    df_male=merged[merged['Male']==1]
    df_female=merged[merged['Male']==-1]

    # return two dataframes: male and female
    return merged[merged['Male']==1], merged[merged['Male']==-1], merged

def drop_unwanted_rows(key, selected, df):
    # for earch id we select the desired number of pictures(key) and drop the rest
    df['drop']=1
    for i in selected:
        ids=list(df[df['id']==i].index)
        df.loc[ids[:key],'drop']=-1

    return df[df['drop']==-1]


def create_csv(key, value, df_male, df_female, merged):
    # select VALUE male ids with occurence > key

    labels_male=df_male[df_male['values']>=key]['id'].unique()[:value]

    # select VALUE female ids with occurence > key
    labels_female=df_female[df_female['values']>=key]['id'].unique()[:value]

    # regroup male and female id list
    selected= list(labels_female)+list(labels_male)

    # Select only desired ids pictures
    merged=merged.loc[merged['id'].isin(selected)]

    merged=merged.reset_index(drop=True)
    # drop rows for ids with occurence > key
    merged=drop_unwanted_rows(key, selected, merged)
    selected = ['name','id']
    return drop_unwanted_columns(selected, merged)

def generate_new_ids(final):
    final=final.reset_index(drop=True)
    # extract labels of dataframe
    labels=list(final.id.unique())
    # Assign a new label to each old label
    dic ={}
    for i in range (len(labels)):
        dic[labels[i]]=i
    # Change labels
    final['new_id']=-1
    # Assign new labels to ids
    for i in range (len(final)):
        final.loc[i,'new_id']=dic[final.loc[i,'id']]
    selected = ['name','new_id']
    return drop_unwanted_columns(selected, final)

def create_repartitions(file_csv, data_folders):
    # Create data repartitions folders and copy pictures

    train_folder=os.path.join(study_path , data_folders[0])
    test_folder=os.path.join(study_path , data_folders[1])
    valid_folder=os.path.join(study_path , data_folders[2])

    os.makedirs(train_folder)
    os.makedirs(test_folder)
    os.makedirs(valid_folder)

    for i in range(len(file_csv)):
        im=os.path.join(study_path, "img_align_celeba/img_align_celeba", file_csv.loc[i,'name'])
        if file_csv.loc[i,'class'] == 0:
            shutil.copy(im, train_folder)
        elif file_csv.loc[i,'class'] == 1:
            shutil.copy(im, valid_folder)
        else :
            shutil.copy(im, test_folder)

if __name__ == '__main__':
    new_data = pd.read_csv('identity_CelebA_with_meg_stimuli.txt', sep=" ", header=None)
    with open("mapping_dict.pickle", "rb") as pickle_file:
        loaded_data = pickle.load(pickle_file)
    stimuli_ids = [value[0] for key, value in loaded_data.items()]
    stimuli_ids.extend([9558, 5183])

    old_data = pd.read_csv('identity_CelebA.txt', sep=" ", header=None)
    old_data.columns =["name", "id"]
    list_attr_celeba = pd.read_csv('list_attr_celeba.csv')
    list_attr_celeba = pd.merge(old_data, list_attr_celeba, on='name', how='outer')
    list_attr_celeba = list_attr_celeba[~list_attr_celeba["id"].isin(stimuli_ids)].reset_index(drop=True)

    # Count the occurence of each id in the dataset
    id_occurence=list_attr_celeba['id'].value_counts()
    id_occurence=id_occurence.to_frame().reset_index()
    id_occurence.columns=['id', 'values']

    # Merge the id_occurence dataframe and merged dataframe
    merged = pd.merge(list_attr_celeba, id_occurence, on="id")

    # Corret some mistakes in gender attribue
    merged=correct_gender(merged)
    df_male=merged[merged['Male']==1]
    df_female=merged[merged['Male']==-1]

    final=create_csv(30, 437, df_male, df_female, merged)
    final = final[final["id"] != 59]

    not_included = new_data[new_data[1].isin(stimuli_ids)].reset_index(drop=True)
    not_included.columns = ["name", "id"]
    final = pd.concat([final, not_included])
    final=generate_new_ids(final)
    final.to_csv("celebA_with_stimuli.csv")

    file_csv=assign_class(final)
    final.to_csv("celebA_with_stimuli_class_dist.csv")

    files = ["identity_CelebA_train_with_meg_stimuli.txt",  "identity_CelebA_test_with_meg_stimuli.txt", "identity_CelebA_valid_with_meg_stimuli.txt"]
    create_txt(file_csv, files)
    data_folders = ['train_with_meg_stimuli', 'test_with_meg_stimuli', 'valid_with_meg_stimuli/']
    create_repartitions(file_csv, data_folders)


    # Now without the MEG_stimuli
    final_without=create_csv(30, 500, df_male, df_female, merged)
    final_without=generate_new_ids(final_without)
    final_without.to_csv("celebA_without_stimuli.csv")
    final_without_csv=assign_class(final_without)
    final_without_csv.to_csv("celebA_without_stimuli_class_dist.csv")
    files = ["identity_CelebA_train_without_meg_stimuli.txt",  "identity_CelebA_test_without_meg_stimuli.txt", "identity_CelebA_valid_without_meg_stimuli.txt"]
    create_txt(final_without_csv, files)
    data_folders = ['train_without_meg_stimuli', 'test_without_meg_stimuli', 'valid_without_meg_stimuli/']
    create_repartitions(final_without_csv, data_folders)