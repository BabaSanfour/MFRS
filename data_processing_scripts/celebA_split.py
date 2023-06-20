"""
    For each CSV file:
    - We take all the pictures and split them into three classes: Train, Valid and Test.
    - Create three txt files: Each file has Train/Valid/Test pictures and their ids.
    - Create data repartitions folders and copy pictures
    ---------------
    Input Files:
    ---------------
    read files in files/csv_files:       8 CSV file with pictures name and id
    ---------------
    Output Files:
    ---------------
    24 txt files in files/txt_files: 3 txt files for each csv file: Train, valid and test classes
    (To divide data into 3 folders for each csv file)
"""
import os
import sys
import glob
import shutil
import random
import pandas as pd

sys.path.append("../../MFRS")
from utils.config import proj_path, study_path
dir_path = os.path.join(proj_path, "files/csv_files/*.csv")
train_dir = os.path.join(proj_path, "files/txt_files/identity_CelebA_train_%s_%s.txt")
test_dir = os.path.join(proj_path, "files/txt_files/identity_CelebA_test_%s_%s.txt")
valid_dir = os.path.join(proj_path, "files/txt_files/identity_CelebA_valid_%s_%s.txt")
fold = study_path

dic = {25:[15,5,5], 12:[6,3,3],
        27:[17,5,5], 13:[7,3,3],
        28:[18,5,5], 14:[8,3,3],
        29:[19,5,5], 15:[9,3,3]}


def read_file(file):
    # read file
    file_csv=pd.read_csv(file)
    # drop unwanted column
    file_csv=file_csv.drop('Unnamed: 0', axis=1)
    # get number of labels in csv file
    length=len(file_csv['new_id'].unique())
    # number of pictures for each id
    key=file_csv[file_csv['new_id']==1]['new_id'].value_counts()[1]
    return file_csv, length, key

def split(a,b,c):
    """
        Give a random class for each picture
    """
    if (a<0 or b<0 or c<0):
        print('We have a problem: code 1 ')
        return -1
    elif(a==0 and b==0 and c==0):
        print('We have a problem: code 2 ')
        return -1
    elif (a==0 and b==0):
        return 2
    elif(a==0 and c==0):
        return 1
    elif (b==0 and c==0):
        return 0
    elif (b==0):
        return 2*random.randrange(2)
    elif(c==0):
        return random.randrange(2)
    elif (a==0):
        return random.randrange(2)+1
    else:
        return random.randrange(3)


def assign_class(file_csv, length, key, dic=dic):
    for i in range (length):
        # get the list of indexes for each id (label)
        ids=list(file_csv[file_csv['new_id']==i].index)
        # get the list with number of occurence for a picture in each class
        counts= dic[key].copy()
        # assign the class randomly and add it to csv file
        for ind in (ids):
            id_class=split(counts[0],counts[1],counts[2])
            counts[id_class]=counts[id_class]-1
            file_csv.loc[ind,'class']=int(id_class)
    return file_csv

def create_txt(file_csv, length, key, train_dir=train_dir, test_dir=test_dir, valid_dir=valid_dir):

    # read train i file
    train_dir_i=train_dir%(length, key)
    train= open(train_dir_i,"w+")
    # read test i file
    test_dir_i=test_dir%(length, key)
    test= open(test_dir_i,"w+")
    # read valid i file
    valid_dir_i=valid_dir%(length, key)
    valid= open(valid_dir_i,"w+")

    # add each picture to the class it belongs to
    for i in range(len(file_csv)):
        name=file_csv.loc[i,'name']+' '+str(int(file_csv.loc[i,'new_id']))+'\n'
        if file_csv.loc[i,'class'] == 0:
            train.write(name)
        elif file_csv.loc[i,'class'] == 1:
            valid.write(name)
        else :
            test.write(name)

def create_repartitions(file_csv, length, key):
    # Create data repartitions folders and copy pictures

    train_folder=fold + 'train_%s_%s/'%(length, key)
    test_folder=fold + 'test_%s_%s/'%(length, key)
    valid_folder=fold + 'valid_%s_%s/'%(length, key)

    os.makedirs(train_folder)
    os.makedirs(test_folder)
    os.makedirs(valid_folder)

    for i in range(len(file_csv)):
        im=os.path.join(fold+"img_align_celeba/img_align_celeba", file_csv.loc[i,'name'])
        if file_csv.loc[i,'class'] == 0:
            shutil.copy(im, train_folder)
        elif file_csv.loc[i,'class'] == 1:
            shutil.copy(im, valid_folder)
        else :
            shutil.copy(im, test_folder)


if __name__ == '__main__':

    # run through all the csv files in files/csv_files
    for file in glob.glob(dir_path):
        # read csv file
        file_csv, length, key = read_file(file)
        # Assign a class to each picture
        file_csv=assign_class(file_csv, length, key)
        # create 3 txt files for each csv file: train, valid, test
        create_txt(file_csv, length, key)
        # Create data repartitions folders and copy pictures
        create_repartitions(file_csv, length, key)
