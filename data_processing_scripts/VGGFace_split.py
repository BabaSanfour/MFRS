"""
    For each folder: a folder contains all different pictures for one identity
    - We take all the pictures and split them into three classes: Train, Valid and Test.
    - Create data repartitions folders and copy pictures into train/id_picture valid/id_picture test/id_picture (generate new ids)
"""
import os
import sys
import glob
import shutil
import random
import numpy as np
sys.path.append("../../MFRS")
from utils.config import study_path
fold = os.path.join(study_path, "VGGface2_HQ_cropped/VGGface2_HQ_cropped/*")
new_fold = os.path.join(study_path, "VGGface2/")

def split(a,b,c):
    """
        Give a random class for each picture
    """
    if (a<0 or b<0 or c<0):
        print('We have a problem: code 1 ')
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


def assign_class(list):
    number_files = len(list)
    # get the list with number of occurence for a picture in each class
    train_count = int(np.around(number_files*0.7))
    test_count = int(np.around(number_files*0.15))
    valid_count = number_files - test_count - train_count
    counts= [train_count, valid_count, test_count]
    # assign the class randomly and add it to csv file
    class_dic = {}
    for picture in list:
        picture_class=split(counts[0],counts[1],counts[2])
        counts[picture_class]=counts[picture_class]-1
        class_dic[picture]=picture_class
    return class_dic

def create_repartitions(folder, new_folder, class_dic):
    # Create data repartitions folders and copy pictures

    train_folder=new_fold + 'train/%s/'%(new_folder)
    test_folder=new_fold + 'test/%s/'%(new_folder)
    valid_folder=new_fold + 'valid/%s/'%(new_folder)

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)
    for picture, cl in class_dic.items():

        im=os.path.join(folder, picture)
        if cl == 0:
            shutil.move(im, train_folder)
        elif cl == 1:
            shutil.move(im, valid_folder)
        else :
            shutil.move(im, test_folder)


if __name__ == '__main__':
    # run through all the csv files in files/csv_files
    i=0
    for folder in glob.glob(fold):
        list = os.listdir(folder) # list of pictures in an id folder
        # Assign a class to each picture
        class_dic=assign_class(list)
        #create new id
        new_folder = '0' * (4-len(str(i))) + str(i)
        # Create data repartitions folders and copy pictures
        create_repartitions(folder,new_folder, class_dic)
        i+=1
