"""
    For each folder: a folder contains all different pictures for one identity
    - We take all the pictures and split them into three classes: Train, Valid and Test.
    - Create data repartitions folders and copy pictures into train/id_picture valid/id_picture test/id_picture (generate new ids)
"""
import os
import glob
import shutil
import random
import numpy as np

fold = "/home/hamza97/projects/def-kjerbi/hamza97/data/data_MFRS/VGGFace/"
new_fold = "/home/hamza97/projects/def-kjerbi/hamza97/data/data_MFRS/VGGFace2/"

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


def assign_class(list, number_files):
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
        class_dic[list]=picture_class
    return class_dic

def create_repartitions(folder, new_folder, class_dic):
    # Create data repartitions folders and copy pictures

    train_folder=new_fold + 'train/%s/'%(new_folder)
    test_folder=new_fold + 'test/%s/'%(new_folder)
    valid_folder=new_fold + 'valid/%s/'%(new_folder)

    os.makedirs(train_folder)
    os.makedirs(test_folder)
    os.makedirs(valid_folder)

    for picture, class in class_dic.items():
        im=os.path.join(fold+folder+'/', picture)
        if class == 0:
            shutil.copy(im, train_folder)
        elif class == 1:
            shutil.copy(im, valid_folder)
        else :
            shutil.copy(im, test_folder)


if __name__ == '__main__':
    # run through all the csv files in files/csv_files
    i=0
    for folder in glob.glob(dir_path):
        list = os.listdir(folder) # list of pictures in an id folder
        # Assign a class to each picture
        class_dic=assign_class(list)
        #create new id
        new_folder = '0' * (4-len(str(i))) + str(i)
        # Create data repartitions folders and copy pictures
        create_repartitions(folder,new_folder, class_dic)
        i+=1
