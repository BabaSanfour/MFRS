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
import pandas as pd
sys.path.append("MFRS/")
from utils.config import proj_path

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

if __name__ == '__main__':
    dir_path = os.path.join(proj_path, "MFRS/files/")

    # read identity_CelebA.txt
    dir_txt = dir_path + 'identity_CelebA.txt'
    identity_file = pd.read_csv(dir_txt, sep=' ', header=None)
    # read list_attr_celeba.csv
    dir_csv = dir_path+'list_attr_celeba.csv'
    list_attr_celeba = pd.read_csv(dir_csv)
    # Combine, process, and clean original files
    # create female and male dataframes ( to select equal %)
    df_male, df_female, merged = clean_csv(identity_file, list_attr_celeba)

    # create a dict where :
    # key => number of pictures per id
    # value => total number of ids per gender
    dic = {25:500, 12:500,
            27:250, 13:250,
            28:150, 14:150,
            29:75, 15:75}

    # run through the dic
    for key in dic:
        final=create_csv(key, dic[key], df_male, df_female, merged)
        final=generate_new_ids(final)
        final.to_csv(dir_path+"csv_files/"+"id_%s_PicPerId_%s.csv"%(dic[key]*2, key))
