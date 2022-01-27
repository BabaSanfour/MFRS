"""
    For each CSV file:
    - We take all the pictures and split them into three classes: Train, Valid and Test.
    - Create three txt files: Each file has Train/Valid/Test pictures and their ids.
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
import glob

if __name__ == '__main__':
    dir_path = "/home/hamza97/projects/def-kjerbi/hamza97/MFRS/files/csv_files/*.csv"

    dic = {25:[15,5,5], 12:[6,3,3],
            27:[17,5,5], 13:[7,3,3],
            28:[18,5,5], 14:[8,3,3],
            30:[20,5,5], 15:[9,3,3]}

    for file in glob.glob(dir_path):
        file_csv=pd.read_csv(file)
        file_csv=csv.drop('Unnamed: 0', axis=1)
        key=file_csv[file_csv['new_id']==1]['new_id'].value_counts()[1]
        value=dic[key]
        dic_nLables={}
        for i in range ():
            dic_nLables[i]=value
