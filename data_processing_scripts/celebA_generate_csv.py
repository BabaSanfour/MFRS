"""
    Transform list_attr_celeba.txt into list_attr_celeba.csv (run a script)
"""

import os
import sys
import pandas as pd
sys.path.append("../../MFRS")
from utils.config import proj_path

path= os.path.join(proj_path, "files/")
file= os.path.join(path+"list_attr_celeba.txt")
new_file= os.path.join(path+"list_attr_celeba.csv")
df = pd.read_csv(file)
names=df.columns[0]
columns_1= names.split(' ')
columns_1.pop()
df.columns=['hamza']

columns=['name']
for i in (columns_1):
    columns.append(i)
df_new = pd.DataFrame(columns = columns)

j=0
for val in df.hamza:
    values=val.split(' ')
    val_fin=[]
    for i in values:
        if i !='':
            val_fin.append(i)
    df_new.loc[j]=list(val_fin)
    j+=1
df_new.to_csv(new_file)
