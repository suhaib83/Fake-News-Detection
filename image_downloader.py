import pandas as pd
import os
from tqdm import tqdm as tqdm
import urllib.request
import numpy as np



file_name_train = '../Data/multimodal_train.csv'
file_name_valid = '../Data/multimodal_valid.csv'
file_name_test = '../Data/multimodal_test.csv'


df = pd.read_csv(file_name_train,encoding="cp437")
df2 = pd.read_csv(file_name_valid,encoding="cp437")
df3 = pd.read_csv(file_name_test,encoding="cp437")

df = df.replace(np.nan, '', regex=True)
df = df.dropna(how='all')

df2 = df2.replace(np.nan, '', regex=True)
df2 = df2.dropna(how='all')

df3 = df3.replace(np.nan, '', regex=True)
df3 = df3.dropna(how='all')


if not os.path.exists("../Data/images/"):
  os.makedirs("../Data/images/")
  
for index, row in df.iterrows():
  if row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    try:
       urllib.request.urlretrieve(image_url, "../Data/images/" + row["id"] + ".jpg")
    except :
        print("Error.")
        if index in df.index:  
            df.drop(index, inplace=True)
print("done Train ")

for index, row in df2.iterrows():
  if row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    try:
       urllib.request.urlretrieve(image_url, "../Data/images/" + row["id"] + ".jpg")
    except :
        print("Error.")
        if index in df.index:  
            df.drop(index, inplace=True)
print("done valid ")


for index, row in df3.iterrows():
  if row["image_url"] != "" and row["image_url"] != "nan":
    image_url = row["image_url"]
    try:
       urllib.request.urlretrieve(image_url, "../Data/images/" + row["id"] + ".jpg")
    except :
        print("Error.")
        if index in df.index:  
            df.drop(index, inplace=True)
print("done test ")




df.to_csv('../Data/multimodal_train.csv')
df2.to_csv('../Data/multimodal_valid.csv')
df3.to_csv('../Data/multimodal_test.csv')