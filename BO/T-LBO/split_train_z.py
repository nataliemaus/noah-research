import pandas as pd
import numpy as np
import sys 


data_files = ["train_data_trip1/", "train_data_vanilla/"]
for file in data_files: 
    path_to_init_train_z = file + 'train_z1.csv'
    train_z= pd.read_csv(path_to_init_train_z, header=None).to_numpy().squeeze()
    print("train z1 shape", train_z.shape) # train z shape (218969, 56)
    print("length of trainz:", len(train_z)) # length of trainz: 218969
    z1_df = pd.DataFrame(train_z)
    z1_df.to_csv(file + 'train_z_pt1.csv', header=None, index = None)


sys.exit()
data_files = ["train_data_trip1/", "train_data_vanilla/"]
for file in data_files: 
    path_to_init_train_z = file + 'train_z.csv'
    train_z= pd.read_csv(path_to_init_train_z, header=None).to_numpy().squeeze()
    print("train z shape", train_z.shape) # train z shape (218969, 56)
    print("length of trainz:", len(train_z)) # length of trainz: 218969
    half_idx = int(len(train_z)/2)
    train_z1 = train_z[0:half_idx]
    train_z2 = train_z[half_idx:len(train_z)]

    print("train z1 shape", train_z1.shape) 
    print("train z2 shape", train_z2.shape) 
    z1_df = pd.DataFrame(train_z1)
    z2_df = pd.DataFrame(train_z2)
    z1_df.to_csv(path_to_init_train_z, header=None, index = None)
    z2_df.to_csv(file + 'train_z2.csv', header=None, index = None)
