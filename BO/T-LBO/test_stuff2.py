import pandas as pd
import os
import pickle
import os
from pathlib import Path


# SUMMARY: LITERALLY JUST CHANGE property_file_path TO None

# ALLL JUST A BUNCH OF MOL TREE OBJECTS, NO NEED TO CHANGE 
train_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_train"
val_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_val"

path=val_data_path + "/tensors_0000000000-0000024333.pkl" # weighted_retraining/data/chem/zinc/orig_model/tensors_train"
with open(path, "rb") as f: 
    property_dict = pickle.load(f)
print(len(property_dict))
print(type(property_dict))
print(property_dict[0])

vcb_file = False
if vcb_file: # checkout vocab file... literally just 780 smiles so all good there 
    vocab_file_path="weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
    arr = pd.read_csv(vocab_file_path, header=None).values
    print(arr[0:5])
    print(arr.shape) # just smiles strings 
    # 780 x 1

if False: #explore property file, just set to None 
    property_file_path="weighted_retraining/data/chem/zinc/orig_model/pen_logP_all.pkl"
        # set to None, they automatically fill in rest even if don't start w/ 300,000 

    # ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
    # property_file = os.path.join(ROOT_PROJECT, property_file_path)
    
    with open(property_file_path, "rb") as f: 
        property_dict = pickle.load(f)

    # print(property_dict)
    print(len(property_dict))
    # 395,373
    print(property_dict['CCCn1ccc2ccc(NC(=O)C(=O)N3CCC(C(=O)NC)CC3)cc21'])


    keys = property_dict.keys()
    # print(f"num keys {num_keys}")

    # new_dict = dict()
    # for key in keys:


