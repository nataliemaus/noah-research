import pandas as pd 
import numpy as np
from weighted_retraining.weighted_retraining.chem.chem_utils import get_guacamol_score_func

def save_data_as_csv(obj_func_key='valt'):
    # /home/nmaus/noah-research/BO/T-LBO/
    path_to_siles = "weighted_retraining/data/chem/zinc/orig_model/all.txt"
    save_path = "weighted_retraining/data/chem/zinc/orig_model/" + obj_func_key + "_all.csv"

    smiles_data = pd.read_csv(path_to_siles, header=None).values.squeeze()

    score_func = get_guacamol_score_func(obj_func_key)
    scores = []
    for ix, smile in enumerate(smiles_data):
        score = score_func(smile) # smile_to_guacamole_score_func(obj_func_key, smile)
        scores.append(score)
        if ix % 100 == 0:
            print('new scores:', scores[-10:])
            score_array = np.array(scores)
            print("saving array of shape:", score_array.shape)
            print(f'evaluated {len(scores)} scores out of {len(smiles_data)}')
            pd.DataFrame(score_array).to_csv(save_path, header=None, index=None)

# above was actually pointless b/c they're are subbpased to be .pkl files hollding DICTIONARIES with keys as smiles and values as scores, whoops


# def convert_csv_to_pkl(obj_func_key='valt')
#     csv_path = "weighted_retraining/data/chem/zinc/orig_model/" + obj_func_key + "_all.csv"
#     df = pd.read_csv(csv_path, header=None).values.squeeze()
#     pkl_path = "weighted_retraining/data/chem/zinc/orig_model/" + obj_func_key + "_all.pkl"
#     df.to_pickle(pkl_path)  
