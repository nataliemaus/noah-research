from utils.utils_plot import plot_results
import matplotlib.pyplot as plt
import torch 
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet, standardize_smiles, penalized_logP, get_mol, QED_score
from weighted_retraining.weighted_retraining.chem.jtnn import MolTreeFolder, MolTreeDataset, Vocab, MolTree
from weighted_retraining.weighted_retraining.robust_opt_scripts.robust_opt_chem import _encode_mol_trees
from weighted_retraining.weighted_retraining import utils
from weighted_retraining.weighted_retraining.chem.chem_data import (
    WeightedJTNNDataset, tensorize,
    WeightedMolTreeFolder,
    get_rec_x_error)
import sys
from weighted_retraining.weighted_retraining import utils

# tmux attach -t setup
# conda activate lsbo_metric_env 

import numpy as np
rdkit_quiet()

# print(1.40e+01) # 14.0!!! yay! #

path_to_vanilla_model  = "weighted_retraining/assets/pretrained_models/chem_vanilla/chem.ckpt" 
vocab_file = "weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
with open(vocab_file) as f:
    vcb= Vocab([x.strip() for x in f.readlines()])
vae_vanilla: JTVAE = JTVAE.load_from_checkpoint(path_to_vanilla_model, vocab=vcb)

# vae_vanilla.decoder_loss()

logP = penalized_logP('CCCC')
print(logP)

sys.exit()

deterministic = False
plot = False
if plot:
    path_to_res = 'results_triplet'
    plot = plot_results(path_to_res, maximisation= True) 
    plt.savefig('results_triplet.png')
    sys.exit()

# RESULT: 
# path results_triplet/seed0
res_path = "results_triplet/seed0/"
print("res path", res_path)
# Best Penalized Log P:
# 24.331820290876678
# Best Mol:
s1 = "Clc1ccc2ccc3cc4c(c(Cl)cc5c(Cl)cc6ccc7c(Nc8c(Cl)c(Cl)c"
s2 = "(Nc9c(Cl)c(Cl)c(Cl)c%10ccc%11c%12cc%13cc%14cc%15ccccc%15cc%14c(Cl)c%13c"
s3 = "(Cl)c%12cc(Cl)c%11c9%10)c(Cl)c8Nc8c(Cl)c(Cl)c(Cl)c9ccc%10c%11cc%12cc%13cc%14cc" 
s4 = "ccc%14cc%13c(Cl)c%12c(Cl)c%11cc(Cl)c%10c89)c(Cl)c(Cl)c(Cl)c7c6c54)c(Cl)c3c2c1"
smile = s1 + s2 + s3 + s4 

logP = penalized_logP(smile) # Logp 24.331820290876678
# mol = get_mol(smile) #gets mol object
# mol = standardize_smiles(smile)
# print("MOL", mol)
print("Initial Logp", logP)
# Clc1ccc2ccc3cc4c(c(Cl)cc5c(Cl)cc6ccc7c(Nc8c(Cl)c(Cl)c(Nc9c(Cl)c(Cl)c(Cl)c%10ccc%11c%12cc%13cc%14cc%15ccccc%15cc%14c(Cl)c%13c(Cl)c%12cc(Cl)c%11c9%10)c(Cl)c8Nc8c(Cl)c(Cl)c(Cl)c9ccc%10c%11cc%12cc%13cc%14ccccc%14cc%13c(Cl)c%12c(Cl)c%11cc(Cl)c%10c89)c(Cl)c(Cl)c(Cl)c7c6c54)c(Cl)c3c2c1
# Best Model Version: 8
# possible model verisons:  [0 1 2 3 4 5 6 7 8 9]

path_to_vanilla_model  = "weighted_retraining/assets/pretrained_models/chem_vanilla/chem.ckpt" 
path_to_best_model= res_path + "retraining/retrain_450/checkpoints/last.ckpt"
path_to_init_triplet_model  = "weighted_retraining/assets/pretrained_models/chem_triplet/chem.ckpt" 

# ckpt = torch.load(path_to_model) 

vocab_file = "weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
with open(vocab_file) as f:
    vcb= Vocab([x.strip() for x in f.readlines()])

vae_vanilla: JTVAE = JTVAE.load_from_checkpoint(path_to_vanilla_model, vocab=vcb)
vae_best: JTVAE = JTVAE.load_from_checkpoint(path_to_best_model, vocab=vcb)
vae_trip: JTVAE = JTVAE.load_from_checkpoint(path_to_best_model, vocab=vcb)
vae_best.eval() 
vae_vanilla.eval() 
vae_trip.eval()

models = [vae_best, vae_vanilla, vae_trip]
model_strings = ["FineTuned_JTVAE", "regular_JTVAE", "triplet_init_JTVAE"] 

idx = 0
for vae in models: 
    print("")
    print(model_strings[idx])

    if deterministic: 
        print("Deterministic")
        mol_tree = tensorize(smile, assm=True)
        # Encode Best Mol: 
        latent_code = _encode_mol_trees(vae, [mol_tree]) # , batch_size=1)
        # latent_code = vae(mol)
        latent_code = torch.tensor(latent_code)
        latent_code.to(device = vae.device, dtype=torch.float64)
        print("LATENT CODE", latent_code)
        print("shape latenet code", latent_code.shape)
        # Now Decode Best Mol:
        with torch.no_grad(): 
            smile_out = vae.decode_deterministic(latent_code )[0] # single element list 
        print("smile out:", smile_out) 
        logP_out = penalized_logP(smile_out)
        print("logP out", logP_out )

    else: 
        print("Non Deterministic")
        smiles_list = [smile]
        logPs = []
        for i in range(3):
            smile_out = vae.jtnn_vae.reconstruct_smiles2smiles(smiles_list, prob_decode=False)[0]
            # Note: Prob decode applies a softmax type thing, still get high scores, unsure what's better 
                # but probably want to stick w/ false b/c that's what they had as default? 
            logP_out = penalized_logP(smile_out)
            # print("log P out:", logP_out)
            logPs.append( logP_out )
        logPs = np.array(logPs)
        mean_logP = np.mean(logPs)
        stdev_logP = np.std(logPs)
        print("Mean logP:", mean_logP, "std:", stdev_logP)
    idx += 1
# non deterministic: (100 runs)

# FineTuned_JTVAE 
# Non Deterministic
# Mean logP: 18.3820816659089 std: 0.18790466015474186

# regular_JTVAE
# Non Deterministic
# Mean logP: -0.9393175302546476 std: 1.1102230246251565e-16

# triplet_init_JTVAE (OKAY INITIAL TRIPLET VAE IS GREAT too! I think extra retraining steps don't matter?!)
# Q: is this model updated Over time?? 
# Non Deterministic
# Mean logP: 18.733921891694276 std: 0.33934347940750553

#Deterministic: 

# last checkpoint vae: 
# smile: Clc1ccc2c(ccc3c(Cl)cc4c(cc(Cl)c5cc6c(ccc7c(Cl)c(Nc8c(Cl)cc(Cl)
# c(Cl)c8Cl)c(Nc8c(Cl)c(Cl)c(Nc9c(Cl)c(Cl)c(Cl)c%10c%11c(ccc9%10)C(Cl)
# Cc9c-%11cc%10cc%11cc%12ccccc%12cc%11c(Cl)c%10c9Cl)c(Cl)c8Cl)c(Cl)c76)
# c(Cl)c54)c32)c1
# logP out 18.5223573160254 !! 
# path_to_model results_triplet/seed0/retraining/retrain_450/checkpoints/last.ckpt


# same thing with vanilla JTNN: 
# smile out: CCl
# logP out -0.9393175302546475
# path_to_model weighted_retraining/assets/pretrained_models/chem_vanilla/chem.ckpt


def get_good_mol_smile(): 
    s1 = "Clc1ccc2ccc3cc4c(c(Cl)cc5c(Cl)cc6ccc7c(Nc8c(Cl)c(Cl)c"
    s2 = "(Nc9c(Cl)c(Cl)c(Cl)c%10ccc%11c%12cc%13cc%14cc%15ccccc%15cc%14c(Cl)c%13c"
    s3 = "(Cl)c%12cc(Cl)c%11c9%10)c(Cl)c8Nc8c(Cl)c(Cl)c(Cl)c9ccc%10c%11cc%12cc%13cc%14cc" 
    s4 = "ccc%14cc%13c(Cl)c%12c(Cl)c%11cc(Cl)c%10c89)c(Cl)c(Cl)c(Cl)c7c6c54)c(Cl)c3c2c1"
    smile = s1 + s2 + s3 + s4 
    return smile