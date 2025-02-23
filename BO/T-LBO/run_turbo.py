from approximate_gp import *
from turbo import *
from torch.utils.data import TensorDataset, DataLoader
from mdn import MDN
from ppgpr import GPModel
import gpytorch
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet, penalized_logP
import pandas as pd
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining.chem.jtnn import Vocab
import sys
rdkit_quiet()

# NOTE: CURRENTLY RUNNING: # tmux attach -t setup , setup2, ... 
# 5. 
# BO w/ "reg" and vanilla (wandb tracking table of SMILES for best! )
# 6.
# BO w/ "reg", vanilla, and VANILLA ENC DATA!
# 1. 
# BO W/ "reg", trip1, trip1, faster turbo (num zs = 5)
# 12. 
# BO W/ "reg", trip1, vanilla, (num zs = 5)
# 13. 
# BO W/ "reg", vanilla, vanilla, (num zs = 5)
# 20
# BO W/ "reg", TRIP1, TRIP1, (num zs = 5)
# 7. 
# BO W/ "reg", TRIP1, vanilla, (num zs = 40)
# 15. 
# trip1, vanilla, PPGPR, 40spz
# 21 
# vanilla, vanilla, PPGPR, 40spz

# 3, 4, 9, 10, 11, 16, 17-19

# 22 (failed on trip1, trip1, ppgpr...)
# 14 (I tihnk I made cuda:1 permanent in this one...)


# next # Turbo PPGPR!! ... 
# BO W/ "reg", trip1, vanilla, PPGPR (num zs = 40)  ... !!!


# _______making data_______checkon!_________
# 2. 
# create y_dec trip1!!!! (20 spz)
# 8. 
# create d_dec trip1, MORE CUDA!!... FASTER?? (10spz)

#NOTE: RAN: (before wandb)
# 2.
# BO with best_model.ckpt model and "reg" data
# 219498) Best value: 14.693258285522461, TR length: 1.25e-02
# 219499) Best value: 14.693258285522461, TR length: 1.25e-02
# 219500) Best value: 14.693258285522461, TR length: 6.25e-03
# 3. 
# BO w/ "reg" and vanilla, 34! --> verification :). (I killed early )


torch.cuda.set_device(0)
# conda activate lsbo_metric_env

track_run = True


which_encoder = "trip1"  # ENCODER --> WHICH DATA TO USE! "vanilla", "trip1"
which_jtvae = "trip1" #DECODER "vanilla", "trip1", "trip_best" (decoder in BO loop!)
model_type = "PPGPR" # "PPGPR"  # "MDN"


which_train_y = "reg"  #dec or reg, preferable dec once I can!
num_epochs = 2  #num epochs to fit GP on each run ....
samples_per_z = 40

project_nm = "turbo-mdn-jtvae" # "dumb" # "turbo-mdn-jtvae"
# https://wandb.ai/nmaus/turbo-mdn-jtvae?workspace=user-nmaus

learning_rte = 0.001
bsz = 1 #essential for simple EI w/ MDN!! 


if which_encoder == "vanilla": 
    use_vanilla_train_data = True #Encoder which? 
else: 
    use_vanilla_train_data = False #Encoder which?   

if model_type == "MDN":
    is_mdn = True
    acq_func="ei"
elif model_type == "PPGPR":
    is_mdn = False
    acq_func="ei"

debug_mode = False
verbose = False
optim = "adam"

if use_vanilla_train_data: 
    train_data_folder = 'train_data_vanilla/'
    which_encoder = "vanilla"
else: 
    which_encoder = "trip1"
    train_data_folder = 'train_data_trip1/'

path_to_train_z1 =  train_data_folder + 'train_z_pt1.csv'
path_to_train_z2 =  train_data_folder + 'train_z2.csv'
train_z1 = pd.read_csv(path_to_train_z1, header=None).to_numpy().squeeze()
train_z1 = torch.from_numpy(train_z1).float()
train_z2 = pd.read_csv(path_to_train_z2, header=None).to_numpy().squeeze()
train_z2 = torch.from_numpy(train_z2).float()
train_z = torch.cat([train_z1,train_z2])
# train_z= pd.read_csv(path_to_train_z, header=None).to_numpy().squeeze()
init_len_train_z = train_z.shape[0]
print("train z shape", train_z.shape) # train z shape torch.Size([218969, 56])
if which_train_y == "dec": 
    path_to_train_y = train_data_folder + 'train_y_dec.csv'
elif which_train_y == "reg":
    path_to_train_y = train_data_folder + 'train_y.csv'
train_y= pd.read_csv(path_to_train_y, header=None).to_numpy() 
train_y = torch.from_numpy(train_y).float()
print("train y shape", train_y.shape) #train y shape torch.Size([218969])
init_max = torch.max(train_y).item()
print(f"CURRENT BEST TRIAN Y {init_max}")
# CURRENT BEST TRIAN Y tensor(4.5208) !!! 

dim = train_z.size(-1)
n_candidates = min(5000, max(2000, 200 * dim))
num_restarts = 10
raw_samples = 512

# get fine-tuned JTVAE model 
path_to_vanilla= "weighted_retraining/assets/pretrained_models/chem_vanilla/chem.ckpt" 
path_to_best_model= "results_triplet/seed0/retraining/retrain_450/checkpoints/last.ckpt"
path_to_init_triplet_model  = "weighted_retraining/assets/pretrained_models/chem_triplet/chem.ckpt" 
if which_jtvae == "trip1": 
    path_to_vae = path_to_init_triplet_model
elif which_jtvae == "vanilla": 
    path_to_vae = path_to_vanilla
elif which_jtvae == "trip_best": 
    path_to_vae = path_to_best_model

vocab_file = "weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
with open(vocab_file) as f:
    vcb= Vocab([x.strip() for x in f.readlines()])
vae_model: JTVAE = JTVAE.load_from_checkpoint(path_to_vae, vocab=vcb).cuda()
print("loaded vae")

total_ultimate_fails = 0
# Now non-deterministic: 
def latent_z_to_max_logP(z_vectors, samps_per_z = samples_per_z):
    global total_ultimate_fails
    z_tree_vecs = z_vectors[:,0:28] # .float()
    # print("tree vecs shape",z_tree_vecs.shape ) tree vecs shape (218969, 28)
    z_mol_vecs = z_vectors[:,28:56]  # .float()
    # print("mols vecs shape",z_mol_vecs.shape ) # mols vecs shape (218969, 28)
    logP_arrays = []
    all_smiles_arrays = []
    failures = 0
    for i in range(samps_per_z):
        reconstructed_smiles = vae_model.jtnn_vae.decode(z_tree_vecs, z_mol_vecs, prob_decode = True)
        # paper uses prob_decode = False --> deterministic decode! 
        logPs = []
        for smile in reconstructed_smiles:
            logp = penalized_logP(smile)
            if logp is not None:
                logPs.append(logp)
            else:
                print("uh oh, invalid smile:", smile)
                logPs.append(0)
                failures += 1
                if failures > (samps_per_z - 1):
                    print('ULTIMATE FAILURE, NO VALID SMILES')
                    total_ultimate_fails += 1
                    return None, None
        logPs = np.array(logPs)
        all_smiles_per_sample = np.array(reconstructed_smiles)
        all_smiles_arrays.append(all_smiles_per_sample)
        # print(f"shape logPs sampe {i}", logPs.shape) # (num_zs,) ie (1,) always for BO!
        logP_arrays.append(logPs)
    all_smiles = np.vstack(all_smiles_arrays)
    all_logps = np.vstack(logP_arrays)
    # print("all shape", all_logps.shape) # all shape (40 , num_zs) ie (40,3)
    max_log_ps = np.max(all_logps, axis = 0)
    idx = np.argmax(all_logps, axis = 0)
    max_smiles = all_smiles[idx]
    # print("max smiles", max_smiles)
    # print("max smiles shape", max_smiles.shape)
    # print("FINAL max log p shape:", max_log_ps.shape) # (num_zs), ie (3,), YAY! In this case always (1,)!!
    return torch.tensor(max_log_ps).float(), max_smiles # model.jtnn_vae.decode(z_tree_vecs, z_mol_vecs, prob_decode = True)

if model_type == "MDN":
    model = MDN(input_dim=dim, num_train=-1, hidden_dims=(1000, 1000, 1000, 500, 50)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rte)
    pretrained_models_folder = "mdn_trained_models/"
    id_num = "0"
    model_name = "mdn" + id_num + "_DATAenc_" + which_encoder  + "_trainY_" + which_train_y + "_nepochs25_bsz128_lr0.001.pt"
    pretrained_model_path = pretrained_models_folder + model_name

elif model_type == "PPGPR": 
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = GPModel(train_z[:128, :].cuda(), likelihood = likelihood ).cuda()
    mll = PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.size(-2))
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        ], lr=learning_rte)
    pretrained_models_folder = "ppgpr_trained_models/"
    id_num = "0"
    model_name = "ppgpr" + id_num + "_DATAenc_" + which_encoder  + "_trainY_" + which_train_y + "_nepochs25_bsz128_lr0.001.pt"
    pretrained_model_path = pretrained_models_folder + model_name

if pretrained_model_path is not None:
    model.load_state_dict(torch.load(pretrained_model_path))
    print("loaded pretrained model")

# model = model.cuda()
state = TurboState(dim, batch_size=bsz)
dtype = torch.float32

if track_run: 
    tracker = wandb.init(
        entity="nmaus", project=project_nm, 
        config={ "use_vanilla_train_data": use_vanilla_train_data,
        "which_jtvae": which_jtvae, "num_restarts": num_restarts,
        "pretrained_mdn_path": pretrained_model_path,"model_type": model_type,
        "which_train_y":which_train_y ,"init_best_in_trainY": init_max,
        "bsz": bsz, "optim": optim, "learning_rte": learning_rte, 
        "num_epochs": num_epochs, "dim":dim, "which_encoder":which_encoder,
        "raw_samples": raw_samples,"samples_per_z":samples_per_z,
        "n_candidates": n_candidates,"path_to_jtvae_used":path_to_vae,})
    
    # cols = ["logPs", "smiles"]
    # smiles_table = wandb.Table(columns=cols)

best_logP_seen = -10000000     # init_max, # WAS init_max for setup5 run... so won't log smiles strings still after 4ish... 
while not state.restart_triggered:
    # Train a GP
    x_lb = train_z.min(axis=0)[0].min().item()
    x_ub = train_z.max(axis=0)[0].max().item()
    X = (train_z - x_lb) / (x_ub - x_lb)

    # if model_type == "MDN":
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rte)
    model.train()
    model.cuda()
    
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda().squeeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=bsz*2000)
    for i in range(num_epochs):
        for ix, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(batch_x)
            if is_mdn:
                loss = model.loss(output, batch_y)
            else:
                loss = -mll(output, batch_y)
            if verbose and ix % 500 == 0:
                print(f'Iteration {ix}, -- loss={loss.item()}')
            loss.sum().backward()
            optimizer.step()

    
    # likelihood(model(test_x))  # Returns the (approximate) predictive posterior distribution at test_x
    # Generate batch with TuRBO
    z_next = generate_batch(
        state=state,
        model=model,
        X=X,
        Y=train_y,
        batch_size=bsz,
        n_candidates=n_candidates,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        acqf=acq_func,
        mdn=is_mdn,
    )
    
    z_next = (z_next) * (x_ub - x_lb) + x_lb
    y_next, from_smiles = latent_z_to_max_logP(z_next.cuda())
    if y_next is not None:
        i = 0
        for logP_val in y_next: 
            if verbose:
                print("y_next", logP_val.item(), "form smiles", from_smiles[0][i])
            if track_run and (logP_val.item() >= best_logP_seen): 
                # print("logging new best smile:")
                print("NEW BEST", logP_val.item(), "FROM SMILE", from_smiles[0][i]) 
                best_logP_seen = logP_val.item() #update best
                # smiles_table.add_data(logP_val.item(), str(from_smiles[0][i]) )
            i += 1


        y_next = y_next.unsqueeze(-1)
        # print("y next shape", y_next.shape) # y next shape torch.Size([1, 1])
        # print("train y shape", train_y.shape) # train y shape torch.Size([218975, 1])
        state = update_state(state=state, Y_next=y_next)
        train_z = torch.cat((train_z.cuda(), z_next.cuda()), dim=-2)
        train_y = torch.cat((train_y.cuda(), y_next.cuda()), dim=-2)


    # Print current status
    if verbose: 
        print( f"{len(train_z)}) Best value: {state.best_value}, TR length: {state.length:.2e}")
    if track_run: 
        tracker.log({"train_z_size":len(train_z), "Best_val": state.best_value, "TR_length":state.length, "n_func_evals":len(train_z) - init_len_train_z}) 

if track_run:
    tracker.log({"total_ultimate_fails":total_ultimate_fails})
    # tracker.log({"smiles_table":smiles_table})
    tracker.finish()
print("path to VAE used:", path_to_vae)
