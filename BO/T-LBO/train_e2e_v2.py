# training end to end... 
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
import sys
import pandas as pd
from weighted_retraining.weighted_retraining.chem.jtnn.mol_tree import Vocab, MolTree
from weighted_retraining.weighted_retraining.chem.jtnn.datautils import tensorize
from weighted_retraining.weighted_retraining.chem.jtnn import MolTreeFolder, MolTreeDataset, Vocab, MolTree
rdkit_quiet()

torch.cuda.set_device(1)

# tmux attach -t e2e
# TO DO: add wandb track (loss, elbo, mdn)
# save ckpt models periodically!


n_train_epochs = 20
bsz = 256
learning_rte = 0.001
save_models_freq = 1 # save every f epochs
track_run = True
project_nm = "train_end2end"
run_id = 0
samps_per_z = 20
model_type = "MDN" # "PPGPR"  # "MDN"

pretrained_models_folder = "mdn_trained_models/"
mdn_model_name = "mdn0_DATAenc_vanilla_trainY_reg_nepochs25_bsz128_lr0.001.pt"
pretrained_surrogate_path  = pretrained_models_folder + mdn_model_name


debug = False
if debug: 
    bsz = 2
    n_train_epochs = 2
# print(train_x.shape ) # (218969,)
# print(len(train_x)) # 218969

path_to_vanilla_model = "weighted_retraining/assets/pretrained_models/chem_vanilla/chem.ckpt" 
vocab_file = "weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
with open(vocab_file) as f:
    vcb= Vocab([x.strip() for x in f.readlines()])
vae: JTVAE = JTVAE.load_from_checkpoint(path_to_vanilla_model, vocab=vcb)
# vae = jtvae_wrapper.jtvade

verbose = False
def latent_z_to_max_logP(z_vectors, samps_per_z = samps_per_z):
    z_tree_vecs = z_vectors[:,0:28].cuda() # .float()
    z_mol_vecs = z_vectors[:,28:56].cuda() # .float()
    if verbose: 
        print("getting max log ps from zs")
        print("tree vecs shape",z_tree_vecs.shape ) # tree vecs shape (218969, 28)
        print("mols vecs shape",z_mol_vecs.shape ) # mols vecs shape (218969, 28)
    logP_arrays = []
    failures = 0
    for i in range(samps_per_z):
        vae.cuda()
        reconstructed_smiles = vae.jtnn_vae.decode(z_tree_vecs.cuda(), z_mol_vecs.cuda(), prob_decode = True)
        if verbose: 
            print(f"finished decode sample {i} /  {samps_per_z}")
        logPs = []
        for smile in reconstructed_smiles: 
            logp = penalized_logP(smile)
            if logp is not None:
                logPs.append(logp) 
            else:
                print("uh oh, invalid smile:", smile)
                logPs.append(np.nan)
                failures += 1
                print("num fails", failures)
        logPs = np.array(logPs)
        if verbose:
            print(f"shape logPs sampe {i}", logPs.shape) # (num_zs,) ie (3,)
        logP_arrays.append(logPs)
    all_logps = np.vstack(logP_arrays)
    # print("all shape", all_logps.shape) # all shape (40 , num_zs) ie (40,3)
    max_log_ps = np.nanmax(all_logps, axis = 0)
    if verbose:
        print("FINAL Y TRAIN SHAPE", max_log_ps.shape) # (num_zs), ie (3,), YAY! 
    return max_log_ps # model.jtnn_vae.decode(z_tree_vecs, z_mol_vecs, prob_decode = True)


# print("METRIC LOSS SHOULD BE NONE!!:", vae.metric_loss) # None :) 
latent_dim = 56
mdn = MDN(input_dim=latent_dim , num_train=-1, hidden_dims=(1000, 1000, 1000, 500, 50)).cuda()
optimizer = torch.optim.Adam([
        {'params': vae.parameters()},
        {'params': mdn.parameters()},
        ], lr=learning_rte)

if pretrained_surrogate_path != "None":
    mdn.load_state_dict(torch.load(pretrained_surrogate_path))
    print("loaded pretrained mdn model")
if track_run: 
    tracker = wandb.init(
        entity="nmaus", project=project_nm, 
        config={ "run_id": run_id ,
        "n_train_epochs": n_train_epochs,
        "batch_size": bsz, "vae_model": "JTVAE", "DATA": "Huawei",
        "surrogate_model": model_type, "pretrained_surrogate_model_path": pretrained_surrogate_path,
        "optim":"adam", "learning_rte": learning_rte,
        "latent_dim":latent_dim, "samples_per_z":samps_per_z,})

mdn.train()
vae.train()
mdn.cuda()
vae.cuda()
train_data_path="weighted_retraining/data/chem/zinc/orig_model/tensors_train" # preprocessed pkl files w/ mol objs
folder = MolTreeFolder(train_data_path, vocab=vae.jtnn_vae.vocab, batch_size=bsz,num_workers=1,)

for i in range(n_train_epochs):
    for batch in folder: 
        batch_x = batch[0]
        optimizer.zero_grad()

        tree_batch, jtenc_holder, mpn_holder, jtmpn_holder = tensorize(batch_x, vae.jtnn_vae.vocab, assm=True)
        x_tree_vecs, x_tree_mess, x_mol_vecs = vae.jtnn_vae.encode(jtenc_holder, mpn_holder)
        z_tree_vecs, tree_kl = vae.jtnn_vae.rsample(x_tree_vecs.cuda(), vae.jtnn_vae.T_mean, vae.jtnn_vae.T_var)
        z_mol_vecs, mol_kl = vae.jtnn_vae.rsample(x_mol_vecs.cuda(), vae.jtnn_vae.G_mean, vae.jtnn_vae.G_var)
        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = vae.jtnn_vae.decoder(tree_batch, z_tree_vecs)
        assm_loss, assm_acc = vae.jtnn_vae.assm(tree_batch, jtmpn_holder, z_mol_vecs, x_tree_mess)
        elbo = word_loss + topo_loss + assm_loss + vae.beta * kl_div
        if verbose:
            print(f"elbo: {elbo.item()}")

        # forward pass surrogate model: 
        z_vectors = torch.cat([z_tree_vecs, z_mol_vecs], axis = 1)
        y_dec = latent_z_to_max_logP(z_vectors.cuda())
        y_dec_valid = y_dec[np.logical_not(np.isnan(y_dec))]
        y_dec_valid = torch.tensor(y_dec_valid).float()
        z_valid = z_vectors[np.logical_not(np.isnan(y_dec))] 
        y_pred = mdn(z_valid.cuda())
        mdn_loss = mdn.loss(y_pred, y_dec_valid.cuda())
        mdn_loss = mdn_loss.sum()
        if verbose:
            print(f"mdn_loss: {mdn_loss.item()}")
        
        # GET FINAL LOSS AND BACK PROP
        loss = mdn_loss + elbo
        print(f"epoch {i+1}/{n_train_epochs}, loss:{loss.item():.4f}", 
                f"elbo:{elbo.item():.4f}, mdn_loss:{mdn_loss.item():.4f}", )
        if track_run: 
            surroate_model_loss = mdn_loss.item()
            tracker.log({ "Loss": loss.item(), "ELBO":elbo.item(), "surroate_model_loss":surroate_model_loss,}) 
        loss.backward()
        optimizer.step()
    if i % save_models_freq == 0:
        string = f"{run_id}_epoch{i+1}"
        torch.save(vae.state_dict(), 'end2end_models/vae' + string  + '.pt')
        torch.save(mdn.state_dict(), 'end2end_models/mdn' + string + '.pt')


string = f"{run_id}_nepochs{n_train_epochs}"
torch.save(vae.state_dict(), 'end2end_models/FINAL_vae' + string + '.pt')
torch.save(mdn.state_dict(), 'end2end_models/FINAL_mdn' + string + '.pt')
print("saved models!")

if track_run: 
    tracker.finish()
