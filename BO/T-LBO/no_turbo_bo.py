from logging import debug
from approximate_gp import *
from turbo import *
from torch.utils.data import TensorDataset, DataLoader
from mdn import MDN
import gpytorch
import torch
import numpy as np
from botorch.models import SingleTaskGP
import wandb
from torch.utils.data import DataLoader
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet, penalized_logP
import pandas as pd
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining.chem.jtnn import Vocab
import sys
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from ppgpr import GPModel
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood 
from botorch.fit import fit_gpytorch_model
rdkit_quiet()
# conda activate lsbo_metric_env  
torch.cuda.set_device(1)

track_run = True
debug_mode = False #smaller trianz
model_type = "MDN" # "PPGPR"  # "MDN" #SVGP
which_encoder = "trip1"  # ENCODER --> WHICH DATA TO USE! "vanilla", "trip1"
which_decoder = "trip1" # "vanilla", "trip1", "trip_best" (decoder in BO loop!)
max_func_evals = 2000

samples_per_z = 40
which_train_y = "reg"  #dec or reg, preferable dec once I can!
num_epochs = 2  #num epochs to fit GP on each run .... 
project_nm = "jtvae-logp-NOturbo"

if which_encoder == "vanilla": 
    use_vanilla_train_data = True #Encoder which?
else: 
    use_vanilla_train_data = False #Encoder which?

learning_rte = 0.001 #SAME AS PAPER

if model_type == "MDN":
    is_mdn = True
    acq_func="ei"
    bsz = 1
elif model_type == "PPGPR" or model_type == "SVGP":
    is_mdn = False
    acq_func="ts"
    bsz = 5
elif model_type == "GP":
    is_mdn = False
    bsz = 1
    acq_func = "ei"

verbose = False
optim = "adam"

data_folder = 'train_data_trip1/'
if use_vanilla_train_data:
    data_folder = 'train_data_vanilla/'
# load data train z
path_to_train_z1 =  data_folder + 'train_z_pt1.csv'
path_to_train_z2 =  data_folder + 'train_z2.csv'
train_z1 = pd.read_csv(path_to_train_z1, header=None).to_numpy().squeeze()
train_z1 = torch.from_numpy(train_z1).float()
train_z2 = pd.read_csv(path_to_train_z2, header=None).to_numpy().squeeze()
train_z2 = torch.from_numpy(train_z2).float()
train_z = torch.cat([train_z1,train_z2])
init_len_train_z = train_z.shape[0]
print("init_len_train_z", init_len_train_z)
# print("train z shape", train_z.shape) # train z shape torch.Size([218969, 56])
# Load data train y
if which_train_y == "dec": 
    path_to_train_y = data_folder + 'train_y_dec.csv'
elif which_train_y == "reg":
    path_to_train_y = data_folder + 'train_y.csv'
train_y= pd.read_csv(path_to_train_y, header=None).to_numpy() 
train_y = torch.from_numpy(train_y).float()
print("train y shape", train_y.shape) #train y shape torch.Size([218969])
init_max = torch.max(train_y).item()
print(f"CURRENT BEST TRIAN Y {init_max}")
# CURRENT BEST TRIAN Y tensor(4.5208) !!!

if debug_mode:
    train_z = train_z[0:400]
    print("train_z small shape", train_z.shape)
    train_y = train_y[0:400]
    print("train_y small shape", train_y.shape)

dim = train_z.size(-1)
n_candidates = min(5000, max(2000, 200 * dim))
num_restarts = 10
raw_samples = 512

# get JTVAE model 
# path_to_best_model= "results_triplet/seed0/retraining/retrain_450/checkpoints/last.ckpt"
path_to_vanilla= "weighted_retraining/assets/pretrained_models/chem_vanilla/chem.ckpt" 
path_to_init_triplet_model  = "weighted_retraining/assets/pretrained_models/chem_triplet/chem.ckpt" 
if which_decoder == "trip1": 
    path_to_vae = path_to_init_triplet_model
elif which_decoder == "vanilla": 
    path_to_vae = path_to_vanilla
# elif which_decoder == "trip_best": 
#     path_to_vae = path_to_best_model
vocab_file = "weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
with open(vocab_file) as f:
    vcb= Vocab([x.strip() for x in f.readlines()])
vae_model: JTVAE = JTVAE.load_from_checkpoint(path_to_vae, vocab=vcb)
vae_model = vae_model.cuda()
print("loaded vae")
total_ultimate_fails = 0

# Now non-deterministic: 
def latent_z_to_max_logP(z_vectors, samps_per_z = samples_per_z):
    global total_ultimate_fails
    z_tree_vecs = z_vectors[:,0:28].cuda() # .float()
    # print("tree vecs shape",z_tree_vecs.shape ) tree vecs shape (218969, 28)
    z_mol_vecs = z_vectors[:,28:56].cuda()  # .float()
    # print("mols vecs shape",z_mol_vecs.shape ) # mols vecs shape (218969, 28)
    logP_arrays = []
    all_smiles_arrays = []
    failures = 0
    for i in range(samps_per_z):
        reconstructed_smiles = vae_model.jtnn_vae.decode(z_tree_vecs, z_mol_vecs, prob_decode = True)
        logPs = []
        for smile in reconstructed_smiles:
            logp = penalized_logP(smile)
            if logp is not None:
                logPs.append(logp)
            else:
                print("uh oh, invalid smile:", smile)
                logPs.append(0)
                failures += 1
                if failures > (samps_per_z - 10):
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

elif model_type == "GP":
    # Create GP model 

    # class ExactGPModel(gpytorch.models.ExactGP):
    #     num_outputs = 1
    #     def __init__(self, train_z, train_y, likelihood):
    #         super(ExactGPModel, self).__init__(train_z, train_y, likelihood)
    #         self.mean_module = gpytorch.means.ConstantMean()
    #         self.covar_module = gpytorch.kernels.RBFKernel() 
        
    #     def forward(self, x):
    #         mean_x = self.mean_module(x)
    #         covar_x = self.covar_module(x)
    #         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    # likelihood = gpytorch.likelihoods.GaussianLikelihood( ) 
    # model = ExactGPModel(train_z.cuda(), train_y.cuda(),likelihood ) 
    # mll = ExactMarginalLogLikelihood(model.likelihood, model)
    covar_module = gpytorch.kernels.RBFKernel().cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood( ).cuda()
    pretrained_model_path  = None

elif model_type == "PPGPR":
    model = GPModel(train_z[:bsz*200, :].cuda()).cuda()
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    mll = PredictiveLogLikelihood(likelihood, model, num_data=train_z.size(-2))
    pretrained_model_path  = None

elif model_type == "SVGP":
    pretrained_model_path  = None

if pretrained_model_path is not None:
    model.load_state_dict(torch.load(pretrained_model_path))
    print("loaded pretrained model")

dtype = torch.float32

if track_run: 
    tracker = wandb.init(
        entity="nmaus", project=project_nm,
        config={ "use_vanilla_train_data": use_vanilla_train_data,
        "which_decoder": which_decoder, "which_encoder": which_encoder, "num_restarts": num_restarts,
        "pretrained_mdn_path": pretrained_model_path, "acq_func": acq_func, 
        "which_train_y":which_train_y ,"init_best_in_trainY": init_max,
        "bsz": bsz, "optim": optim, "learning_rte": learning_rte, "max_func_evals":max_func_evals,
        "num_epochs": num_epochs, "dim":dim, "model_type": model_type,
        "raw_samples": raw_samples,"samples_per_z":samples_per_z,
        "n_candidates": n_candidates,"path_to_jtvae_used":path_to_vae,})
    
    # cols = ["logPs", "smiles"]
    # smiles_table = wandb.Table(columns=cols)
verbose2 = False
best_logP_seen = -10000000 
restart_triggered = False
while not restart_triggered:
    if verbose2:
        print("starting bo")
    # Train a GP (or MDN)
    if model_type == "GP":
        model = SingleTaskGP(train_z.cuda(), train_y.cuda(), likelihood, covar_module)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.likelihood.train()
        model.train()
        model.cuda()
        # fit_gpytorch_model(mll)
        # torch.cuda.empty_cache()
        # print("fit entire model")

    elif model_type == "PPGPR":
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
            ], lr=learning_rte)
    elif model_type == "SVGP":
        if verbose2:
            print("getting SVGP")
        likelihood = gpytorch.likelihoods.GaussianLikelihood( ) #.cuda()
        if verbose2:
            print("gOT LIKELIHOOD, GETTING MODEL")
        # likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        # model = SingleTaskVariationalGP(train_z[:10, :], train_y[:10, :], likelihood=likelihood) #.cuda() 
        model = SingleTaskVariationalGP(train_z[:bsz*200, :], train_y[:bsz*200, :], likelihood=likelihood) #.cuda() 
        mll = VariationalELBO(model.likelihood, model.model.cuda(), num_data = train_z.shape[0])
        optimizer = torch.optim.Adam([
            {'params': model.model.parameters()},
            {'params': model.likelihood.parameters()},
            ], lr=0.01)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rte)

    model.cuda()
    model.train()
    train_dataset = TensorDataset(train_z.cuda(), train_y.cuda().squeeze(-1))
    train_loader = DataLoader(train_dataset, batch_size=2000)
    if verbose:
        print("start training!")
    for i in range(num_epochs):
        for ix, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(batch_x.cuda())
            if model_type == "MDN":
                loss = model.loss(output, batch_y)
                loss.sum().backward()
            else:
                loss = -mll(output, batch_y)
                loss.sum().backward()
            if verbose and ix % 2 == 0:
                print(f'Iteration {ix}, -- loss={loss.item()}')
            optimizer.step()

    # print('AIGHT')
    # restart_triggered = True
    # NOTE: I THINK GP CANT DO BATCH?!!

    ## IDKKKK !!!

    # Create acquisition function and optimize it to get a candidate
    if acq_func == "ei":
        if model_type == "MDN":
            ei = ExpectedImprovementMDN(model.cuda(), train_y.max().cuda(), maximize=True) #I wrote in turbo.py
        # elif model_type == "PPGPR":
        #     model.num_outputs = 1
        #     ei = qExpectedImprovement(model.cuda(), train_y.max().cuda(), maximize=True)
        else:
            ei = ExpectedImprovement(model.cuda(), best_f=train_y.max().cuda(), maximize=True).cuda()
        bounds = torch.stack([-3 * torch.ones(train_z.size(-1)), 3 * torch.ones(train_z.size(1))]).cuda()
        candidate, acq_value = optimize_acqf(ei, bounds=bounds, q=bsz, num_restarts=num_restarts, raw_samples=raw_samples,)
    elif acq_func == "ts": #PPGPR
        dim = train_z.size(-1)
        sobol = SobolEngine(dim, scramble=True)
        cand = sobol.draw(n_candidates).to(dtype=dtype).cuda()
        model.eval()
        thompson_sampling = MaxPosteriorSampling(model=model.cuda(), replacement=False)
        candidate = thompson_sampling(cand , num_samples=bsz)
        if verbose: 
            print("shape of candidates from ts", candidate.shape)
            # shape of candidates from ts torch.Size([5, 56]) (bsz x 56)

    # elif acq_func == "ts":

    z_next = candidate # .cuda()
    y_next, from_smiles = latent_z_to_max_logP(z_next)
    if y_next is not None:
        i = 0
        for logP_val in y_next:
            if verbose:
                print("y_next", logP_val.item(), "form smiles", from_smiles[0][i])
            if (logP_val.item() >= best_logP_seen): 
                # print("logging new best smile:")
                print("NEW BEST:", logP_val.item(), "from smile:", from_smiles[0][i])
                best_logP_seen = logP_val.item() #update best
                # if track_run: 
                #     smiles_table.add_data(logP_val.item(), from_smiles[0][i] )
            i += 1

        y_next = y_next.unsqueeze(-1)
        train_z = torch.cat((train_z.cpu(), z_next.cpu()), dim=-2)
        train_y = torch.cat((train_y, y_next), dim=-2)

    n_func_evals = len(train_z) - init_len_train_z
    if track_run: 
        tracker.log({"train_z_size":len(train_z), "Best_val": best_logP_seen, "n_func_evals":len(train_z) - init_len_train_z}) 
    if n_func_evals > max_func_evals: 
        restart_triggered = True

if track_run:
    tracker.log({"total_ultimate_fails":total_ultimate_fails})
    # tracker.log({"smiles_table":smiles_table})
    tracker.finish()
print("path to VAE used (to decode):", path_to_vae)
