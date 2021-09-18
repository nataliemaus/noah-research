import torch
from mdn import MDN
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm
# import wandb
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet, penalized_logP
# conda activate base ! 
rdkit_quiet()

which_train_y = "reg"  #dec or reg
which_encoder = "vanilla" # "trip1"

if which_encoder == "vanilla":
    data_file = 'train_data_vanilla/'
elif which_encoder == "trip1":
    data_file = 'train_data/'
path_to_train_z =  data_file + 'train_z.csv'
train_z= pd.read_csv(path_to_train_z, header=None).to_numpy().squeeze()
train_z = torch.from_numpy(train_z).float()
# print("train z shape", train_z.shape) # train z shape torch.Size([218969, 56])
if which_train_y == "dec":
    path_to_train_y = data_file + 'train_y_dec.csv'
elif which_train_y == "reg":
    path_to_train_y = data_file + 'train_y.csv'
train_y= pd.read_csv(path_to_train_y, header=None).to_numpy().squeeze()
train_y = torch.from_numpy(train_y).float()
# print("train y shape", train_y.shape) #train y shape torch.Size([218969])

verbose = True
learning_rte = 0.001
bsz = 128 # 1024?
num_epochs = 25
id_num = "0"
model_name = "mdn" + id_num + "_DATAenc_" + which_encoder  + "_trainY_" + which_train_y + "_nepochs" + str(num_epochs) + "_bsz" + str(bsz) + "_lr" + str(learning_rte)
print("model name", model_name)
# Train MDN:
model = MDN(input_dim=train_z.size(-1), num_train=-1,
            hidden_dims=(1000, 1000, 1000, 500, 50)).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rte)

model = model.cuda()

# train_dataset = TensorDataset(train_z, train_y.squeeze(-1))
train_dataset = TensorDataset(train_z.cuda(), train_y.cuda().squeeze(-1))
train_loader = DataLoader(train_dataset, batch_size=bsz)

for i in tqdm(range(num_epochs), desc="Epoch"):
    minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
    for ix, (batch_x, batch_y) in enumerate(minibatch_iter):
        optimizer.zero_grad()
        output = model(batch_x)
        loss = model.loss(output, batch_y)
        minibatch_iter.set_postfix(loss=loss.item())
        if ix % 100 == 0:
            if verbose:
                print(f'Iteration {ix}, -- loss={loss.item()}')
        loss.backward()
        optimizer.step()

save_model = True
if save_model:
    path_ = 'mdn_trained_models/' + model_name +  '.pt'
    print("saving at", path_)
    torch.save(model.state_dict(), path_)
