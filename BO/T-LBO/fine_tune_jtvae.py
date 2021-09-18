import torch
from weighted_retraining import weighted_retraining 
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
import pytorch_lightning as pl
from weighted_retraining.weighted_retraining.chem.jtnn import Vocab
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet, penalized_logP
import pandas as pd
from weighted_retraining.weighted_retraining.chem.chem_data import WeightedJTNNDataset
from weighted_retraining.weighted_retraining import utils
rdkit_quiet()
 
# XXXXXX NAHHH 

def test_vae(vae): 
    s1 = "Clc1ccc2ccc3cc4c(c(Cl)cc5c(Cl)cc6ccc7c(Nc8c(Cl)c(Cl)c"
    s2 = "(Nc9c(Cl)c(Cl)c(Cl)c%10ccc%11c%12cc%13cc%14cc%15ccccc%15cc%14c(Cl)c%13c"
    s3 = "(Cl)c%12cc(Cl)c%11c9%10)c(Cl)c8Nc8c(Cl)c(Cl)c(Cl)c9ccc%10c%11cc%12cc%13cc%14cc" 
    s4 = "ccc%14cc%13c(Cl)c%12c(Cl)c%11cc(Cl)c%10c89)c(Cl)c(Cl)c(Cl)c7c6c54)c(Cl)c3c2c1"
    smile = s1 + s2 + s3 + s4 
    smile_out = vae.jtnn_vae.reconstruct_smiles2smiles([smile], prob_decode=False)[0]
    logP_out = penalized_logP(smile_out)
    print("log P of good mol:", logP_out)

path_to_vanilla_model  = "weighted_retraining/assets/pretrained_models/chem_vanilla/chem.ckpt" 
vocab_file = "weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
with open(vocab_file) as f:
    vcb= Vocab([x.strip() for x in f.readlines()])

vae_vanilla: JTVAE = JTVAE.load_from_checkpoint(path_to_vanilla_model, vocab=vcb)

max_epochs = 100
limit_train_batches = 1.0

# Conclusion: the parsing args stuff 
# is a bit of a nightmare and I'd rather 
# just do BO in latent space of model 
# we know pretrained_chem was created by one 
# round of weighted elbo/triplet training (I think... double check)
# and it's dope now so let's run BO in that latent space! 
# ... 
# parser = argparse.ArgumentParser()
# parser.register('type', list, parse_list)
# parser.register('type', dict, parse_dict)
data_weighter = utils.DataWeighter(['weight_type=rank'])
datamodule = WeightedJTNNDataset(args, data_weighter  )
datamodule.setup("fit", n_init_points=args.n_init_bo_points)

path_to_train_data = "weighted_retraining/data/chem/zinc/orig_model/tensors_train/tensors_0000000000-0000050000.pkl"
# that is a list of 50000 mol tree objects 
# and train.txt is smiles strings
# I think the datamodule must do something fancy to get actual numbers... 

# data = pd.read_csv(path_to_train_data, sep=" ", header=None).to_numpy()
data = pd.read_pickle(path_to_train_data)
print(data[0])
print(len(data))
print(data.shape)

# (218969, 1)
# data_df = pd.DataFrame(path_to_train_data, Header = None).values()
data = torch.tensor(data)
train_loader = torch.utils.data.DataLoader(data, batch_size=32)
    # Create trainer
trainer = pl.Trainer(
    gpus=1,
    max_epochs=max_epochs,
    limit_train_batches=limit_train_batches,
    limit_val_batches=1,
    terminate_on_nan=True,
    gradient_clip_val=20.0,  # Model is prone to large gradients
)

# Fit model
trainer.fit(vae_vanilla, train_loader)

vae_vanilla.eval()

print("post training:")
test_vae(vae_vanilla)





# retrain_model(
#         model=vae, datamodule=datamodule, save_dir=retrain_dir,
#         version_str=version, num_epochs=num_epochs, gpu=args.gpu, store_best=args.train_only,
#         best_ckpt_path=args.save_model_path
#     )

# sample_x, sample_y = latent_sampling(
#                         args, model, datamodule, args.samples_per_model,
#                         pbar=sample_pbar
#                     )

#  x_new, y_new = latent_optimization( ...

# datamodule.append_train_data(x_new, y_new)