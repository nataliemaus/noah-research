import torch
import pandas as pd
import numpy as np
from weighted_retraining.weighted_retraining.chem.chem_model import JTVAE
from weighted_retraining.weighted_retraining.chem.jtnn import Vocab
from weighted_retraining.weighted_retraining.chem.chem_utils import rdkit_quiet, penalized_logP
import pandas as pd
import sys
rdkit_quiet()

# get fine-tuned JTVAE model
path_to_model  = "weighted_retraining/assets/pretrained_models/chem_triplet/chem.ckpt" 
vocab_file = "weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
with open(vocab_file) as f:
    vcb= Vocab([x.strip() for x in f.readlines()])
model: JTVAE = JTVAE.load_from_checkpoint(path_to_model, vocab=vcb)

# Get y_dec (logPs from decoded z_vecs)
def latent_z_to_max_logP(z_vectors, samps_per_z = 40):
    z_tree_vecs = torch.tensor(z_vectors[:,0:28]).float()
    # print("tree vecs shape",z_tree_vecs.shape ) tree vecs shape (218969, 28)
    z_mol_vecs = torch.tensor(z_vectors[:,28:56]).float()
    # print("mols vecs shape",z_mol_vecs.shape ) # mols vecs shape (218969, 28)
    logP_arrays = []
    failures = 0
    for i in range(samps_per_z):
        reconstructed_smiles = model.jtnn_vae.decode(z_tree_vecs, z_mol_vecs, prob_decode = True)
        print("finished decode!")
        logPs = []
        for smile in reconstructed_smiles: 
            logp = penalized_logP(smile)
            if logp is not None:
                logPs.append(logp) 
            else:
                print("uh oh, invalid smile:", smile)
                logPs.append(0)
                failures += 1
                print("num fails", failures)
        logPs = np.array(logPs)
        print(f"shape logPs sampe {i}", logPs.shape) # (num_zs,) ie (3,)
        logP_arrays.append(logPs)
    all_logps = np.vstack(logP_arrays)
    # print("all shape", all_logps.shape) # all shape (40 , num_zs) ie (40,3)
    max_log_ps = np.max(all_logps, axis = 0)
    print("FINAL Y TRAIN SHAPE", max_log_ps.shape) # (num_zs), ie (3,), YAY! 
    return max_log_ps # model.jtnn_vae.decode(z_tree_vecs, z_mol_vecs, prob_decode = True)

path_to_train_z = 'train_data/train_z.csv'
train_z= pd.read_csv(path_to_train_z, header=None).to_numpy().squeeze()
# train_z = train_z[0:3]
print("train z shape", train_z.shape) # train z shape (218969, 56)
max_LOGPS = latent_z_to_max_logP(train_z)
num_zeros = max_LOGPS[np.where(max_LOGPS == 0)].size
print(f"Number of zeros (implying faiulre to generate valid mol): {num_zeros}")
y_df = pd.DataFrame(max_LOGPS)
y_df.to_csv('train_data/train_y_dec.csv', header=None, index = None)


sys.exit()

def smiles_to_z_vecs(smiles_list):
    # smiles_list = [get_good_mol_smile()]
    # Do the encoding
    tree_vecs, mol_vecs = model.jtnn_vae.encode_from_smiles(smiles_list)
    # Random sampling
    z_tree_vecs, _ = model.jtnn_vae.rsample(tree_vecs, model.jtnn_vae.T_mean, model.jtnn_vae.T_var)
    z_mol_vecs, _ = model.jtnn_vae.rsample(mol_vecs, model.jtnn_vae.G_mean, model.jtnn_vae.G_var)
    # print(z_tree_vecs.shape, z_mol_vecs.shape)
    z_vec = torch.cat([z_tree_vecs, z_mol_vecs], axis = 1)
    # print(z_vec.shape) # bsz x 56 :)
    # sys.exit()
    return z_vec

# smiles_list = [get_good_mol_smile(), get_good_mol_smile()]
# smiles_to_z_vecs(smiles_list)
debug = False
path_to_train_data = "weighted_retraining/data/chem/zinc/orig_model/train.txt"
train_x = pd.read_csv(path_to_train_data, sep=" ", header=None).to_numpy().squeeze()
if debug: 
    train_x = train_x[0:39]
# train_x = list(train_x)
print(train_x.shape ) # (218969,) 
print(len(train_x)) # 218969
# killed 

with torch.no_grad():
    train_z_list = []
    bsz = 10000
    step = bsz
    for i in range(0, len(train_x), step):
        start = i
        stop = i + bsz 
        if stop > len(train_x):
            stop = len(train_x)
        print(f"from {start} to {(stop)}")
        train_z = smiles_to_z_vecs(train_x[start:stop])
        print("train z shape", train_z.shape)
        train_z_list.append(train_z)

    train_z = torch.cat(train_z_list)
    print("final train z shape", train_z.shape)
    z_np = train_z.numpy() 
    z_df = pd.DataFrame(z_np)
    z_df.to_csv('train_data/train_z.csv', header=None, index = None)

logPs = []
for smile in train_x: 
    logPs.append(penalized_logP(smile)) 

y_np = np.array(logPs)
print("train_y_shape", y_np.shape)
y_df = pd.DataFrame(y_np)
y_df.to_csv('train_data/train_y.csv', header=None, index = None)
# may have to edit this so y's are 
# from DECODED z vecs like before! 
# do both and compare results? 



sys.exit()
def z_to_logP(zs, num_samples_per_z):
    num_zs = zs.size(-2)
    print("num zs", num_zs)
    # num zs 176074
    # Step 2: Repeat each z 10 times (because we want 10 samples for each z)
    zs = zs.repeat(num_samples_per_z, 1)
    print("repeated zs:", zs.shape)
    # repeated zs: torch.Size([1760740, 128])

    # Step 3: Call model.sample for each "repeated" z
    print("getting smiles")
    smiles_strings = model.sample(zs.size(-2), z=zs)
    print("got smiles") #havn't gotten here yet
    separated_smiles_strings = np.array(smiles_strings).reshape(num_samples_per_z, num_zs)
    print("smiles shape", separated_smiles_strings.shape)
    # smiles shape (10, 176074)

    z_logps = []
    for i, z_strings in enumerate(separated_smiles_strings.T):
        # z_strings are the smiles strings generated for the ith latent code z
        valid_z_strings = remove_invalid(z_strings, canonize=True)
        if len(valid_z_strings) == 0:
            raise RuntimeError('Failed to generate any valid molecules with the current latent code.')
        molecules = [get_mol(smiles_str) for smiles_str in valid_z_strings]
        
        penalized_log_ps = [logP(mol) - SA(mol) for mol in molecules]
        max_logp = np.max(penalized_log_ps)
        z_logps.append(max_logp)
        #print("max log p added: ", max_logp)
    
    z_logps = torch.tensor(z_logps)

    print("done, final y shape:", z_logps.shape)
    
    return z_logps


def save_data(type='train', nsamps=10, save_y=True, save_z=True):
    print(f"using {nsamps} samples per z")
    global new_data
    # get moses smiles data
    if new_data:
        data_path = "Hawei_zinc_data_csv/train.csv"
        print("data:", data_path)
        smiles_data = pd.read_csv(data_path).values.squeeze()
        print("first smile", smiles_data[0])
        print("smiles shape", smiles_data.shape)
    else:
        smiles_data = moses.get_dataset(type)

    # sort smiles strings data by length
    # (sorting essential for string2tensor later)
    sorted_smiles = sorted(smiles_data, key=len, reverse=True)


    # Convert smiles strings to tensors/ numerical (x)
    x = [model.string2tensor(string, device=model.device)
            for string in sorted_smiles]

    # feed x data into model to get latent codes (z)
    if type == "test":
        z, kl_loss = model.forward_encoder(x)
        # .detach removes grad (grad = False)
        z = z.detach()
        print("test z shape", z.shape)
        # test z shape torch.Size([176074, 128])

        if save_z:
            # save test_z to csv
            z_np = z.numpy()
            z_df = pd.DataFrame(z_np)
            z_df.to_csv('test_z_' + str(nsamps) + 'samps.csv', header=None)
        
        y = max_logP(z, num_samples_per_z=nsamps).unsqueeze(-1)
        print("y shape", y.shape)
        # y shape torch.Size([176074, 1])

    elif type == 'train':
        # must use batches for training data to
        # avoid memory allocation error
        if new_data:
            bsz = 40000 # 50,000x4=200,000 -> 218,969 (for 40 samps per z)
            mult = 5
        else:
            bsz = 20000 # 50,000 # 100000
            mult = 79
        #31x50,000->1,550,000 , 20,000x79->1,580,000, /1,584,663
        
        # must split train_z into two files so size
        # is small enough to push to git repo
        zs = []
        zs1 = []
        zs2 = []
        ys = []
        for i in range(mult): 
            print(i*bsz, "to", (i*bsz)+bsz)
            z_for_batch, kl_loss = model.forward_encoder(x[i*bsz:(i*bsz)+bsz])
            ys.append( max_logP(z_for_batch.detach(), num_samples_per_z=nsamps).unsqueeze(-1) )
            if new_data: 
                zs.append(z_for_batch.detach())
            else:
                if i <= mult/2:
                    zs1.append(z_for_batch.detach())
                else:
                    zs2.append(z_for_batch.detach())
        if not new_data:
            z_for_batch, kl_loss = model.forward_encoder(x[bsz*mult:len(x)])
            ys.append( max_logP(z_for_batch.detach(), num_samples_per_z=nsamps).unsqueeze(-1) )
            zs2.append(z_for_batch.detach())

            outsz1 = torch.Tensor(len(zs1), 128)
            z1 = torch.cat(zs1, out=outsz1)
            print("z1 shape", z1.shape)
            # z1 shape torch.Size([800000, 128])
            outsz2 = torch.Tensor(len(zs2), 128)
            z2 = torch.cat(zs2, out=outsz2)
            print("z2 shape", z2.shape)
            # z2 shape torch.Size([784663, 128])
            print("total number of z vecors is", z1.shape[0] + z2.shape[0])
            # total number of z vecors is 1584663
            # total train: torch.Size([1,584,663, 128])
        
        else: 
            outsz = torch.Tensor(len(zs), 128)
            z = torch.cat(zs, out=outsz)
            print("z shape", z.shape)


        if save_z:
            if new_data: 
                # save z to csv
                z_np = z.numpy()
                z_df = pd.DataFrame(z_np)
                z_df.to_csv('train_z_Hawei_' + str(nsamps) + 'samps.csv', header=None)

            else:
                # save z1 to csv
                z1_np = z1.numpy()
                z1_df = pd.DataFrame(z1_np)
                z1_df.to_csv('train_z1_' + str(nsamps) + 'samps.csv', header=None)

                # save z2 to csv
                z2_np = z2.numpy()
                z2_df = pd.DataFrame(z2_np)
                z2_df.to_csv('train_z2_' + str(nsamps) + 'samps.csv', header=None)
        
        y = torch.cat(ys)
        print("final y shape:", y.shape)
        # train: torch.Size([1584663, 1])

    if save_y:
        # save to csv
        y_np = y.numpy()
        y_df = pd.DataFrame(y_np)
        if new_data:
            y_df.to_csv(type + '_y_Hawei_' + str(nsamps) + 'samps.csv', header=None)
        else:
            y_df.to_csv(type + '_y_' + str(nsamps) + 'samps.csv', header=None)



#save_data('test', nsamps = 10)
print("new_data?", new_data)
save_data('train', nsamps = 40)
    # save_data('train', nsamps = 100)
    # quit running because this data only has logPs < 3.5 :( 


# Note: 1 and 2 samples --> always get error: 
# RuntimeError('Failed to generate any valid molecules with the current latent code.')

# cannot save train_y and train_z in same run,
# will get memory allocaiton error
# save_data('train', save_z=False)
# save_data('train', save_y=False)
