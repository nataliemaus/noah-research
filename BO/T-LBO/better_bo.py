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

path_to_model  = "weighted_retraining/assets/pretrained_models/chem_triplet/chem.ckpt" 
vocab_file = "weighted_retraining/data/chem/zinc/orig_model/vocab.txt"
with open(vocab_file) as f:
    vcb= Vocab([x.strip() for x in f.readlines()])

vae: JTVAE = JTVAE.load_from_checkpoint(path_to_model, vocab=vcb)

# ... now see bayesopt repo to implement 
# step 1: need to create z's parid w/ y's dataset to train MDN on! 
# starting that now

