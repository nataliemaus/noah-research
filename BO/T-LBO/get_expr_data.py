# test stuff 
import weighted_retraining.weighted_retraining.expr.eq_grammar as grammar
from weighted_retraining.weighted_retraining.expr.expr_model_pt import EquationVaeTorch
import numpy as np
import torch

import h5py
from numpy import exp, sin
import sys
import nltk

data_info = grammar.gram.split('\n')

# datamodule = WeightedExprDataset(args, DataWeighter(args))

model_path = "results_expr/vanilla_model/best.ckpt"
vae: EquationVaeTorch = EquationVaeTorch.load_from_checkpoint(model_path, charset_length=len(data_info)) 


def pop_or_nothing(S):
    try:
        return S.pop()
    except:
        return 'Nothing'


def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''

_productions = grammar.GCFG.productions()
_lhs_map = {}
for ix, lhs in enumerate(grammar.lhs_list):
    _lhs_map[lhs] = ix

def _sample_using_masks(unmasked):
    """ Samples a one-hot vector, masking at each timestep.
        This is an implementation of Algorithm ? in the paper. """
    eps = 1e-100
    X_hat = np.zeros_like(unmasked)

    # Create a stack for each input in the batch
    S = np.empty((unmasked.shape[0],), dtype=object)
    for ix in range(S.shape[0]):
        S[ix] = [str(grammar.start_index)]

    # Loop over time axis, sampling values and updating masks
    for t in range(unmasked.shape[1]):
        next_nonterminal = [_lhs_map[pop_or_nothing(a)] for a in S]
        mask = grammar.masks[next_nonterminal]
        masked_output = np.exp(unmasked[:, t, :]) * mask + eps
        sampled_output = np.argmax(np.random.gumbel(
            size=masked_output.shape) + np.log(masked_output), axis=-1)
        X_hat[np.arange(unmasked.shape[0]), t, sampled_output] = 1.0

        # Identify non-terminals in RHS of selected production, and
        # push them onto the stack in reverse order
        rhs = [
            [a for a in _productions[i].rhs() if (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None')]
            for i in sampled_output]
        for ix in range(S.shape[0]):
            S[ix].extend(list(map(str, rhs[ix]))[::-1])
    return X_hat  # , ln_p


# NOTE, THIS IS THE TRUE FUNCTION, THEY MESSED UP IN DESCRIPTION IN PAPER APPENDIX!
# confirmed here: http://proceedings.mlr.press/v70/kusner17a/kusner17a.pdf
def score_function(inputs, target_eq='1 / 3 + x + sin( x * x )', worst=7.0) -> np.ndarray:
    """ compute equation scores of given inputs """
    # define inputs and outputs of ground truth target expression
    x = np.linspace(-10, 10, 1000)
    try:
        yT = np.array(eval(target_eq))
    except NameError as e:
        print(target_eq)
        raise e
    scores = []
    for inp in inputs:
        try:
            y_pred = np.array(eval(inp))
            scores.append(np.minimum(worst, np.log(1 + np.mean((y_pred - yT) ** 2))))
        except:
            scores.append(worst)
    return np.array(scores)


def latent_z_to_scores(latent_codes):
    unmasked = vae.decode_deterministic(torch.tensor(latent_codes).float()).cpu().detach().numpy()
    # print(unmasked.shape) # (40008, 15, 12)

    X_hat = _sample_using_masks(unmasked)
    # Convert from one-hot to sequence of production rules
    prod_seq = [[_productions[X_hat[index, t].argmax()]
                    for t in range(X_hat.shape[1])]
                for index in range(X_hat.shape[0])]
    eqs_dec = [prods_to_eq(prods) for prods in prod_seq]

    # print(eqs_dec[0:5]) # decoded equations  '1/2+sin(2)+exp(3)', 'x+1+sin(1)+(3+1)'
    # print(len(eqs_dec)) # 40008 

    data_scores = score_function(eqs_dec)
    print("data scores", data_scores.shape)
    print(data_scores[0:10])

    return data_scores


def get_train_data():
    dataset_path="weighted_retraining/data/expr/expr_P65_5_0.npz"
    property_key='scores'
    with np.load(dataset_path) as npz:
        one_hot_encs = npz["data"]
        scores = npz[property_key]
        # all_exprs = npz[self.expr_key]
    print("one_hot_encs", one_hot_encs.shape) # (40008, 15, 12) 
    print("all properites", scores.shape) # (40008,)

    input_ = torch.tensor(one_hot_encs).float()
    # print("one hot input shape", input_.shape) #  torch.Size([40008, 15, 12])

    latent_codes = vae.encode_to_params(input_)[0].cpu().detach().numpy()
    # print("eoncded latent shape", latent_codes.shape) # (40008, 25) 
    dec_scores = latent_z_to_scores(latent_codes)

    return latent_codes, dec_scores


def get_og_data():
    # get equation strings
    data_dir ="weighted_retraining/assets/data/expr/" # equation2_15_dataset.txt"
    fname = 'equation2_15_dataset.txt'
    with open(data_dir + fname) as f:
        eqs = f.readlines()
    for i in range(len(eqs)):
        eqs[i] = eqs[i].strip().replace(' ', '')

    print("eqs:")
    print(eqs[0:5])
    print("num eqs", len(eqs)) # num eqs 100000

    # data enc (get 1 hot encoded version of data )
    fname = 'eq2_grammar_dataset.h5'
    h5f = h5py.File(data_dir + fname, 'r')
    one_hot = h5f['data'][:]
    h5f.close()
    print("one hot:")
    print(one_hot.shape)  # (100000, 15, 12) (eqch 1-hot encoding is 15x12!! )
    print("")

    data_scores = score_function(eqs)
    print("data scores", data_scores.shape)
    print(data_scores[0:5])
    # WANT TO MINIMIZE SCORE!! 

# latent_codes, dec_scores = get_train_data()
# print(latent_codes.shape, dec_scores.shape) # (40008, 25) (40008,) 
