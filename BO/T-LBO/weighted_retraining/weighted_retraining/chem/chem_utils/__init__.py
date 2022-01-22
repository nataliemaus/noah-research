""" Contains many chem utils codes """
import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen, QED
import networkx as nx
from rdkit.Chem import rdmolops

from csv import writer 

guacamol = False
if guacamol:
    from guacamol import standard_benchmarks
    # My imports
    from weighted_retraining.weighted_retraining.chem.chem_utils.SA_Score import sascorer

    med1 = standard_benchmarks.median_camphor_menthol().objective.score #'Median molecules 1'
    med2 = standard_benchmarks.median_tadalafil_sildenafil().objective.score #'Median molecules 2',
    pdop = standard_benchmarks.perindopril_rings().objective.score # 'Perindopril MPO',
    osmb = standard_benchmarks.hard_osimertinib().objective.score  # 'Osimertinib MPO',
    adip = standard_benchmarks.amlodipine_rings().objective.score  # 'Amlodipine MPO'
    siga = standard_benchmarks.sitagliptin_replacement().objective.score #'Sitagliptin MPO'
    zale = standard_benchmarks.zaleplon_with_other_formula().objective.score # 'Zaleplon MPO'
    valt = standard_benchmarks.valsartan_smarts().objective.score  #'Valsartan SMARTS',
    dhop = standard_benchmarks.decoration_hop().objective.score # 'Deco Hop'
    shop = standard_benchmarks.scaffold_hop().objective.score # Scaffold Hop'
    rano= standard_benchmarks.ranolazine_mpo().objective.score #'Ranolazine MPO'

    guacamol_objs = {"med1":med1,"pdop":pdop, "adip":adip, "rano":rano, "osmb":osmb,
            "siga":siga, "zale":zale, "valt":valt, "med2":med2,"dhop":dhop, "shop":shop}
        
    valid_keys = ['med1', 'pdop', 'adip', 'rano', 'osmb', 'siga', 
                            'zale', 'valt', 'med2', 'dhop', 'shop']

# TAKES IN SINGLE SMILES, GIVES SCORE! 

# tmux attach -t dock_huawei

def hardcoded_smile_to_guacamol(smile):
    obj_func_key = 'rano'
    func = guacamol_objs[obj_func_key]
    score = func(smile)
    # record score in csv file!! 
    List=[str(score)]
    with open('true_' + obj_func_key + '_results_v2.csv', 'a') as f_object: 
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()
    return score

tdc_dock = True
if tdc_dock:
    from tdc import Oracle
    protien_name = "3pbl_docking"
    tdc_oracle = Oracle(name=protien_name) 
    import time
    from multiprocessing.pool import ThreadPool 
    # import wandb
    # tracker = wandb.init(entity="nmaus", project="huawei_docking", config={"task":"dock_3pbl"})


def smile_is_valid_mol(smile):
    if smile is None or len(smile)==0:
        return False
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return False
    return True

# conda install -c metric-learning pytorch-metric-learning

# chmod u+x ./weighted_retraining/scripts/robust_opt/robust_opt_chem.sh
# ./weighted_retraining/scripts/robust_opt/robust_opt_chem.sh

def smiles_to_dock_3pbl(smiles_str):
    score = smile_to_tdc_docking_score(smiles_str, tdc_oracle=tdc_oracle)
    if score is None:
        score = 0
    else:
        score = score * -1 # turn min to max prob! 
    # record score in csv file!! 
    # tracker.log({"score":score})
    List=[str(score)]
    with open('true_dock_3pbl_results_run2.csv', 'a') as f_object: 
        writer_object = writer(f_object)
        writer_object.writerow(List)
        f_object.close()
    return score

def smile_to_tdc_docking_score(smiles_str, tdc_oracle, max_smile_len=1000, timeout=800):
    # goal of function:
    #          return docking score (score = tdc_oracle(smiles_str) ) iff it can be computed within timout seconds
    #           otherwisse, return None
    if not smile_is_valid_mol(smiles_str):
        return None
    smiles_str = Chem.CanonSmiles(smiles_str)
    if len(smiles_str) > max_smile_len:
        return None
    start = time.time()

    def get_the_score(smiles_str):
        docking_score = tdc_oracle(smiles_str)
        return docking_score

    pool = ThreadPool(1)

    async_result = pool.apply_async(get_the_score, (smiles_str,))
    from multiprocessing.context import TimeoutError
    try:
        ret_value = async_result.get(timeout=timeout)
    except Exception as e:
        print("Error occurred getting score from smiles str::",smiles_str,  e)
        # print('TimeoutError encountered getting docking score for smiles_str:', smiles_str)
        ret_value = None

    print(f"getting docking score: {ret_value} from protien took {time.time()-start} seconds")
    return ret_value

def get_guacamol_score_func(obj_func_key):
    return guacamol_objs[obj_func_key]


def smile_to_guacamole_score_func(obj_func_key, smile):
    if smile is None or len(smile)==0:
        return None
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    func = guacamol_objs[obj_func_key]
    score = func(smile)
    if score is None:
        return None
    if score < 0:
        return None
    return score 

# Make rdkit be quiet
def rdkit_quiet():
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)


def get_mol(smiles_or_mol):
    '''                                                                                                                                       
    Loads SMILES/molecule into RDKit's object                                   
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def standardize_smiles(smiles):
    """ Get standard smiles without stereo information """
    mol = get_mol(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def penalized_logP(smiles: str, min_score=-float("inf")) -> float:
    """ calculate penalized logP for a given smiles string """
    if smiles is None:
        print("not a smile")
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("invalid mol")
        return None
    logp = Crippen.MolLogP(mol)
    print("log P raw", logp)
    sa = SA(mol)
    print("sa raw", sa)

    # Calculate cycle score
    cycle_length = _cycle_score(mol)
    print("cycle score raw", cycle_length)

    """
    Calculate final adjusted score.
    These magic numbers are the empirical means and
    std devs of the dataset.

    I agree this is a weird way to calculate a score...
    but this is what previous papers did!
    """
    score = (
            (logp - 2.45777691) / 1.43341767
            + (-sa + 3.05352042) / 0.83460587
            + (-cycle_length - -0.04861121) / 0.28746695
    )
    return max(score, min_score)


def SA(mol):
    return sascorer.calculateScore(mol)


def _cycle_score(mol):
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def QED_score(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return QED.qed(mol)
