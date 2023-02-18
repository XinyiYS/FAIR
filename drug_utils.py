
import os
import json
import numpy as np
import pandas as pd
from zipfile import ZipFile
from copy import deepcopy

def convert_y_unit(y, from_, to_):
	array_flag = False
	if isinstance(y, (int, float)):
		y = np.array([y])
		array_flag = True
	y = y.astype(float)    
	# basis as nM
	if from_ == 'nM':
		y = y
	elif from_ == 'p':
		y = 10**(-y) / 1e-9

	if to_ == 'p':
		zero_idxs = np.where(y == 0.)[0]
		y[zero_idxs] = 1e-10
		y = -np.log10(y*1e-9)
	elif to_ == 'nM':
		y = y
        
	if array_flag:
		return y[0]
	return y

def load_process_DAVIS(path = './data', binary = False, convert_to_log = True, threshold = 30):
    print('Beginning Processing...')

    if not os.path.exists(path):
        os.makedirs(path)

    saved_path = './data/DAVIS_raw.zip'

    ## IMPORTANT: Download the DAVIS dataset file from 'https://github.com/futianfan/DeepPurpose_Data/blob/main/DAVIS.zip?raw=true'
    ## and place the dataset file in the './data/' folder. Extract the zip file using the code below
    # print('Beginning to extract zip file...')
    # with ZipFile(saved_path, 'r') as zip:
    #     zip.extractall(path = path)

    affinity = pd.read_csv(path + '/DAVIS/affinity.txt', header=None, sep = ' ')

    with open(path + '/DAVIS/target_seq.txt') as f:
        target = json.load(f)

    with open(path + '/DAVIS/SMILES.txt') as f:
        drug = json.load(f)

    target = list(target.values())
    drug = list(drug.values())

    SMILES = []
    Target_seq = []
    y = []

    for i in range(len(drug)):
        for j in range(len(target)):
            SMILES.append(drug[i])
            Target_seq.append(target[j])
            y.append(affinity.values[i, j])

    if binary:
        print('Default binary threshold for the binding affinity scores are 30, you can adjust it by using the "threshold" parameter')
        y = [1 if i else 0 for i in np.array(y) < threshold]
    else:
        if convert_to_log:
            print('Default set to logspace (nM -> p) for easier regression')
            y = convert_y_unit(np.array(y), 'nM', 'p')
        else:
            y = y
    print('Done!')
    return np.array(SMILES), np.array(Target_seq), np.array(y)


def process_data():

    X_drug, X_target, y = load_process_DAVIS('data/', binary=True)
    print('Setting the target site to be the amino acid with sequence "{}"'.format(X_target[1]))
    idx = X_target==X_target[1]
    X_drug, X_target, y = X_drug[idx], X_target[idx], y[idx]

    # Unique X_drug to a specific X_target, y is binary
    idx = np.unique(X_drug, return_index=True)[1]
    X_drug, X_target, y = X_drug[idx], X_target[idx], y[idx]

    return X_drug, X_target, y 

X_drug, X_target, y = process_data()

import numpy as np
import re

class Kernel():
    def __init__(self, kernel = None, y_true = None):
        self.check_size(kernel)
        self.kernel = kernel
        self.y_true = y_true
    
    @classmethod    
    def get_kernel(cls, path):
        kernel = []
        with open (path) as input:
            for line in input:
                # delete blank lines or lines of all 0's
                if re.match(r'^\s*$', line) or float(line.split(' ')[0]) - 0 < 0.00001:
                    continue
                kernel.append(list(map(float,line.strip().split(' '))))
        kernel = np.array(kernel)
        return (kernel)

    @classmethod
    def subseq_helper(cls, smiles, **kwargs):
        lamda = kwargs['lamda'] if 'lamda' in kwargs else None            
        p = kwargs['p'] if 'p' in kwargs else None
        tune = kwargs['p'] if 'tune' in kwargs else None

        if not tune:
            if lamda and p:
                K= gws(smiles, p, lamda)
            elif lamda:
                K = gws(smiles, 4, lamda)
            elif p:
                K = gws(smiles, p, 0.8)
            else:
                K = gws(smiles)
        else:   # tune both parameters
            K = gws(smiles, None, None, True)
            
        return (K)

    @classmethod
    def mismatch_helper(cls, smiles, **kwargs):
        k = kwargs['k'] if 'k' in kwargs else None
        m = kwargs['m'] if 'm' in kwargs else None

        if k and m:
            K = mismatch(smiles, k, m)
        elif k:
            K = mismatch(smiles, k, 1)
        elif m:
            K = mismatch(smiles, 4, m)
        else:
            K = mismatch(smiles)

        return (K)
    
    def check_size(self, K):
        if K.shape[0] != K.shape[1]:
            raise Exception("The Gram matrix doesn't have correct size.")

    @classmethod
    def from_file(cls, path, y_true):
        K = cls.get_kernel(path)
        return cls(K, y_true)
    
    @classmethod
    def from_smi(cls, smiles, y_true, kern_type, **kwargs):

        # string kernel
        if kern_type == 'subsequence':
            K = cls.subseq_helper(smiles, **kwargs)    
            return (cls(K, y_true))            
            
        # mismatch kernel
        if kern_type == 'mismatch':
            K = cls.mismatch_helper(smiles, **kwargs)
            return (cls(K, y_true))

import math
# mismatch kernel
# k for length of compared substrings, m for # of mismatches allowed
def mismatch(smiles, k = 4, m = 1):
    N = len(smiles)
    count_tree = mismatch_count(smiles, k, m)
    gram = mismatch_matrix(count_tree, N)
    return (gram)

def mismatch_count(smiles, k = 5, m = 1):
    # construct the alphabet for smiles representation of mols
    alphabet = set()
    for smile in smiles:
        for c in smile:
            alphabet.add(c)

    # compute the dict for counting characters (l=1)
    N = len(smiles)
    t1 = {i:{} for i in alphabet}
    for x in alphabet:
        for i in range(N):
            for j in range(len(smiles[i])):
                t1[x][(i,j)] = 0 if smiles[i][j] == x else 1
    
    prev = t1
    
    for i in range(1, k - 1):
        cur = {}
        for key in prev:
            for c in alphabet:                
                temp = tuple(list(key) + [c])  # potential substrings
                for key2 in prev[key]:
                    i, j = key2
                    if prev[key][key2] <= m and j <= len(smiles[i]) - 2:                
                        if temp not in cur:
                            cur[temp] = {}
                        count = prev[key][key2] if smiles[i][j+1] == c else prev[key][key2] + 1  # count mismatches
                        if count <= m:
                            cur[temp][(i,j+1)] = count
        prev = cur
                            
    count_tree = {}
    for key in prev:
        for c in alphabet:
            temp = tuple(list(key) + [c])
            for key2 in prev[key]:
                i, j = key2
                if prev[key][key2] <= m and j <= len(smiles[i]) - 2:                
                    if temp not in count_tree:
                        count_tree[temp] = {}
                    if i not in count_tree[temp]:
                        count_tree[temp][i] = {}
                    count = prev[key][key2] if smiles[i][j+1] == c else prev[key][key2] + 1
                    if count <= m:
                        count_tree[temp][i][j+1] = count
    return (count_tree) 

def mismatch_matrix(count_tree, N):
    # compute gram matrix                    
    K = np.zeros((N,N))
    for key1 in count_tree:
        l = count_tree[key1].keys()
        for i in l:
            for j in l:
                K[i, j] += len(count_tree[key1][i].keys()) * len(count_tree[key1][j].keys())
                
    # normalize
    gram = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            gram[i,j] = K[i,j] / math.sqrt(K[i,i] * K[j,j])
            
    return (gram)


# Gap-weighted subsequences kernels
# Input: strings s and t of lengths n and m, length p, weight λ
def gws_kern(mol1, mol2, p, lamda):
    n = len(mol1)
    m = len(mol2)
    DPS = np.zeros((n, m))
    kern = [0]*p
    
    for i in range(n):
        for j in range(m):
            if mol1[i] == mol2[j]:
                DPS[i,j] = lamda**2
        
    kern[0] = np.sum(DPS)
    DP = np.zeros((n-1, m-1))
    for l in range(1,p):
        DP[0,0] = DPS[0,0]
        # boundary values
        for i in range(1,n-1):
            DP[i,0] = DPS[i,0] + lamda * DP[i-1,0]
        for j in range(1,m-1):
            DP[0,j] = DPS[0,j] + lamda * DP[0,j-1]
            
        # update DP
        for i in range(1,n-1):
            for j in range(1,m-1):
                DP[i,j] = DPS[i,j] + lamda * DP[i-1,j] + lamda * DP[i,j-1] + (lamda**2) * DP[i-1,j-1]
            
        # update DPS and kernel value
        for i in range(1,n):
            for j in range(1,m):
                if mol1[i] == mol2[j]:
                    DPS[i,j] = (lamda**2) * DP[i-1,j-1]
                    kern[l] = kern[l] + DPS[i,j]
                    
    return (kern)


def gws_matrix(smiles, p, lamda):
    kern = [[] for _ in range(p)]
    for i in range(len(smiles)):
        temp = gws_kern(smiles[i],smiles[i],p,lamda)
        for l in range(p):
            # compute square root here
            kern[l].append(math.sqrt(temp[l]))
    
    sim = [[] for _ in range(p)]
    for i in range(len(smiles)):
        for l in range(p):
            sim[l].append([])
        for j in range(len(smiles)):
            temp = gws_kern(smiles[i],smiles[j],p,lamda)
            for l in range(p):
                # normalize
                sim[l][i].append(temp[l]/(kern[l][i]*kern[l][j]))
                
    str_kern = np.array(sim)
    return (str_kern)


def gws(smiles, p = 4, lamda = 0.8, tune = False):
    if not tune:
        gram = gws_matrix(smiles, p, lamda)[p-1]
        return (gram)


kernel_name = 'subsequence' ## or 'mismatch'

if kernel_name == 'subsequence':
    subsequence = Kernel.from_smi(X_drug, y, 'subsequence')
    kernel_matrix = subsequence.kernel
elif kernel_name == 'mismatch':
    mismatch = Kernel.from_smi(X_drug, y, 'mismatch')
    kernel_matrix = mismatch.kernel
else:
    raise NotImplementedError()

inputs = X_drug

kernel_dict = dict()
for i in range(len(inputs)):
    kernel_dict[inputs[i]] = i

def kernel_func(x1, x2):
    matrix = np.zeros((len(x1), len(x2)))

    for i in range(len(x1)):
        for j in range(len(x2)):
            matrix[i, j] = kernel_matrix[kernel_dict[x1[i]], kernel_dict[x2[j]]]

    return matrix

kernel = kernel_func



from utils.utils import cwd, set_up_plotting
from itertools import product

plt = set_up_plotting()

def posterior_predictive(X, X_train, Y_train, sigma_y=1e-8):
    K = kernel(X_train, X_train) + np.square(sigma_y) * np.eye(len(X_train))
    K_s = kernel(X_train, X)
    K_ss = kernel(X, X)
    
    K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))

    mu_s = K_s.T @ K_inv @ Y_train
    cov_s = K_ss - K_s.T @ K_inv @ K_s

    return mu_s, cov_s


def posterior_covariance(X, X_train, sigma_y=1e-8):
    K = kernel(X_train, X_train) + np.square(sigma_y) * np.eye(len(X_train))
    K_s = kernel(X_train, X)
    K_ss = kernel(X, X)
    K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))

    cov_s = K_ss - K_s.T @ K_inv @ K_s

    return cov_s

def IG_sum(acquired_obs, Ts, prior_logdets, betas):
    
    return sum(_IG(acquired_obs, T, prior_logdet) *1.0 / beta for T, prior_logdet, beta in zip(Ts, prior_logdets, betas) )

def _IG(acquired_obs, T, prior_logdet):
    post_cov = posterior_covariance(X=T, X_train=acquired_obs)
    _ , post_logdet = np.linalg.slogdet(post_cov)
    return 0.5 * (prior_logdet - post_logdet)

def coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], d=1):
    if len(betas) == 0:
        betas = np.ones(len(Ts)) / len(Ts)
    else:
        assert len(betas) == len(Ts)
        betas = np.asarray(betas) / sum(betas)
        
    acquired_obs = np.asarray([])
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        prev_IG = 0

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs])

            delta_IG = IG_sum(temp_obs, Ts, prior_logdets, betas) - prev_IG
            # the weighted sum of difference in IG_k - IG_{k-1} in Equation (2)
            
            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_])
        
        prev_IG = IG_sum(acquired_obs, Ts, prior_logdets, betas)
        
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            remove_idx = np.argwhere((S == ob).all(-1))
            keep_idx = np.setdiff1d(np.arange(len(S)), remove_idx)
            Supports[i] = S[keep_idx]

    return acquired_obs

def coordinated_joint(Ss, joint_target, budget, prior_logdet_joint, subset_size=1000, d=1):
    acquired_obs = np.asarray([])
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    
    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        prev_IG = 0

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
              
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs])
            
            delta_IG = _IG(temp_obs, joint_target, prior_logdet_joint) - prev_IG
            
            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs
        
        acquired_obs = np.append(acquired_obs, [obs_])               
        
        prev_IG = _IG(acquired_obs, joint_target, prior_logdet_joint)

        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            remove_idx = np.argwhere((S == ob).all(-1))
            keep_idx = np.setdiff1d(np.arange(len(S)), remove_idx)
            Supports[i] = S[keep_idx]

    return acquired_obs


def coordinated_random(Ss, Ts, budget, prior_logdets, d=1):
    acquired_obs = np.asarray([])
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    
    for _ in range(budget):

        full_cartesian = np.asarray(list(product(*Supports)))
        
        obs_ = full_cartesian[np.random.choice(len(full_cartesian), size=1, replace=False)].squeeze()
        acquired_obs = np.append(acquired_obs, [obs_])

        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            remove_idx = np.argwhere((S == ob).all(-1))
            keep_idx = np.setdiff1d(np.arange(len(S)), remove_idx)
            Supports[i] = S[keep_idx]

    return acquired_obs

def entropy_sum(acquired_obs, Ts, betas=[]):
    if len(betas) == 0:
        betas = np.ones(len(Ts)) / len(Ts)
    else:
        betas = np.asarray(betas)
    
    return sum(_entropy(acquired_obs, T) *1.0 / beta for T, beta in zip(Ts, betas) )

def _entropy(acquired_obs, T):
    '''
    Note this is not the exact differential entropy formula, 
    instead it ignores some constant terms including the dimension d of data.
    
    For the purpose of maximum entropy search, it is sufficient since we only need the rank and not the 
    absolute value of entropy.
    '''
    post_cov = posterior_covariance(T, acquired_obs)
    _ , post_logdet = np.linalg.slogdet(post_cov)
    
    return post_logdet


def coordinated_entropy(Ss, Ts, budget, subset_size=1000, d=1):
    acquired_obs = np.asarray([])
    
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    for _ in range(budget):
        obs_ = None
        
        entropy_max = -float('inf')

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]

        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs])

            curr_entropy = entropy_sum(temp_obs, Ts)
            
            if curr_entropy > entropy_max:
                entropy_max = curr_entropy
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_])
        
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            remove_idx = np.argwhere((S == ob).all(-1))
            keep_idx = np.setdiff1d(np.arange(len(S)), remove_idx)
            Supports[i] = S[keep_idx]

    return acquired_obs

def individual_greedy(S, T, prior_logdet, budget, d=1, subset_size=1000):
        
    acquired_obs = np.asarray([])
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Support = (S)
    IG_trail = []
    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        prev_IG = 0
        
        subset_size = min(subset_size, len(Support))
        sub_support = Support[np.random.choice(len(Support), size=subset_size, replace=False)]
        for obs in sub_support:        
            temp_obs = np.append(acquired_obs, [obs])

            delta_IG = _IG(temp_obs, T, prior_logdet) - prev_IG

            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs

        IG_trail.append(delta_IG_max)
        acquired_obs = np.append(acquired_obs, [obs_])
        prev_IG = _IG(acquired_obs, T, prior_logdet)

        # Support = S[S!= obs_].reshape(-1,d)
        remove_idx = np.argwhere((S == obs_).all(-1))
        keep_idx = np.setdiff1d(np.arange(len(S)), remove_idx)
        Support = S[keep_idx]

    return acquired_obs, IG_trail

def get_IG_trails(obs, Ts, prior_logdets, betas=[]):
    if len(betas) == 0:
        betas = np.ones(len(Ts)) / len(Ts)
    else:
        assert len(betas) == len(Ts)
        betas = np.asarray(betas) / sum(betas)
        
    IG_sep_trail, IG_sum_trail = [], []

    acquired_obs = np.asarray([])
    
    for ob in obs:
        curr_IG_sep = []

        acquired_obs = np.append(acquired_obs, [ob])
        for i, (T, prior_logdet) in enumerate(zip(Ts, prior_logdets)):

            IG_i = _IG(acquired_obs, T, prior_logdet)
            curr_IG_sep.append(IG_i)
        IG_sep_trail.append(curr_IG_sep)
        
        IG_sum_trail.append(sum(IG*1.0/beta for beta, IG in zip(betas, curr_IG_sep)) )

    return IG_sep_trail, IG_sum_trail

def coordinated_greedy_IG_sum(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], d=1):
    '''
    Greedily maximizing the total sum of IGs in coordination instead of the marginal to the total sum of IGs 
    as in coordinated_greedy().
    
    This method does NOT satisfy near-optimality guarantee but may help with "cumulative" fairness of overall
    IGs.
    
    '''
    betas = _check_betas(n=len(Ts), betas=betas)

    acquired_obs = np.asarray([])
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    for _ in range(budget):
        IG_sum_max = -float('inf')
        obs_ = None

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs])

            IG_sum_curr = IG_sum(temp_obs, Ts, prior_logdets, betas)
            # Directly try to maximize the total sum of IGs
            
            if IG_sum_curr > IG_sum_max:
                IG_sum_max = IG_sum_curr
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_])
                
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob]

    return acquired_obs

from scipy.special import softmax
def coordinated_dynamic_beta(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], beta_coef=0.5, d=1):
    '''
    The beta values are dynamically updated according to the latest IGs of the agents to help improve
    "cumulative" fairness.

    
    '''
    betas = _check_betas(n=len(Ts), betas=betas)

    acquired_obs = np.asarray([])
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    for _ in range(budget):
        IG_sum_max = -float('inf')
        obs_ = None

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs])

            IG_sum_curr = IG_sum(temp_obs, Ts, prior_logdets, betas)
            # Directly try to maximize the total sum of IGs
            
            if IG_sum_curr > IG_sum_max:
                IG_sum_max = IG_sum_curr
                obs_ = obs

        acquired_obs = np.append(acquired_obs, [obs_])

        individual_IGs = [_IG(acquired_obs, T, prior_logdet) for T, prior_logdet in zip(Ts, prior_logdets) ]
        
        updated_betas = softmax(individual_IGs)
        betas = beta_coef * betas + (1-beta_coef) * updated_betas
    
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob]

    return acquired_obs

def _check_betas(n, betas=[]):
    if len(betas) == 0:
        betas = np.ones(n) / n
    else:
        assert len(betas) == n
        betas = np.asarray(betas) / sum(betas)
    return betas
