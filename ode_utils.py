# avoid GPU due to Cholesky decompositions
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import warnings
warnings.filterwarnings("ignore")

# import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np

import tensorflow as tf


from npde_master.utils import gen_data, plot_model, eval_model
from npde_master.npde_helper import build_model, fit_model, save_model, load_model


from copy import deepcopy
from itertools import product

from utils.utils import cwd, set_up_plotting

def generate_ode_data_and_fit_model(index=918273, trajectory_lengths = [20,40,60]):

	sess = tf.compat.v1.InteractiveSession()

	np.random.seed(index) # just for illustration purposes
	x0,t,Y,X,D,f,g = gen_data('vdp', Ny=trajectory_lengths, tbegin=0, tend=8, nstd=0.1)

	npde = build_model(sess, t, Y, model='ode', sf0=1.0, ell0=np.ones(2), W=6, ktype="id")

	prior_sn = npde.sn.eval()
	# print(f'Prior noise variances are {prior_sn}.')

	npde = fit_model(sess, npde, t, Y, num_iter=500, print_every=50, eta=0.02, plot_=False)

	posterior_sn = npde.sn.eval()
	# print(f'Posterior noise variances are {posterior_sn}.')

	return npde, t, Y, prior_sn, posterior_sn



def generate_vdp_data_and_fit_model(index=918273, trajectory_lengths = [35,40,60]):

	sess = tf.compat.v1.InteractiveSession()

	np.random.seed(index)
	x0,t,Y,X,D,f,g = gen_data('vdp-cdiff', Ny=trajectory_lengths, tbegin=0, tend=8, nstd=0.1)


	# in order to constant diffusion, lengthscale of the diffusion process (ellg0) must be initialized to a big number
	npde = build_model(sess, t, Y, model='sde', sf0=1.0, ell0=np.ones(2), W=6, ellg0=[1e5], ktype="id")

	prior_sn = npde.sn.eval()

	npde = fit_model(sess, npde, t, Y, Nw=100, num_iter=1000, print_every=50, eta=0.02, plot_=False)

	posterior_sn = npde.sn.eval()
	# print(f'Posterior noise variances are {posterior_sn}.')
	return npde, t, Y, prior_sn, posterior_sn



def generate_sde_data_and_fit_model(index=918273, trajectory_lengths = [40,30,60]):

	sess = tf.compat.v1.InteractiveSession()

	np.random.seed(index)
	x0,t,Y,X,D,f,g = gen_data('vdp-sdiff', Ny=trajectory_lengths, tbegin=0, tend=8, nstd=0.1)

	npde = build_model(sess, t, Y, model='sde', sf0=1.0, ell0=np.ones(2), W=6, ellg0=[1.0], ktype="id", fix_Z=True)

	prior_sn = npde.sn.eval()

	npde = fit_model(sess, npde, t, Y, Nw=50, num_iter=1000, print_every=50, eta=0.01, plot_=False)


	posterior_sn = npde.sn.eval()
	# print(f'Posterior noise variances are {posterior_sn}.')

	return npde, t, Y, prior_sn, posterior_sn






def IG_sum(de_model, X, Ts, sn, prior_logdets, betas):
    return sum(_IG(de_model, X, T, sn, prior_logdet) *1.0 / beta for T, prior_logdet, beta in zip(Ts, prior_logdets, betas) )

def get_covariance_full(de_model, X, T, sn):
    KTT = de_model.kern._vectorK(T, T)
    
    KTX = de_model.kern._vectorK(T, X)
    
    sigma_matrix = tf.diag(sn).eval()
    I_for_X = tf.eye(int(X.shape[0])).eval()
    tilde_KXX = de_model.kern._vectorK(X, X) + np.kron(sigma_matrix, I_for_X)
    
    return (KTT - KTX @ tf.linalg.inv(tilde_KXX) @ tf.transpose(KTX)).eval()


def _IG(de_model, X, T, sn, prior_logdet, jitter=1e-7):
    post_cov = get_covariance_full(de_model, X, T, sn)
    _, post_logdet = np.linalg.slogdet(post_cov + np.eye(post_cov.shape[0])*jitter)

    return 0.5* (prior_logdet - post_logdet)

    
def entropy_sum(de_model, acquired_obs, Ts, sn, betas=[]):
    if len(betas) == 0:
        betas = np.ones(len(Ts)) / len(Ts)
    else:
        betas = np.asarray(betas)
    
    return sum(_entropy(de_model, acquired_obs, T, sn) *1.0 / beta for T, beta in zip(Ts, betas) )

def _entropy(de_model, acquired_obs, T, sn, exact=False, jitter=1e-7):
    '''
    Note this is not the exact differential entropy formula, 
    instead it ignores some constant terms including the dimension d of data.
    
    For the purpose of maximum entropy search, it is sufficient since we only need the rank and not the 
    absolute value of entropy.
    '''
    post_cov = get_covariance_full(de_model, acquired_obs, T, sn)
    _ , post_logdet = np.linalg.slogdet(post_cov + np.eye(post_cov.shape[0])* jitter )

    if not exact:        
        return post_logdet    
    else:
        d = len(acquired_obs[0])
        differential_entropy = 0.5 * post_logdet + d /2.0 * np.log(2 * np.pi * np.exp(1))
        
        return differential_entropy


def coordinated_greedy(de_model, Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], d=2, sn=[]):
    if len(betas) == 0:
        betas = np.ones(len(Ts)) / len(Ts)
    else:
        assert len(betas) == len(Ts)
        betas = np.asarray(betas) / sum(betas)
        
    acquired_obs = np.asarray([]).reshape(-1, d)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    if len(sn) == 0:
        sn = de_model.sn
    
    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        prev_IG = 0

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))
        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            delta_IG = IG_sum(de_model, temp_obs, Ts, sn, prior_logdets, betas) - prev_IG
            # the weighted sum of difference in IG_k - IG_{k-1} in Equation (2)

            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        prev_IG = IG_sum(de_model, acquired_obs, Ts, sn, prior_logdets, betas)
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs

def coordinated_greedy_index(de_model, Ss, times, Ts, budget, prior_logdets, subset_size=1000, betas=[], d=2, sn=[]):
    if len(betas) == 0:
        betas = np.ones(len(Ts)) / len(Ts)
    else:
        assert len(betas) == len(Ts)
        betas = np.asarray(betas) / sum(betas)
        
    acquired_obs = np.asarray([]).reshape(-1, d)

    ob_times = np.asarray([]).reshape(-1, 1)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    full_indices = [np.arange(len(support)) for support in Supports]
    if len(sn) == 0:
        sn = de_model.sn
    
    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        index_tuple_ = None
        prev_IG = 0

        full_cartesian_indices = np.asarray(list(product(*full_indices)))
        subset_size = min(subset_size, len(full_cartesian_indices))
        subset_cartesian_indices = full_cartesian_indices[np.random.choice(len(full_cartesian_indices), size=subset_size, replace=False)]

        for index_tuple in subset_cartesian_indices:
            obs = [ support[index] for index, support in zip(index_tuple, Supports) ]

            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            delta_IG = IG_sum(de_model, temp_obs, Ts, sn, prior_logdets, betas) - prev_IG
            # the weighted sum of difference in IG_k - IG_{k-1} in Equation (2)

            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs
                index_tuple_ = index_tuple

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)

        prev_IG = IG_sum(de_model, acquired_obs, Ts, sn, prior_logdets, betas)

        for i, index in enumerate(index_tuple_):
            full_indices[i] = full_indices[i][np.where(full_indices[i] != index)]

        selected_times = [time[index] for index, time in zip(index_tuple_, times)]
        ob_times = np.append(ob_times, [selected_times]).reshape(-1, 1)

    return acquired_obs, ob_times



def coordinated_joint(de_model, Ss, joint_target, budget, prior_logdet_joint, subset_size=1000, d=2, sn=[]):
    acquired_obs = np.asarray([]).reshape(-1, d)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    
    if len(sn) == 0:
        sn = de_model.sn
    
    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        prev_IG = 0

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
              
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)
            
            delta_IG = _IG(de_model, temp_obs, joint_target, sn, prior_logdet_joint) - prev_IG
            
            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs
        
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)                
        
        prev_IG = _IG(de_model, acquired_obs, joint_target, sn, prior_logdet_joint)

        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs


def coordinated_joint_index(de_model, Ss, times, joint_target, budget, prior_logdet_joint, subset_size=1000, d=2, sn=[]):
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    full_indices = [np.arange(len(support)) for support in Supports]    
    
    acquired_obs = np.asarray([]).reshape(-1, d)
    ob_times = np.asarray([]).reshape(-1, 1)

    if len(sn) == 0:
        sn = de_model.sn
    
    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        index_tuple_ = None
        prev_IG = 0

        full_cartesian_indices = np.asarray(list(product(*full_indices)))
        subset_size = min(subset_size, len(full_cartesian_indices))
        subset_cartesian_indices = full_cartesian_indices[np.random.choice(len(full_cartesian_indices), size=subset_size, replace=False)]
     
        for index_tuple in subset_cartesian_indices:
            obs = [ support[index] for index, support in zip(index_tuple, Supports) ]
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)
            
            delta_IG = _IG(de_model, temp_obs, joint_target, sn, prior_logdet_joint) - prev_IG
            
            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs
                index_tuple_ = index_tuple

        
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)                
    
        for i, index in enumerate(index_tuple_):
            full_indices[i] = full_indices[i][np.where(full_indices[i] != index)]

        selected_times = [time[index] for index, time in zip(index_tuple_, times)]
        ob_times = np.append(ob_times, [selected_times]).reshape(-1, 1)

    return acquired_obs, ob_times

def coordinated_random(Ss, Ts, budget, prior_logdets, d=2):
    acquired_obs = np.asarray([]).reshape(-1, d)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    
    for _ in range(budget):

        full_cartesian = np.asarray(list(product(*Supports)))
        
        obs_ = full_cartesian[np.random.choice(len(full_cartesian), size=1, replace=False)].squeeze()
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)

        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs

def coordinated_random_index(Ss, times, Ts, budget, prior_logdets, d=2):
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    full_indices = [np.arange(len(support)) for support in Supports]    
    
    acquired_obs = np.asarray([]).reshape(-1, d)
    ob_times = np.asarray([]).reshape(-1, 1)
    
    for _ in range(budget):
        full_cartesian_indices = np.asarray(list(product(*full_indices)))
        
        random_i = np.random.choice(len(full_cartesian_indices), size=1, replace=False)

        index_tuple = full_cartesian_indices[random_i].squeeze()
        obs_ = [support[index] for index, support in zip(index_tuple, Supports) ]
  
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        
        for i, index in enumerate(index_tuple):
            full_indices[i] = full_indices[i][np.where(full_indices[i] != index)]

        selected_times = [time[index] for index, time in zip(index_tuple, times)]
        ob_times = np.append(ob_times, [selected_times]).reshape(-1, 1)
    
    return acquired_obs, ob_times

def coordinated_entropy(de_model, Ss, Ts, budget, subset_size=1000, d=2, sn=[]):
    acquired_obs = np.asarray([]).reshape(-1, d)
    
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    
    if len(sn) == 0:
        sn = de_model.sn

    for _ in range(budget):
        obs_ = None
        
        entropy_max = -float('inf')

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]

        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            curr_entropy = entropy_sum(de_model, temp_obs, Ts, sn)
            
            if curr_entropy > entropy_max:
                entropy_max = curr_entropy
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs


def coordinated_entropy_index(de_model, Ss, times, Ts, budget, subset_size=1000, d=2, sn=[]):
    
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    full_indices = [np.arange(len(support)) for support in Supports]    
    
    acquired_obs = np.asarray([]).reshape(-1, d)
    ob_times = np.asarray([]).reshape(-1, 1)
    
    if len(sn) == 0:
        sn = de_model.sn

    for _ in range(budget):
        obs_ = None
        index_tuple_ = None
        
        entropy_max = -float('inf')

        full_cartesian_indices = np.asarray(list(product(*full_indices)))
        subset_size = min(subset_size, len(full_cartesian_indices))
        subset_cartesian_indices = full_cartesian_indices[np.random.choice(len(full_cartesian_indices), size=subset_size, replace=False)]
     
        for index_tuple in subset_cartesian_indices:
            obs = [ support[index] for index, support in zip(index_tuple, Supports) ]

            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            curr_entropy = entropy_sum(de_model, temp_obs, Ts, sn)
            
            if curr_entropy > entropy_max:
                entropy_max = curr_entropy
                obs_ = obs
                index_tuple_ = index_tuple


#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        
        for i, index in enumerate(index_tuple_):
            full_indices[i] = full_indices[i][np.where(full_indices[i] != index)]

        selected_times = [time[index] for index, time in zip(index_tuple_, times)]
        ob_times = np.append(ob_times, [selected_times]).reshape(-1, 1)

    return acquired_obs, ob_times

def individual_greedy(de_model, S, T, prior_logdet, budget, d=2, subset_size=1000, sn=[]):
        
    acquired_obs = np.asarray([]).reshape(-1, d)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    if len(sn) == 0:
        sn = de_model.sn
    
    Support = (S)
    IG_trail = []
    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        prev_IG = 0
        
        subset_size = min(subset_size, len(Support))
        sub_support = Support[np.random.choice(len(Support), size=subset_size, replace=False)]
        for obs in sub_support:        
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            delta_IG = _IG(de_model, temp_obs, T, sn, prior_logdet) - prev_IG

            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs

        IG_trail.append(delta_IG_max)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        prev_IG = _IG(de_model, acquired_obs, T, sn, prior_logdet)

        Support = S[S!= obs_].reshape(-1,d)

    return acquired_obs, IG_trail

def individual_greedy_index(de_model, S, time, T, prior_logdet, budget, d=2, subset_size=1000, sn=[]):
        
    acquired_obs = np.asarray([]).reshape(-1, d)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    ob_times = np.asarray([]).reshape(-1, 1)

    if len(sn) == 0:
        sn = de_model.sn
    
    Support = (S)
    full_index = np.arange(len(Support))

    IG_trail = []
    for _ in range(budget):
        delta_IG_max = -float('inf')
        obs_ = None
        prev_IG = 0
        index_ = None
        
        subset_size = min(subset_size, len(Support))
        
        subset_indices = np.random.choice(full_index, size=subset_size, replace=False).squeeze()
        
        sub_support = Support[np.random.choice(len(Support), size=subset_size, replace=False)]

        for index in subset_indices:
            obs = Support[index]
#         for obs in sub_support:        
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            delta_IG = _IG(de_model, temp_obs, T, sn, prior_logdet) - prev_IG

            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs
                index_ = index

        IG_trail.append(delta_IG_max)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        prev_IG = _IG(de_model, acquired_obs, T, sn, prior_logdet)

        full_index = full_index[np.where(full_index != index_)]

#         selected_times = [time[index] for index, time in zip(index_tuple, time)]
        ob_times = np.append(ob_times, time[index_]).reshape(-1, 1)    
        

    return acquired_obs, ob_times, IG_trail


def coordinated_greedy_IG_sum(de_model, Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], d=2, sn=[]):
    '''
    Greedily maximizing the total sum of IGs in coordination instead of the marginal to the total sum of IGs 
    as in coordinated_greedy().
    
    This method does NOT satisfy near-optimality guarantee but may help with "cumulative" fairness of overall
    IGs.
    
    '''
    betas = _check_betas(n=len(Ts), betas=betas)

    acquired_obs = np.asarray([]).reshape(-1, d)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    if len(sn) == 0:
        sn = de_model.sn
    
    for _ in range(budget):
        IG_sum_max = -float('inf')
        obs_ = None

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            IG_sum_curr = IG_sum(de_model, temp_obs, Ts, sn, prior_logdets, betas)
            # Directly try to maximize the total sum of IGs
            
            if IG_sum_curr > IG_sum_max:
                IG_sum_max = IG_sum_curr
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
                
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs


def coordinated_greedy_IG_sum_index(de_model, Ss, times, Ts, budget, prior_logdets, subset_size=1000, betas=[], d=2, sn=[]):
    '''
    Greedily maximizing the total sum of IGs in coordination instead of the marginal to the total sum of IGs 
    as in coordinated_greedy().
    
    This method does NOT satisfy near-optimality guarantee but may help with "cumulative" fairness of overall
    IGs.
    
    '''
    betas = _check_betas(n=len(Ts), betas=betas)

    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    full_indices = [np.arange(len(support)) for support in Supports]    
    
    acquired_obs = np.asarray([]).reshape(-1, d)
    ob_times = np.asarray([]).reshape(-1, 1)

    if len(sn) == 0:
        sn = de_model.sn
    
    for _ in range(budget):
        IG_sum_max = -float('inf')
        obs_ = None
        index_tuple_ = None

        full_cartesian_indices = np.asarray(list(product(*full_indices)))
        subset_size = min(subset_size, len(full_cartesian_indices))
        subset_cartesian_indices = full_cartesian_indices[np.random.choice(len(full_cartesian_indices), size=subset_size, replace=False)]
     
        for index_tuple in subset_cartesian_indices:
            obs = [ support[index] for index, support in zip(index_tuple, Supports) ]
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            IG_sum_curr = IG_sum(de_model, temp_obs, Ts, sn, prior_logdets, betas)
            # Directly try to maximize the total sum of IGs
            
            if IG_sum_curr > IG_sum_max:
                IG_sum_max = IG_sum_curr
                obs_ = obs
                index_tuple_ = index_tuple

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
                
        for i, index in enumerate(index_tuple_):
            full_indices[i] = full_indices[i][np.where(full_indices[i] != index)]

        selected_times = [time[index] for index, time in zip(index_tuple_, times)]
        ob_times = np.append(ob_times, [selected_times]).reshape(-1, 1)

    return acquired_obs, ob_times

from scipy.special import softmax
def coordinated_dynamic_beta(de_model, Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], beta_coef=0.5, d=1, sn=[]):
    '''
    The beta values are dynamically updated according to the latest IGs of the agents to help improve
    "cumulative" fairness.

    '''
    betas = _check_betas(n=len(Ts), betas=betas)

    acquired_obs = np.asarray([]).reshape(-1, d)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    if len(sn) == 0:
        sn = de_model.sn
    
    
    for _ in range(budget):
        IG_sum_max = -float('inf')
        obs_ = None

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            IG_sum_curr = IG_sum(de_model, temp_obs, Ts, sn, prior_logdets, betas)
            # Directly try to maximize the total sum of IGs
            
            if IG_sum_curr > IG_sum_max:
                IG_sum_max = IG_sum_curr
                obs_ = obs

        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)

        individual_IGs = [_IG(de_model, acquired_obs, T, sn, prior_logdet) for T, prior_logdet in zip(Ts, prior_logdets) ]
        
        updated_betas = softmax(individual_IGs)
        betas = beta_coef * betas + (1-beta_coef) * updated_betas
    
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs

def coordinated_dynamic_beta_index(de_model, Ss, times, Ts, budget, prior_logdets, subset_size=1000, betas=[], beta_coef=0.5, d=1, sn=[]):
    '''
    The beta values are dynamically updated according to the latest IGs of the agents to help improve
    "cumulative" fairness.

    '''
    betas = _check_betas(n=len(Ts), betas=betas)

    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)
    full_indices = [np.arange(len(support)) for support in Supports]    
    
    acquired_obs = np.asarray([]).reshape(-1, d)
    ob_times = np.asarray([]).reshape(-1, 1)

    if len(sn) == 0:
        sn = de_model.sn
    
    
    for _ in range(budget):
        IG_sum_max = -float('inf')
        obs_ = None
        index_tuple_ = None


        full_cartesian_indices = np.asarray(list(product(*full_indices)))
        subset_size = min(subset_size, len(full_cartesian_indices))
        subset_cartesian_indices = full_cartesian_indices[np.random.choice(len(full_cartesian_indices), size=subset_size, replace=False)]
     
        for index_tuple in subset_cartesian_indices:
            obs = [ support[index] for index, support in zip(index_tuple, Supports) ]
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            IG_sum_curr = IG_sum(de_model, temp_obs, Ts, sn, prior_logdets, betas)
            # Directly try to maximize the total sum of IGs
            
            if IG_sum_curr > IG_sum_max:
                IG_sum_max = IG_sum_curr
                obs_ = obs
                index_tuple_ = index_tuple

        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)

        individual_IGs = [_IG(de_model, acquired_obs, T, sn, prior_logdet) for T, prior_logdet in zip(Ts, prior_logdets) ]
        
        updated_betas = softmax(individual_IGs)
        betas = beta_coef * betas + (1-beta_coef) * updated_betas
    
        for i, index in enumerate(index_tuple_):
            full_indices[i] = full_indices[i][np.where(full_indices[i] != index)]

        selected_times = [time[index] for index, time in zip(index_tuple_, times)]
        ob_times = np.append(ob_times, [selected_times]).reshape(-1, 1)

    return acquired_obs, ob_times

def _check_betas(n, betas=[]):
    if len(betas) == 0:
        betas = np.ones(n) / n
    else:
        assert len(betas) == n
        betas = np.asarray(betas) / sum(betas)
    return betas


def get_IG_trails(de_model, obs, Ts, prior_logdets, betas=[], sn=[]):
    betas = _check_betas(n=len(Ts), betas=betas)
    if len(sn) == 0:
        sn = de_model.sn
    
    IG_sep_trail, IG_sum_trail = [], []

    d = obs.shape[1]
    acquired_obs = np.asarray([]).reshape(-1, d)
    
    for ob in obs:
        curr_IG_sep = []

        acquired_obs = np.append(acquired_obs, [ob]).reshape(-1, d)
        for i, (T, prior_logdet) in enumerate(zip(Ts, prior_logdets)):

            IG_i = _IG(de_model, acquired_obs, T, sn, prior_logdet)
            curr_IG_sep.append(IG_i)
        IG_sep_trail.append(curr_IG_sep)
        
        IG_sum_trail.append(sum(IG*1.0/beta for beta, IG in zip(betas, curr_IG_sep)) )
       
    return IG_sep_trail, IG_sum_trail
