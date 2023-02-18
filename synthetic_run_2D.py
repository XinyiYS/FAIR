
import time
import datetime
import numpy as np

from copy import deepcopy

from utils.utils import cwd, set_up_plotting

plt = set_up_plotting()

from synthetic_utils import *

def run_single_experiment(results_dir, index, data_setting='mismatch'):

    start_time = time.time()

    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, DotProduct

    rng = np.random.RandomState(index)

    # xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    # rng = np.random.RandomState(0)
    X = rng.randn(1000, 2)
    d = 2 # dimension of data
    n = 3 # total number of agents

    # setting up number of players, dimension, budget, and the domains for targets and supports
    Ts, Ss = [], []
    mT, mS = 50, 100

    sub_domains = []
    n_sub_domains = 4

    if data_setting == 'mismatch':
        budget = 20

        sub_domain_1 = X[(X[:,0] >= 0) & (X[:, 1] >=0)]
        sub_domain_2 = X[(X[:,0] < 0) & (X[:, 1] >=0)]
        sub_domain_3 = X[(X[:,0] >= 0) & (X[:, 1] <0)]
        sub_domain_4 = X[(X[:,0] < 0) & (X[:, 1] <0)]

        # full_domain = X
        sub_domains = [sub_domain_1, sub_domain_2, sub_domain_3, sub_domain_4]

        for i in range(n):    
            t_subdomain = sub_domains[i]
            target_indices = np.random.choice(t_subdomain.shape[0], size=mT, replace=False)
            T = t_subdomain[target_indices].reshape(-1, d)
            
            s_subdomain = sub_domains[(i+1)%n]
            support_indices = np.random.choice(s_subdomain.shape[0], size=mS, replace=False)
            S = s_subdomain[support_indices].reshape(-1, d)
        #     S = np.random.choice(np.linspace( (i+1)%4*2.5  - 5 ,  (i+1)%4*2.5 - 5 + 2.5, 1000), size=mS, replace=False).reshape(-1,1)
            Ts.append(T)
            Ss.append(S)
    else:
        budget = 5

        sub_domain_1 = X[(X[:,0] <= 2.5) & (X[:, 1] <= 2.5)]
        sub_domain_2 = X[(X[:,0] < 3) & (X[:, 1] <= 2 )]
        sub_domain_3 = X[(X[:,0] <= 2) & (X[:, 1] < 3)]
        sub_domain_4 = X[(X[:,0] < 2.5) & (X[:, 1] <2.5)]

        sub_domains = [sub_domain_1, sub_domain_2, sub_domain_3, sub_domain_4]

        for i in range(n):    
            t_subdomain = sub_domains[0] # same target for all
            target_indices = np.random.choice(t_subdomain.shape[0], size=mT, replace=False)
            T = t_subdomain[target_indices].reshape(-1, d)
            
            s_subdomain = sub_domains[0] # same support for all
            support_indices = np.random.choice(s_subdomain.shape[0], size=mS, replace=False)
            S = s_subdomain[support_indices].reshape(-1, d)
        #     S = np.random.choice(np.linspace( (i+1)%4*2.5  - 5 ,  (i+1)%4*2.5 - 5 + 2.5, 1000), size=mS, replace=False).reshape(-1,1)
            Ts.append(T)
            Ss.append(S)
            

    target_supports = {'Ts':Ts, 'Ss':Ss}


    prior_logdets, prior_covs = [], []

    for T in Ts:
        prior_cov = kernel(T, T)
        _, logdet = np.linalg.slogdet(prior_cov)
        prior_covs.append(prior_cov)
        prior_logdets.append(logdet)

    prior_logdet_joint, prior_cov_joint = None, None
    joint_target = np.unique(np.concatenate(Ts), axis=0)
        
    prior_cov_joint = kernel(joint_target, joint_target)
    _, prior_logdet_joint = np.linalg.slogdet(prior_cov)


    # from favoring the first to uniform
    greedy_obs_1 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[10,5,1], d=d)
    greedy_obs_2 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[10,2,1], d=d)
    greedy_obs_3 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[10,1,1], d=d)
    greedy_obs_4 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, d=d)

    rand_obs = coordinated_random(Ss, Ts, budget, prior_logdets, d=d)
    joint_obs = coordinated_joint(Ss, joint_target, budget, prior_logdet_joint, subset_size=1000, d=d)
    entropy_obs = coordinated_entropy(Ss, Ts, budget, d=d)


    greedy_sum_obs = coordinated_greedy_IG_sum(Ss, Ts, budget, prior_logdets, subset_size=1000, d=d)
    dynamic_beta_obs = coordinated_dynamic_beta(Ss, Ts, budget, prior_logdets, subset_size=1000, beta_coef=0.2, d=d)

    indiv_greedy_obs,  ind_greedy_IG_trails = [], []
    for i in range(n):
        ind_greedy_ob, ind_greedy_IG_trail= individual_greedy(Ss[i], Ts[i], prior_logdets[i], budget, d=d, subset_size=1000)
        indiv_greedy_obs.append(ind_greedy_ob)
        ind_greedy_IG_trails.append(ind_greedy_IG_trail)

    obs = {'greedy_obs_1': greedy_obs_1, 'greedy_obs_2': greedy_obs_2, 'greedy_obs_3': greedy_obs_3,
              'greedy_obs_4': greedy_obs_4,
              'greedy_sum_obs': greedy_sum_obs, 'dynamic_beta_obs':dynamic_beta_obs,
              'joint_obs': joint_obs, 'rand_obs': rand_obs,
              'entropy_obs': entropy_obs, 'indiv_greedy_obs': indiv_greedy_obs}


    greedy_IG_sep_trail_1, greedy_IG_sum_trail_1 = get_IG_trails(greedy_obs_1, Ts, prior_logdets)
    greedy_IG_sep_trail_2, greedy_IG_sum_trail_2 = get_IG_trails(greedy_obs_2, Ts, prior_logdets)
    greedy_IG_sep_trail_3, greedy_IG_sum_trail_3 = get_IG_trails(greedy_obs_3, Ts, prior_logdets)
    greedy_IG_sep_trail_4, greedy_IG_sum_trail_4 = get_IG_trails(greedy_obs_4, Ts, prior_logdets)
    entropy_IG_sep_trail, entropy_IG_sum_trail = get_IG_trails(entropy_obs, Ts, prior_logdets)
    joint_IG_sep_trail, joint_IG_sum_trail = get_IG_trails(joint_obs, Ts, prior_logdets)
    rand_IG_sep_trail, rand_IG_sum_trail = get_IG_trails(rand_obs, Ts, prior_logdets)
    greedy_IGsum_sep_trail, greedy_IGsum_sum_trail = get_IG_trails(greedy_sum_obs, Ts, prior_logdets)
    dynamic_beta_sep_trail, dynamic_beta_sum_trail = get_IG_trails(dynamic_beta_obs, Ts, prior_logdets)

    IG_trail = {'greedy_IG_sep_trail_1':greedy_IG_sep_trail_1, 'greedy_IG_sum_trail_1': greedy_IG_sum_trail_1, 
                   'greedy_IG_sep_trail_2':greedy_IG_sep_trail_2, 'greedy_IG_sum_trail_2': greedy_IG_sum_trail_2,
                   'greedy_IG_sep_trail_3': greedy_IG_sep_trail_3, 'greedy_IG_sum_trail_3': greedy_IG_sum_trail_3,
                   'greedy_IG_sep_trail_4': greedy_IG_sep_trail_4, 'greedy_IG_sum_trail_4': greedy_IG_sum_trail_4,
                   'greedy_IGsum_sep_trail': greedy_IGsum_sep_trail, 'greedy_IGsum_sum_trail': greedy_IGsum_sum_trail,
                   'dynamic_beta_sep_trail': dynamic_beta_sep_trail, 'dynamic_beta_sum_trail': dynamic_beta_sum_trail,
                   
                  'entropy_IG_sep_trail': entropy_IG_sep_trail, 'entropy_IG_sum_trail': entropy_IG_sum_trail,
                   'joint_IG_sep_trail': joint_IG_sep_trail, 'joint_IG_sum_trail': joint_IG_sum_trail,
                   'rand_IG_sep_trail': rand_IG_sep_trail, 'rand_IG_sum_trail': rand_IG_sum_trail,
                   'ind_greedy_IG_trails': ind_greedy_IG_trails,
                  }
    
    end_time = time.time()
    time_str = datetime.datetime.fromtimestamp(start_time).strftime('%m-%d-%H:%M:%S')

    with cwd(results_dir):
        np.savez(f'trial-{index}-setting-{data_setting}-time-{time_str}', Ts=Ts, Ss=Ss, d=d, budget=budget, obs=obs, IG_trail=IG_trail)

    return

import os
from os.path import join as oj

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


if __name__ == '__main__':

    results_dir = oj('results', 'synthetic_2D')
    os.makedirs(results_dir, exist_ok=True)

    n_trials = 5
    print(f'Start running {n_trials} trials for synthetic 2D experiments.')

    n_cores = n_trials
    with ThreadPool(n_cores) as pool:
        input_arguments = [(results_dir,i +1, 'mismatch')  for i in range(n_trials)]
        # input_arguments = [(results_dir,i +1, 'identical')  for i in range(n_trials)]
        output = pool.starmap(run_single_experiment, input_arguments)



