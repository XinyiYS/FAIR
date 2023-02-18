
import time
import datetime
import numpy as np
import os
from os.path import join as oj
from copy import deepcopy

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


from drug_utils import *

def run_single_experiment(results_dir, index, X_drug):
    start_time = time.time()

    np.random.seed(index)


    n = 3
    d = 0
    exclude = ['Cl', 'F', 'CS']
    Ts, Ss = [], []

    for i_agent in range(n):
        rows_contain_exclude = []
        for i in range(len(X_drug)):
            if exclude[i_agent] in X_drug[i]:
                rows_contain_exclude.append(i)
        rows_idx = np.setdiff1d(np.arange(len(X_drug)), rows_contain_exclude)
        Ss.append(X_drug[rows_idx])
        Ts.append(X_drug[rows_contain_exclude])


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

    # Might take 10 minutes for budget == 10
    budget = 7

    # Agent 1 from being altruistic to more increasingly greedy 
    # greedy_obs_1 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[10,1,1], d=0)
    # greedy_obs_2 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[10,2,1], d=0)
    # greedy_obs_3 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[10,5,1], d=0)
    # greedy_obs_4 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[10,10,1], d=0)
    greedy_obs_1 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[19,1,10], d=0)
    greedy_obs_2 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[18,2,10], d=0)
    greedy_obs_3 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[16,5,10], d=0)
    greedy_obs_4 = coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[10,10,10], d=0)

    greedy_IG_sep_trail_1, greedy_IG_sum_trail_1 = get_IG_trails(greedy_obs_1, Ts, prior_logdets)
    greedy_IG_sep_trail_2, greedy_IG_sum_trail_2 = get_IG_trails(greedy_obs_2, Ts, prior_logdets)
    greedy_IG_sep_trail_3, greedy_IG_sum_trail_3 = get_IG_trails(greedy_obs_3, Ts, prior_logdets)
    greedy_IG_sep_trail_4, greedy_IG_sum_trail_4 = get_IG_trails(greedy_obs_4, Ts, prior_logdets)

    rand_obs = coordinated_random(Ss, Ts, budget, prior_logdets, d=0)
    joint_obs = coordinated_joint(Ss, joint_target, budget, prior_logdet_joint, subset_size=1000, d=0)
    entropy_obs = coordinated_entropy(Ss, Ts, budget, d=0)

    entropy_IG_sep_trail, entropy_IG_sum_trail = get_IG_trails(entropy_obs, Ts, prior_logdets)
    joint_IG_sep_trail, joint_IG_sum_trail = get_IG_trails(joint_obs, Ts, prior_logdets)
    rand_IG_sep_trail, rand_IG_sum_trail = get_IG_trails(rand_obs, Ts, prior_logdets)

    greedy_sum_obs = coordinated_greedy_IG_sum(Ss, Ts, budget, prior_logdets, subset_size=1000, d=0)
    dynamic_beta_obs = coordinated_dynamic_beta(Ss, Ts, budget, prior_logdets, subset_size=1000, beta_coef=0.2, d=0)

    greedy_IGsum_sep_trail, greedy_IGsum_sum_trail = get_IG_trails(greedy_sum_obs, Ts, prior_logdets)
    dynamic_beta_sep_trail, dynamic_beta_sum_trail = get_IG_trails(dynamic_beta_obs, Ts, prior_logdets)

    indiv_greedy_obs,  ind_greedy_IG_trails = [], []
    for i in range(n):
        ind_greedy_ob, ind_greedy_IG_trail= individual_greedy(Ss[i], Ts[i], prior_logdets[i], budget, d=0, subset_size=1000)
        indiv_greedy_obs.append(ind_greedy_ob)
        ind_greedy_IG_trails.append(ind_greedy_IG_trail)

    obs = {'greedy_obs_1': greedy_obs_1, 'greedy_obs_2': greedy_obs_2, 'greedy_obs_3': greedy_obs_3,
              'greedy_obs_4': greedy_obs_4, 
              'greedy_sum_obs': greedy_sum_obs, 'dynamic_beta_obs':dynamic_beta_obs,
              'joint_obs': joint_obs, 'rand_obs': rand_obs, 'entropy_obs': entropy_obs, 
              'indiv_greedy_obs': indiv_greedy_obs}


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
      np.savez(f'trial-{index}-time-{time_str}', Ts=Ts, Ss=Ss, d=d, budget=budget, obs=obs, IG_trail=IG_trail)

    return


import os
from os.path import join as oj
from copy import deepcopy

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


if __name__ == '__main__':

    results_dir = oj('results', 'drug')


    os.makedirs(results_dir, exist_ok=True)
    n_trials = 5
    print(f'Start running {n_trials} trials for drug discovery experiments.')

    # X_drug, X_target, y  = process_data()

    n_cores = n_trials
    with ThreadPool(n_cores) as pool:
        input_arguments = [(results_dir,i +1,  deepcopy(X_drug))  for i in range(n_trials)]
        output = pool.starmap(run_single_experiment, input_arguments)