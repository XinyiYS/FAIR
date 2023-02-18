import time
import datetime
import numpy as np


from ode_utils import *

jitter = 1e-7

def run_single_experiment(results_dir, index):

    start_time = time.time()

    np.random.seed(index)
    de_model, t, Y, prior_sn, posterior_sn = generate_sde_data_and_fit_model(index)

    d = 2 # dimension of data
    n = 3 # total number of agents

    budget = 10

    # select the index of the generated trajectories
    trajectory_index = 2

    trajectory = Y[trajectory_index]
    time_trajectory = t[trajectory_index]

    trajectory_length = len(trajectory)
    assert len(time_trajectory) == trajectory_length, "Check the trajectory index or the Y,t."

    Ts, Ss = [], []
    mt, ms = int(trajectory_length // 4.5), trajectory_length // 3
    times = []

    for i in range(n):

        S = trajectory[i*ms:(i+1)*ms]
        Ss.append(S)
        
        individual_time = time_trajectory[i*ms:(i+1)*ms]
        times.append(individual_time)
        
        shifted_i = (i+1) % (n)    
        
        T = trajectory[shifted_i*mt:(shifted_i+1)*mt]
        Ts.append(T)

    # plt.figure(figsize=(8, 6))
    # for S in Ss:    
    #     plt.scatter(S[:,0], S[:,1])
    # plt.title('Supports')
        
    # plt.figure(figsize=(8, 6))
    # for T in Ts:    
    #     plt.scatter(T[:,0], T[:,1])
    # plt.title('Targets')

    prior_logdets, prior_covs = [], []

    for T in Ts:
        prior_cov = get_covariance_full(de_model, T, T, prior_sn)
        _, logdet = np.linalg.slogdet(prior_cov + np.eye(prior_cov.shape[0]) * jitter )
        prior_covs.append(prior_cov)
        prior_logdets.append(logdet)
    

    joint_target = np.unique(np.concatenate(Ts), axis=0)
        
    prior_cov_joint = get_covariance_full(de_model, joint_target, joint_target, prior_sn)
    _, prior_logdet_joint = np.linalg.slogdet(prior_cov+ np.eye(prior_cov.shape[0]) * jitter)


    # from favoring the first to uniform
    greedy_obs_1 = coordinated_greedy(de_model, Ss, Ts, budget, prior_logdets, 
                                      subset_size=5, betas=[10,5,1], d=d, sn=posterior_sn)
    greedy_obs_2 = coordinated_greedy(de_model, Ss, Ts, budget, prior_logdets, 
                                      subset_size=5, betas=[10,2,1], d=d, sn=posterior_sn)
    greedy_obs_3 = coordinated_greedy(de_model, Ss, Ts, budget, prior_logdets, 
                                      subset_size=5, betas=[10,1,1], d=d, sn=posterior_sn)
    greedy_obs_4 = coordinated_greedy(de_model, Ss, Ts, budget, prior_logdets, 
                                      subset_size=5, d=d, sn=posterior_sn)

    joint_obs = coordinated_joint(de_model, Ss, joint_target, budget, prior_logdet_joint, subset_size=5, d=d, sn=posterior_sn)

    rand_obs = coordinated_random(Ss, Ts, budget, prior_logdets, d=d)


    entropy_obs = coordinated_entropy(de_model, Ss, Ts, budget, subset_size=5, d=d, sn=posterior_sn)

    greedy_sum_obs = coordinated_greedy_IG_sum(de_model, Ss, Ts, budget, prior_logdets, subset_size=5, betas=[], d=d, sn=posterior_sn)

    dynamic_beta_obs = coordinated_dynamic_beta(de_model, Ss, Ts, budget, prior_logdets, subset_size=5, betas=[], beta_coef=0.2, d=d, sn=posterior_sn)

    indiv_greedy_obs,  ind_greedy_IG_trails = [], []
    for i in range(n):
        ind_greedy_ob, ind_greedy_IG_trail= individual_greedy(de_model, Ss[i], Ts[i], prior_logdets[i], 
                                                              budget, d=d, subset_size=5, sn=posterior_sn)
        indiv_greedy_obs.append(ind_greedy_ob)
        ind_greedy_IG_trails.append(ind_greedy_IG_trail)


    obs = {'greedy_obs_1': greedy_obs_1, 'greedy_obs_2': greedy_obs_2, 'greedy_obs_3': greedy_obs_3,
              'greedy_obs_4': greedy_obs_4, 
              'greedy_sum_obs': greedy_sum_obs, 'dynamic_beta_obs':dynamic_beta_obs,
              'joint_obs': joint_obs, 'rand_obs': rand_obs, 'entropy_obs': entropy_obs, 
              'indiv_greedy_obs': indiv_greedy_obs}



    greedy_IG_sep_trail_1, greedy_IG_sum_trail_1 = get_IG_trails(de_model, greedy_obs_1, Ts, prior_logdets, sn=posterior_sn )
    greedy_IG_sep_trail_2, greedy_IG_sum_trail_2 = get_IG_trails(de_model, greedy_obs_2, Ts, prior_logdets, sn=posterior_sn)
    greedy_IG_sep_trail_3, greedy_IG_sum_trail_3 = get_IG_trails(de_model, greedy_obs_3, Ts, prior_logdets, sn=posterior_sn)
    greedy_IG_sep_trail_4, greedy_IG_sum_trail_4 = get_IG_trails(de_model, greedy_obs_4, Ts, prior_logdets, sn=posterior_sn)

    entropy_IG_sep_trail, entropy_IG_sum_trail = get_IG_trails(de_model, entropy_obs, Ts, prior_logdets, sn=posterior_sn)
    joint_IG_sep_trail, joint_IG_sum_trail = get_IG_trails(de_model, joint_obs, Ts, prior_logdets, sn=posterior_sn)
    rand_IG_sep_trail, rand_IG_sum_trail = get_IG_trails(de_model, rand_obs, Ts, prior_logdets, sn=posterior_sn)


    greedy_IGsum_sep_trail, greedy_IGsum_sum_trail = get_IG_trails(de_model, greedy_sum_obs, Ts, prior_logdets, sn=posterior_sn)
    dynamic_beta_sep_trail, dynamic_beta_sum_trail = get_IG_trails(de_model, dynamic_beta_obs, Ts, prior_logdets, sn=posterior_sn)

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
      np.savez(f'trial-{index}-setting-SDE-time-{time_str}', t=t, Y=Y, Ts=Ts, Ss=Ss, times=times, 
             d=d, budget=budget, obs=obs, IG_trail=IG_trail)

    return


import os
from os.path import join as oj
from copy import deepcopy

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool


if __name__ == '__main__':

    results_dir = oj('results', 'DE')


    os.makedirs(results_dir, exist_ok=True)
    n_trials = 5
    print(f'Start running {n_trials} trials for DE-SDE experiments.')


    n_cores = n_trials
    with ThreadPool(n_cores) as pool:
        input_arguments = [(results_dir,i +1)  for i in range(n_trials)]
        output = pool.starmap(run_single_experiment, input_arguments)
