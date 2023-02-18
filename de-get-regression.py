import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

import numpy as np
from npde_master.utils import gen_data, plot_model, eval_model, em_int
from npde_master.npde_helper import build_model, fit_model, save_model, load_model

import os
from os.path import join as oj
import pandas as pd
import numpy as np
from scipy.stats import sem
from collections import defaultdict

from utils.utils import cwd

def create_and_fit_model(t, Y, de_type):
    sess = tf.InteractiveSession()
    de_type = de_type.lower()
    if de_type == 'ode':

    # for ode
        npde = build_model(sess, t, Y, model='ode', sf0=1.0, ell0=np.ones(2), W=6, ktype="id")

        prior_sn = npde.sn.eval()
        # print(f'Prior noise variances are {prior_sn}.')

        npde = fit_model(sess, npde, t, Y, num_iter=500, print_every=50, eta=0.02, plot_=False)

        posterior_sn = npde.sn.eval()

    elif de_type == 'vdp':
    # for vdp
        npde = build_model(sess, t, Y, model='sde', sf0=1.0, ell0=np.ones(2), W=6, ellg0=[1e5], ktype="id")

        prior_sn = npde.sn.eval()

        npde = fit_model(sess, npde, t, Y, Nw=100, num_iter=1000, print_every=50, eta=0.02, plot_=False)

        posterior_sn = npde.sn.eval()

    elif de_type == 'sde':
    # for sde 
        npde = build_model(sess, t, Y, model='sde', sf0=1.0, ell0=np.ones(2), W=6, ellg0=[1.0], ktype="id", fix_Z=True)

        prior_sn = npde.sn.eval()

        npde = fit_model(sess, npde, t, Y, Nw=50, num_iter=1000, print_every=50, eta=0.01, plot_=False)

        posterior_sn = npde.sn.eval()
    
    else:
        raise NotImplementedError(f"DE type: {de_type} not implemented ")
    
    return npde, prior_sn, posterior_sn

def get_times(overall_t, overall_Y, baseline_obs):
    baseline_obs = np.unique(baseline_obs, axis=0)

    baseline_times = []
    baseline_obs_time_synced = []
    for t_i, y_i in zip(overall_t, overall_Y):
        for ob_i in baseline_obs:
            if (y_i == ob_i).all():
                baseline_times.append(t_i)
                baseline_obs_time_synced.append(list(y_i))
                
    baseline_obs_time_synced = np.asarray(baseline_obs_time_synced)
    baseline_times = np.asarray(baseline_times)
    assert len(baseline_times) == len(baseline_obs_time_synced)
    return baseline_times, baseline_obs_time_synced 

def get_rmse(overall_t, overall_Y, baseline_obs, target, de_type):     
    baseline_times, baseline_obs_synced = get_times(overall_t, overall_Y, baseline_obs)
    npde, prior_sn, posterior_sn = create_and_fit_model([baseline_times], [baseline_obs_synced], de_type)
    
    target_times, target_synced = get_times(overall_t, overall_Y, target)
    
    return eval_model(npde, [target_times], [target_synced], Nw=1)

def get_regression_df(mse_results):
    data_df = defaultdict(list)
    for collab_type, mse_list in mse_results.items():
        baseline = collab_type.replace('-avg-mses', '').replace('-mses', '').replace('_obs','')
        if baseline not in data_df['Baselines']:
            data_df['Baselines'].append(baseline)
        if '-avg-mses' in collab_type:
            avg = np.mean(mse_list)
            se = sem(mse_list)
            data_df['Avg MSE'].append(avg)
            data_df['Stderr'].append(se)
        else:
            stds = np.std(mse_list, axis=1)

            mean_std_mse = np.mean(stds)
            se_std_mse = sem(stds)
            data_df['Std MSE'].append(mean_std_mse)
            data_df['Stderr Std'].append(se_std_mse)
    
    return pd.DataFrame(data=data_df)


def get_mses(collab_type, n, overall_t, overall_Y, baseline_obs, Ts, de_type):
    if 'indiv' in collab_type:
        mses = [get_rmse(overall_t, overall_Y, baseline_obs[i], Ts[i], de_type) for i in range(n) ]

    else:
        mses = []
        baseline_times, baseline_obs_synced = get_times(overall_t, overall_Y, baseline_obs)
        npde, prior_sn, posterior_sn = create_and_fit_model([baseline_times], [baseline_obs_synced], de_type)
        
        for i, target in enumerate(Ts):
            target_times, target_synced = get_times(overall_t, overall_Y, target)
            mses.append(eval_model(npde, [target_times], [target_synced], Nw=1))                                   
    return mses


def run_regression_analysis(de_type='ODE', n_trials=5, trajectory_index=2, n=5):

    results_dir = oj('results', 'DE')

    # load stored experimental results data
    result_datas = []
    for file in os.listdir(results_dir):
        if file.endswith('.npz') and de_type.upper() in file:
            result_datas.append(np.load(oj(results_dir, file), allow_pickle=True))


    # start analysis on regression performance
    mse_results = defaultdict(list)
    
    for trial_i in range(n_trials):
        print(f'**** trial index {trial_i+1} **** ')
        # print()
        obs = result_datas[trial_i]['obs'].item()
        times = result_datas[trial_i]['times']

        overall_t = result_datas[trial_i]['t'][trajectory_index]
        overall_Y = result_datas[trial_i]['Y'][trajectory_index]
        
        Ts = result_datas[trial_i]['Ts']
        for collab_type, baseline_obs in obs.items():   
#                 if 'rand' in collab_type:
#                 if 'rand' in collab_type or 'indiv' in collab_type: continue
            try:
                mses = get_mses(collab_type, n, overall_t, overall_Y, baseline_obs, Ts, de_type)
                mse = np.mean(mses)

                mse_results[collab_type+'-avg-mses'].append(mse)        
                mse_results[collab_type+'-mses'].append(mses)

                # print(f'*** {collab_type} MSEs: {mses} Avg MSE: {mse} ***')
            except Exception as inst:
                print(f'{trial_i} - {collab_type}')
                print(type(inst))    # the exception instance
                print(inst.args)     # arguments stored in .args
                print(inst)          # __str__ allows args to be printed directly,
                                     # but may be overridden in exception subclasses

    # store regression performance results
    regression_dir = oj(results_dir, 'regression_results')
    os.makedirs(regression_dir, exist_ok=True)
    
    time_str = datetime.datetime.fromtimestamp(time.time()).strftime('%m-%d-%H:%M:%S')

    regression_df = get_regression_df(mse_results)
    regression_df.to_latex(oj(regression_dir, f'{de_type}-regression-{time_str}.tex'), index=False) 
    regression_df.to_csv(oj(regression_dir, f'{de_type}-regression-{time_str}.csv'), index=False) 
    return


import os
from os.path import join as oj
from copy import deepcopy

from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import time
import datetime

if __name__ == '__main__':

    n_cores = 3
    with ThreadPool(n_cores) as pool:
        input_arguments = [['ODE'] , ['SDE'], ['VDP']]

        print(input_arguments)
        output = pool.starmap(run_regression_analysis, input_arguments)

