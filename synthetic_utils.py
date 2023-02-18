
import numpy as np

from copy import deepcopy

def plot_gp(mu, cov, X, samples=[]):
    X = X.reshape(-1)
    mu = mu.reshape(-1)

    # 95% confidence interval
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.figure(figsize=(8, 6))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label='sample_{}'.format(i))

    plt.legend()
    
    
def gaussian_rbf(x1, x2, l=1, sigma_f=1):
    # distance between each rows
    dist_matrix = np.sum(np.square(x1), axis=1).reshape(-1, 1) + np.sum(np.square(x2), axis=1) - 2 * np.dot(x1, x2.T)
    return np.square(sigma_f) * np.exp(-1 / (2 * np.square(l)) * dist_matrix)
kernel = gaussian_rbf


def posterior_predictive(X, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    K = kernel(X_train, X_train, l, sigma_f) + np.square(sigma_y) * np.eye(len(X_train))
    K_s = kernel(X_train, X, l, sigma_f)
    K_ss = kernel(X, X, l, sigma_f)
    
    K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))

    mu_s = K_s.T @ K_inv @ Y_train
    cov_s = K_ss - K_s.T @ K_inv @ K_s

    return mu_s, cov_s


def posterior_covariance(X, X_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    K = kernel(X_train, X_train, l, sigma_f) + np.square(sigma_y) * np.eye(len(X_train))
    K_s = kernel(X_train, X, l, sigma_f)
    K_ss = kernel(X, X, l, sigma_f)
    K_inv = np.linalg.inv(K + 1e-6 * np.eye(len(K)))

    cov_s = K_ss - K_s.T @ K_inv @ K_s

    return cov_s


def IG_sum(acquired_obs, Ts, prior_logdets, betas):
    
    return sum(_IG(acquired_obs, T, prior_logdet) *1.0 / beta for T, prior_logdet, beta in zip(Ts, prior_logdets, betas) )

def _IG(acquired_obs, T, prior_logdet):
    d = acquired_obs.shape[1]
    post_cov = posterior_covariance(X=T, X_train=acquired_obs.reshape(-1, d))
    _ , post_logdet = np.linalg.slogdet(post_cov)
    return 0.5 * (prior_logdet - post_logdet)


from itertools import product


def coordinated_greedy(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], d=1):
    if len(betas) == 0:
        betas = np.ones(len(Ts)) / len(Ts)
    else:
        assert len(betas) == len(Ts)
        betas = np.asarray(betas) / sum(betas)
        
    acquired_obs = np.asarray([]).reshape(-1, d)
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
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            delta_IG = IG_sum(temp_obs, Ts, prior_logdets, betas) - prev_IG
            # the weighted sum of difference in IG_k - IG_{k-1} in Equation (2)
            
            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        
        prev_IG = IG_sum(acquired_obs, Ts, prior_logdets, betas)
        
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs

def coordinated_joint(Ss, joint_target, budget, prior_logdet_joint, subset_size=1000, d=1):
    acquired_obs = np.asarray([]).reshape(-1, d)
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
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)
            
            delta_IG = _IG(temp_obs, joint_target, prior_logdet_joint) - prev_IG
            
            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs
        
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)                
        
        prev_IG = _IG(acquired_obs, joint_target, prior_logdet_joint)

        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs


def coordinated_random(Ss, Ts, budget, prior_logdets, d=1):
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

def entropy_sum(acquired_obs, Ts, betas=[]):
    if len(betas) == 0:
        betas = np.ones(len(Ts)) / len(Ts)
    else:
        betas = np.asarray(betas)
    
    return sum(_entropy(acquired_obs, T) *1.0 / beta for T, beta in zip(Ts, betas) )

def _entropy(acquired_obs, T, exact=False):
    '''
    Note this is not the exact differential entropy formula, 
    instead it ignores some constant terms including the dimension d of data.
    
    For the purpose of maximum entropy search, it is sufficient since we only need the rank and not the 
    absolute value of entropy.
    '''
    post_cov = posterior_covariance(T, acquired_obs)
    _ , post_logdet = np.linalg.slogdet(post_cov)

    if not exact:        
        return post_logdet
    
    else:
        d = len(acquired_obs[0])
        differential_entropy = 0.5 * post_logdet + d /2.0 * np.log(2 * np.pi * np.exp(1))
        
        return differential_entropy


def coordinated_entropy(Ss, Ts, budget, subset_size=1000, d=1):
    acquired_obs = np.asarray([]).reshape(-1, d)
    
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    for _ in range(budget):
        obs_ = None
        
        entropy_max = -float('inf')

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]

        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            curr_entropy = entropy_sum(temp_obs, Ts)
            
            if curr_entropy > entropy_max:
                entropy_max = curr_entropy
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs

def individual_greedy(S, T, prior_logdet, budget, d=1, subset_size=1000):
        
    acquired_obs = np.asarray([]).reshape(-1, d)
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
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            delta_IG = _IG(temp_obs, T, prior_logdet) - prev_IG

            if delta_IG > delta_IG_max:
                delta_IG_max = delta_IG
                obs_ = obs

        IG_trail.append(delta_IG_max)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
        prev_IG = _IG(acquired_obs, T, prior_logdet)

        Support = S[S!= obs_].reshape(-1,d)

    return acquired_obs, IG_trail

def _check_betas(n, betas=[]):
    if len(betas) == 0:
        betas = np.ones(n) / n
    else:
        assert len(betas) == n
        betas = np.asarray(betas) / sum(betas)
    return betas
    

def coordinated_greedy_IG_sum(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], d=1):
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

    for _ in range(budget):
        IG_sum_max = -float('inf')
        obs_ = None

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            IG_sum_curr = IG_sum(temp_obs, Ts, prior_logdets, betas)
            # Directly try to maximize the total sum of IGs
            
            if IG_sum_curr > IG_sum_max:
                IG_sum_max = IG_sum_curr
                obs_ = obs

#         print("budget: {}".format(_), IG_max, obs_)
        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)
                
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs

from scipy.special import softmax
def coordinated_dynamic_beta(Ss, Ts, budget, prior_logdets, subset_size=1000, betas=[], beta_coef=0.5, d=1):
    '''
    The beta values are dynamically updated according to the latest IGs of the agents to help improve
    "cumulative" fairness.

    
    '''
    betas = _check_betas(n=len(Ts), betas=betas)

    acquired_obs = np.asarray([]).reshape(-1, d)
    # supports is a copy of Ss so we do not need to repeatedly initialize Ss
    Supports = deepcopy(Ss)

    for _ in range(budget):
        IG_sum_max = -float('inf')
        obs_ = None

        full_cartesian = np.asarray(list(product(*Supports)))
        subset_size = min(subset_size, len(full_cartesian))

        subset_cartesian = full_cartesian[np.random.choice(len(full_cartesian), size=subset_size, replace=False)]
        for obs in subset_cartesian:
            temp_obs = np.append(acquired_obs, [obs]).reshape(-1, d)

            IG_sum_curr = IG_sum(temp_obs, Ts, prior_logdets, betas)
            # Directly try to maximize the total sum of IGs
            
            if IG_sum_curr > IG_sum_max:
                IG_sum_max = IG_sum_curr
                obs_ = obs

        acquired_obs = np.append(acquired_obs, [obs_]).reshape(-1, d)

        individual_IGs = [_IG(acquired_obs, T, prior_logdet) for T, prior_logdet in zip(Ts, prior_logdets) ]
        
        updated_betas = softmax(individual_IGs)
        betas = beta_coef * betas + (1-beta_coef) * updated_betas
    
        for i, (S, ob) in enumerate(zip(Supports, obs_)):
            Supports[i] = S[S != ob].reshape(-1, d)

    return acquired_obs


def get_IG_trails(obs, Ts, prior_logdets, betas=[]):
    betas = _check_betas(n=len(Ts), betas=betas)

    IG_sep_trail, IG_sum_trail = [], []

    d = obs.shape[1]
    acquired_obs = np.asarray([]).reshape(-1, d)
    
    for ob in obs:
        curr_IG_sep = []

        acquired_obs = np.append(acquired_obs, [ob]).reshape(-1, d)
        for i, (T, prior_logdet) in enumerate(zip(Ts, prior_logdets)):

            IG_i = _IG(acquired_obs, T, prior_logdet)
            curr_IG_sep.append(IG_i)
        IG_sep_trail.append(curr_IG_sep)
        
        IG_sum_trail.append(sum(IG*1.0/beta for beta, IG in zip(betas, curr_IG_sep)) )

    return IG_sep_trail, IG_sum_trail