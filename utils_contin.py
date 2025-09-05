import numpy as np
import warnings
import math
from scipy.linalg import hankel
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt

###------------------------------------###
#       data process
###------------------------------------###

def hankel_matrix(data, q, p=None):
    """
    Find the Hankel matrix dimensionwise for multiple multidimensional 
    time series
    
    Arguments
    data : [N, T, 1] or [N, T, D] ndarray
        A collection of N time series of length T and dimensionality D
    q : int
        The width of the matrix (the number of features)
    p : int
        The height of the matrix (the number of samples)
    
    """
    
    if len(data.shape) == 3:
        return np.stack([_hankel_matrix(item, q, p) for item in data])
    
    if len(data.shape) == 1:
        data = data[:, None]
    return _hankel_matrix(data, q, p)  
    

def _hankel_matrix(data, q, p=None):
    """
    Calculate the hankel matrix of a multivariate timeseries
    
    data : array
        T x D multidimensional time series
    """
    if len(data.shape) == 1:
        data = data[:, None]

    # Hankel parameters
    if not p:
        p = len(data) - q + 1
    all_hmats = list()
    for row in data.T:
        first, last = row[-(p + q -1) : -(p-1)], row[-p :]
        if (p==1):
            first, last = row[-(p + q -1) : ], row[-p :]
        out = hankel(first, last)
        all_hmats.append(out)
    out = np.dstack(all_hmats)
    return np.transpose(out, (1, 0, 2))#[:-1]


def resample_dataset(
    data, n_samples=None, randomize=True, random_state=None
):
    """
    Generate random samples from a dataset. This function is comparable to the
    deprecated np.shuffle, but it protects the original dataset
    
    Arguments
    data : [N, ...] ndarray
        A collection of N datasets
    n_samples : int
        The number of rows to sample from the matrix
    randomize : bool
        Select random subsets of the data without replacement, and
        return the indices of the chosen subsets
    random_state : int
        The random seed when using randomization
    """
    np.random.seed(random_state)
    if not n_samples:
        n_samples = data.shape[0]
    if randomize:
        selection_indices = np.random.choice(
            np.arange(data.shape[0]), n_samples, replace=False
        )
    else:
        selection_indices = np.arange(n_samples)
    return selection_indices, data[selection_indices]


def standardize_ts(a, scale=1.0):
    """
    Standardize a T x D time series along its first dimension
    For dimensions with zero variance, divide by one instead of zero
    """
    stds = np.std(a, axis=0, keepdims=True)
    stds[stds==0] = 1
    return (a - np.mean(a, axis=0, keepdims=True))/(scale*stds)

def scale_ts(a, Vmin=0, Vmax=1):
    """
    Scale a T x D time series along its first dimension to the range [Vmin, Vmax]

    """
    Amin, Amax = np.min(a, axis=0), np.max(a, axis=0)
    a_scale = (a - Amin)/ (Amax-Amin)
    return (Vmax-Vmin)*a_scale + Vmin


def train_valid_split(x, y, z, time_window, pred_horizon, valid_split=0.0):
    """
    split original time series into train_set and valid_set by 'valid_split'
    This core function is designed to create original mixed embedding vectors
    valid_set is not necessary in our current method but reserved for future learning algorithm
    x : ndarray, the cause   
    y : ndarray, the effect
    z : list whose element is ndarray, the condition  
    """
        # X -> Y  condition on Z

    if z!=[]:
        z1 = z.copy()           
        z1 = np.array(z1)            
        xx = x[None,:]                 
        yy = y[None,:]
        reduce_mat = np.concatenate((yy, z1), axis=0).T             
        full_mat = np.concatenate((yy, z1, xx), axis=0).T
    else:
        reduce_mat = y
        full_mat = np.stack([y, x], axis=1)

    # reduce_mat = standardize_ts(reduce_mat)[:-(time_window)]
    # full_mat = standardize_ts(full_mat)[:-(time_window)]
    # label_mat = standardize_ts(y)[(time_window):] 
    reduce_mat = reduce_mat[:-(time_window)]
    full_mat = full_mat[:-(time_window)]
    label_mat = y[(time_window):] 

    train_index = int( len(label_mat)*(1 - valid_split) )
    train_maxlen = max(min(train_index - time_window, train_index - pred_horizon), 0)

    valid_index = int( 1 + len(label_mat)* valid_split )
    valid_maxlen = max(min(valid_index - time_window, valid_index - pred_horizon) + 1, 0)
    # print(valid_maxlen)
    label_train = hankel_matrix(label_mat[:train_index], pred_horizon)[:train_maxlen]
    label_valid = hankel_matrix(label_mat[-valid_index:], pred_horizon)[:valid_maxlen]
    label_train = label_train.squeeze()
    label_valid = label_valid.squeeze()

    X0 = hankel_matrix(reduce_mat, time_window)
    Y0 = hankel_matrix(full_mat, time_window)

    reduce_train = X0[:train_maxlen, :time_window ]
    full_train = Y0[:train_maxlen, :time_window ]

    reduce_valid = X0[-valid_maxlen:, :time_window ][:valid_maxlen]
    full_valid = Y0[-valid_maxlen:, :time_window ][:valid_maxlen]
    if pred_horizon == 1:
        label_train = label_train[:,None]
        label_valid = label_valid[:,None]
    return (reduce_train, full_train, label_train), (reduce_valid, full_valid, label_valid)


def time_shifted_surrogates(x, N=100, lag=None, random_seed=None):
    """
    generate a time-shifted surrogate time series
    N : int,  the number of surrogates
    lag : list/1D-nadarray shape (N,),  the cutting position
    random_seed : int,  ensures a reprocudible result
    """
    np.random.seed(random_seed)
    if lag is None:
        lag = np.random.randint(20, x.shape[0]-20, size=N)

    surrogates = []
    for i in range(N):
        x_surrogate = np.concatenate((x[lag[i]:], x[:lag[i]]), axis=0)
        surrogates.append(x_surrogate)

    return np.array(surrogates)



def contimpro(embed_reduce, embed_full, target, R=5, metric='minkowski', p=2, Theiler_window=0):
    """
    realization of Continuity Improvement
    embed_reduce:  the embedding of preimage set of f_reduced
    embed_full:  the embedding of preimage set of f_full
    target:  the image set of underlying map/equation
    """
    def dis_func(idxs, target):
        ex =  [np.max(cdist(target[k].reshape(-1,target[k].shape[0]), target[idxs[k]], metric=metric, p=p))  for k in range(len(idxs))]
        return np.array(ex)

    fullTree = KDTree(embed_full, metric=metric, p=p)
    reduceTree = KDTree(embed_reduce, metric=metric, p=p)

    _, ind_reduce0 = reduceTree.query(embed_reduce, k=R+1+2*Theiler_window)
    valid_mask = np.abs(ind_reduce0 - np.arange(embed_reduce.shape[0])[:,np.newaxis]) > Theiler_window
    idxs_reduce = np.asarray([ind_reduce0[ii][valid_mask[ii]][:R] for ii in range(ind_reduce0.shape[0])])

    _, ind_full0 = fullTree.query(embed_full, k=R+1+2*Theiler_window)
    valid_mask = np.abs(ind_full0 - np.arange(embed_full.shape[0])[:,np.newaxis]) > Theiler_window
    idxs_full = np.asarray([ind_full0[ii][valid_mask[ii]][:R] for ii in range(ind_full0.shape[0])])

    dis_full = dis_func(idxs_full, target)
    dis_reduce = dis_func(idxs_reduce, target)

    m_re = dis_reduce.mean() 
    m_fu = dis_full.mean()                
    ci = max((m_re - m_fu) / m_re, 0)  # output: a scalar value
    # print(ci)
    return ci, dis_reduce, dis_full


###------------------------------------###
#       component selection
###------------------------------------###


def project(embed_vec, flag):
    """
    an auxiliary function for Advanced Selection
    produce a new embedding based on 'flag' that records reserved coordinates/components 
    """
    cloud = []
    for i in range(embed_vec.shape[1]):
        if flag[i]==1:
            cloud.append(embed_vec[:,i])
    cloud = np.array(cloud).transpose()
    return cloud

def is_low(eps1, eps2, bound):
    if eps1 < (eps2 + bound):
        return True
    else:
        return False


def calc_epsilon(input_cloud, output_cloud, k=5, metric='minkowski', p=2, Theiler_window=0):    
    """
    an auxiliary function For Advanced Selection
    calculate the Continuity statistic given preimage set And image set
    """

    featTree = KDTree(input_cloud, metric=metric, p=p)
    _, featinds = featTree.query(input_cloud, k=k+1+2*Theiler_window)
    valid_mask = np.abs(featinds - np.arange(featinds.shape[0])[:,np.newaxis]) > Theiler_window
    idxs = np.asarray([featinds[ii][valid_mask[ii]][:k] for ii in range(featinds.shape[0])])
    epsilon = [np.max(cdist(output_cloud[k].reshape(-1, output_cloud[k].shape[0]), output_cloud[idxs[k]], metric=metric, p=p)) for k in range(len(idxs))]
    #print(np.array(epsilon).shape)
    epsilon = np.array(epsilon).mean()
    # print(epsilon)
    return epsilon


def Advanced_Select1(embed_vec, label_vec, k=5, significant_bound=0.005, start_indice=0, metric='minkowski', Greedy=True, p=2, Theiler_window=0):
    """
    The first Advanced selection
    significant_bound:  float,  determines μ
    start_indice:  int,  the number of principle components reserved without consideration for reducing the statistic
    Greedy:  boolean,  applying greedy startegy in reducing/minimizing the statistic
    """

    def DFS(dep, flag):
        # deep first search
        # this function is reserved for solving a global optimization
        global optim
        global Flag
        if dep>tol_indice and (flag[0:tol_indice+1]==0).all():
            return
        if dep>=(embed_vec.shape[1]):
            new_eps = calc_epsilon( project(sort_embed, flag), label_vec, k=k, metric=metric, p=p, Theiler_window=Theiler_window)

            if (is_low(new_eps, optim, -1*significant_bound*radius)): 
                if (new_eps<optim):
                    optim = new_eps
                Flag = flag.copy()
            return
        flag[dep]=0
        DFS(dep+1, flag)
        flag[dep]=1
        DFS(dep+1, flag)
        return


    if Greedy is None:
        if embed_vec.shape[1]>=5:
            Greedy = True
        else:
            Greedy = False

    radius = cdist(label_vec, label_vec.mean(axis=0).reshape(-1, label_vec.shape[1]), metric=metric, p=p).std()
    global Flag
    Flag = np.zeros(embed_vec.shape[1])
    if start_indice <= 0:
    # sort 'embed_vec' by the variance of each dimension
        sort_idx = np.argsort(-np.std(embed_vec, axis=0))

        sort_embed = embed_vec[:, sort_idx]
    else:
        sort_embed = embed_vec

    tol_indice = max(start_indice-1, 0)

    global optim
    optim = calc_epsilon(sort_embed[:, 0:tol_indice+1], label_vec, k=k, metric=metric, p=p, Theiler_window=Theiler_window)
    if Greedy:
        flag = Flag.copy()
        flag[0: tol_indice+1] = 1
        for i in range(tol_indice+1, embed_vec.shape[1]):
            flag[i] = 1
            eps = calc_epsilon( project(sort_embed, flag), label_vec, k=k, metric=metric, p=p, Theiler_window=Theiler_window)
            if (is_low(eps, optim, -1*significant_bound*radius)):
                optim = eps
            else:
                flag[i]=0
        Flag = flag.copy()
    else:
        DFS(0, Flag.copy())

    return project(sort_embed, Flag), Flag



def Advanced_Select2(embed_vec, cause_vec, label_vec, k=5, start_indice=0, significant_bound=0.005, metric='minkowski', p=2, Theiler_window=0):
    """
    The second Advanced selection
    significant_bound:  float,  determines μ
    start_indice:  int,  the number of principle components reserved without consideration for reducing the statistic
    """
    radius = cdist(label_vec, label_vec.mean(axis=0).reshape(-1, label_vec.shape[1]), metric=metric, p=p).std()
    global Flag
    Flag = np.zeros(cause_vec.shape[1])
    global optim
    optim = calc_epsilon(embed_vec, label_vec, k=k, metric=metric, p=p, Theiler_window=Theiler_window)
    # flag = Flag.copy()

    for i in range(Flag.shape[0]):
        eps_dict = {}    
        for j in range(Flag.shape[0]):
            if Flag[j] == 0:
                cat_embed = np.concatenate((embed_vec, cause_vec[:embed_vec.shape[0], j:j+1]), axis=1)
                eps = calc_epsilon(cat_embed, label_vec, k=k, metric=metric, p=p, Theiler_window=Theiler_window)
                eps_dict[Flag.shape[0]-j] = eps
        min_key = min(eps_dict, key=eps_dict.get)
        if (is_low(eps_dict[min_key], optim, -1*significant_bound*radius)):
            optim = eps_dict[min_key] 
            ii = Flag.shape[0]-min_key    
            Flag[ii] = 1
            embed_vec = np.concatenate((embed_vec, cause_vec[:embed_vec.shape[0], ii:ii+1]), axis=1)
        else:
            break
    return embed_vec, Flag


def intercept_embedvec(embed_vec, var_proportion=0.95):
    L_var = []
    for i in range(embed_vec.shape[1]):
        L_var.append(np.std(embed_vec[:,i])**2)

    totalvar = sum(L_var)
    accum = 0
    coords = []
    indices = np.argsort( -np.array(L_var) ) # descend order
    for i in range(len(indices)):
        coords.append(embed_vec[:, indices[i]])
        accum += L_var[indices[i]]
        if accum/totalvar >= var_proportion:
            break
            
    coords = np.array(coords).T
    # m = coords.shape[1] # the number of effective coordinates

    return coords


###------------------------------------###
#       data analysis
###------------------------------------###

def mutual_information(X, max_lag):
    '''
    Calculates the mutual information between the an unshifted time series
	and a shifted time series. Utilizes scikit-learn's implementation of
	the mutual information found in sklearn.metrics.
	Parameters
	----------
    X : 1-D time series, array-like
	max_lag : integer
	    maximum amount to shift the time series
	Returns
	-------
	m_score : 1-D array
	    mutual information at between the unshifted time series and the
	    shifted time series
	'''

	#number of bins - say ~ 20 pts / bin for joint distribution
	#and that at least 4 bins are required
    N = max(X.shape)
    num_bins = max(4.,np.floor(np.sqrt(N/20)))
    num_bins = int(num_bins)

    m_score = np.zeros((max_lag))

    for jj in range(max_lag):
        lag = jj+1

        ts = X[0:-lag]
        ts_shift = X[lag::]

        min_ts = np.min(X)
        max_ts = np.max(X)+.0001 #needed to bin them up

        bins = np.linspace(min_ts,max_ts,num_bins+1)

        bin_tracker = np.zeros_like(ts)
        bin_tracker_shift = np.zeros_like(ts_shift)

        for ii in range(num_bins):

            locs = np.logical_and( ts>=bins[ii], ts<bins[ii+1] )
            bin_tracker[locs] = ii

            locs_shift = np.logical_and( ts_shift>=bins[ii], ts_shift<bins[ii+1] )
            bin_tracker_shift[locs_shift]=ii


        m_score[jj] = mutual_info_score(bin_tracker,bin_tracker_shift)
    return m_score

from scipy.stats import binom
def Continuity_index(Origin, Recons, n_p=200, delta0=0.2, epsilon=None, random_state=32, metric='minkowski', p=2):
    assert Origin.shape[0] == Recons.shape[0]
    N = Origin.shape[0]

    std_o = cdist(Origin, Origin.mean(axis=0).reshape(-1,Origin.shape[1]), metric=metric, p=p).std()
    std_r = cdist(Recons, Recons.mean(axis=0).reshape(-1,Recons.shape[1]), metric=metric, p=p).std()
    if epsilon is None:
        epsilon = np.array([0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,1.2]) * std_r
    else:
        epsilon = np.asarray(epsilon)*std_r 
    C_list = []

    np.random.seed(random_state)
    inds = np.random.choice(N, n_p, replace=True)

    O_tree = KDTree(Origin, metric=metric, p=p)
    R_tree = KDTree(Recons, metric=metric, p=p)
    for i in range(len(epsilon)):
        Continuity_stat = 0
        num = 0
        for ii in range(n_p):
            delta = delta0 * std_o
            count_ = -1
            while (not (count_==1)):

                delta_set = O_tree.query_radius(Origin[inds[ii]].reshape(1,-1), delta)[0]
                count_ = delta_set.shape[0]

                dist = cdist(Recons[delta_set], Recons[inds[ii]].reshape(1,-1), metric=metric, p=int(2))
                # print('count', count_)
                # print(Recons[delta_set], Recons[inds[ii]], dist)
                if np.concatenate([dist[k] < epsilon[i] for k in range(len(dist))], axis=0).all():
                    break
                else:
                    delta = delta * 0.8
      
            n_delta = count_ - 1

            n_eps = R_tree.query_radius(Recons[inds[ii]].reshape(1,-1), epsilon[i], count_only=True)[0] - 1
            # print(delta, epsilon[i], (n_delta), (n_eps))
            p = round(n_eps / N, 4)
            # print(ii, n_delta, p)
            p_max = max([binom.pmf(k=k, n=n_delta, p=p) for k in range(0,n_delta+1)])
            # if n_delta > 0:
            Continuity_stat += (1 - (p ** n_delta) / p_max)
            num += 1
            # print(Continuity_stat, num)
        Continuity_stat = Continuity_stat/num
        
        C_list.append(Continuity_stat)

    return C_list


def evaluate(causalmat, causaldict, count):
    true_pos = 0
    true_neg = 0
    fals_pos = 0
    fals_neg = 0
    for i in range(len(causaldict)):
        cas_var = i
        eff = np.array(causaldict[i])
        for j in range(len(causaldict)):
            if (i==j):
                continue
            if (j==eff).any():
                eff_var = j
                true_pos = true_pos + causalmat[cas_var][eff_var]
                fals_neg = fals_neg + count - causalmat[cas_var][eff_var]
            else: 
                fals_pos = fals_pos + causalmat[cas_var][j]
                true_neg = true_neg + count - causalmat[cas_var][j] 
    ACC = (true_pos+true_neg)/(true_pos+fals_pos+fals_neg+true_neg)
    TPR = true_pos/(true_pos+fals_neg) # sensitivity/ recall
    Recall = TPR
    Precision = true_pos/(true_pos+fals_pos)
    F1_score = (2*true_pos)/(2*true_pos + fals_pos + fals_neg)
    if ((true_pos+fals_pos)*(true_pos+fals_neg)*(true_neg+fals_pos)*(true_neg+fals_neg)==0):
        MCC = 0
    else:
        MCC = (true_pos*true_neg - fals_pos*fals_neg) / math.sqrt((true_pos+fals_pos)*(true_pos+fals_neg)*(true_neg+fals_pos)*(true_neg+fals_neg))
    FPR = fals_pos/(fals_pos+true_neg)  
    #TNR = true_neg/(fals_pos+true_neg) # specificity        
    return (true_pos, true_neg, fals_pos, fals_neg), (ACC, FPR, Recall, Precision, F1_score, MCC)
