import warnings
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA, SparsePCA, KernelPCA
from sklearn.neighbors import KDTree
from utils_contin import scale_ts, hankel_matrix, train_valid_split, Continuity_index, contimpro, Advanced_Select1, Advanced_Select2
import math
from tqdm import trange
# from matplotlib import pyplot as plt


class StandardContinuityImprove(object):
    def __init__(
        self,
        time_window,
        pred_horizon,
        n_latent,
        model='PCA',
        random_state=None,
        kernel='rbf',
        n_variables=None,
        *args,
        **kwargs
        ):
        self.time_window = time_window
        self.pred_horizon = pred_horizon
        self.n_latent = n_latent
        self.random_state = random_state
        self.model_name = model
        self.kernel = kernel
        self.n_variables = n_variables


    def fit_transform(
        self,
        cause_ts,
        effect_ts,
        condition_ts=[],
        valid_split=0.0,
        **kwargs
        ):
        (reduce_train, _, label_train), (_, _, _) = \
               train_valid_split(cause_ts, effect_ts, condition_ts, self.time_window, self.pred_horizon, valid_split)
 
        if self.model_name is None:
            return np.reshape(reduce_train, (reduce_train.shape[0], -1), order='F'), label_train

        else:

            model_reduced = PCA(
                n_components=self.n_latent,
                random_state=self.random_state
            )
            # print(model_reduced)
            model_reduced.fit(np.reshape(reduce_train, (reduce_train.shape[0], -1), order='F'))
            return model_reduced.transform(np.reshape(reduce_train, (reduce_train.shape[0], -1), order='F')), label_train


    def get_mean_significance(self, rad_r, rad_f, test_name="ttest"):
        Test = {"ttest":0, "wilcox": 1}
        if Test[test_name] is 0:
            statis, pval = ttest_rel(rad_r, rad_f, nan_policy='raise', alternative='greater')
        elif Test[test_name] is 1:
            statis, pval = wilcoxon(rad_r, rad_f, correction=True, alternative='greater')
        else:
            raise KeyError("The test_name is not right! Please specify 'ttest' or 'wilcox'.")   
        return statis, pval 


    def premise_test(self, cause, effect, condition, advanced_select=True, R=5, scale=False, DistanceMetric='minkowski', p=2):
        if scale == True:
            cause = scale_ts(cause)
            effect = scale_ts(effect)
            condition = [scale_ts(condition[i]) for i in range(len(condition))]

        latent_reduce, label = self.fit_transform(cause, effect, condition)
        if advanced_select:
            embed_reduce, _ = Advanced_Select1(latent_reduce, label, k=R, metric=DistanceMetric, p=p)
        else:
            embed_reduce = latent_reduce

        record = np.zeros((self.time_window,))
        for i in range(self.time_window):
            record[i] = Continuity_index(embed_reduce, (cause[i:])[:embed_reduce.shape[0], np.newaxis], epsilon=[1.0], metric=DistanceMetric, p=p)[0] 
        return record


    def onepair_test(self, cause, effect, condition, R=5, pair_test="ttest", scale=False, \
            advanced_select=True, embed_reduce=None, DistanceMetric='minkowski', p=2, Theiler_window=1):
        '''
        cause/effect: array-like, ndarray shape (T,)  scalar time series recordings
        condition: list or array-like, list [x1, x2, ..., xn] xi is ndarray type shape (T,) or ndarray shape (T,N)
        advanced_select: bool, if True, 'select_embedvec' is employed
        embed_reduce: None by default, array-like, ndarray shape (T',n_l), if it's provided, do 'embed_full' only
        '''
        if scale == True:
            cause = scale_ts(cause)
            effect = scale_ts(effect)
            condition = [scale_ts(condition[i]) for i in range(len(condition))]


        if embed_reduce is None:

            latent_reduce, label = self.fit_transform(cause, effect, condition)

            if advanced_select:
                embed_reduce,_ = Advanced_Select1(latent_reduce, label, k=R, metric=DistanceMetric, p=p)
            else:
                embed_reduce = latent_reduce 
        else:
            pass
        
        (_, full, label), (_,_,_) = train_valid_split(cause, effect, condition, self.time_window, self.pred_horizon)
        cause_vec = full[:,:,-1]

        latent_full = np.concatenate((embed_reduce, cause_vec.reshape(cause_vec.shape[0], -1))[:embed_reduce.shape[0]], axis=1)

        if advanced_select:
            embed_full, flag = Advanced_Select2(embed_reduce, cause_vec, label, k=R, metric=DistanceMetric, p=p)
        else:
            embed_full = latent_full       
        ci, dis_reduce, dis_full = contimpro(embed_reduce, embed_full, label, R=R, metric=DistanceMetric, p=p, Theiler_window=Theiler_window)

        if embed_full.shape[1] == embed_reduce.shape[1]:
            alpha = 1
        else:
            _, alpha = self.get_mean_significance(dis_reduce, dis_full, test_name=pair_test)

        delay = []
        if advanced_select:
            # flag = flag[embed_reduce.shape[1]:]
            W = len(flag)
            for i in range(W):
                if flag[i] == 1:
                    delay.append(W-i)

        return ci, alpha, dis_reduce, dis_full, delay


    def run_mci(self, data, advanced_select=True, R=5, pair_test='ttest', scale=False, DistanceMetric='minkowski', p=2, Theiler_window=1):
        '''
        data: list or array-like, 
              e.g. list [x1, x2, ..., xn]  xi is ndarray type shape (T,)  ndarray shape (T, N)
                   T -- the length of time series, N -- the number of variables 
        '''
        if scale == True:
            if type(data) is list:
                data = [scale_ts(data[i]) for i in range(len(data))]
            else:
                data = scale_ts(data)


        if type(data) is not list:
            data = data.transpose()
        N = len(data)
        self.MCI_valmatrix = np.zeros((N,N))
        self.MCI_signmatrix = np.ones((N,N))
        self.MCI_delaymatrix = {i: [[] for j in range(N)] for i in range(1,N+1)}

        for i in trange(N):

            cause_ts = data[i]


            for j in range(N):
                if i != j:

                    lis = [k for k in range(N)]
                    lis.remove(i)
                    lis.remove(j)
                    effect_ts = data[j]
                    condition_ts = [data[k] for k in lis]   
                    
                    (_, full, label), (_,_,_) = train_valid_split(cause_ts, effect_ts, condition_ts, self.time_window, self.pred_horizon)
                    cause_vec = full[:,:,-1]
                    latent_reduce, _ = self.fit_transform(cause_ts, effect_ts, condition_ts)
            
                    embed_reduce,_ = Advanced_Select1(latent_reduce, label, k=R, metric=DistanceMetric, p=p, Theiler_window=Theiler_window) if advanced_select else latent_reduce
                 
                    latent_full = np.concatenate((embed_reduce, cause_vec), axis=1)
 
                    embed_full, flag = Advanced_Select2(embed_reduce, cause_vec, label, k=R, metric=DistanceMetric, p=p, Theiler_window=Theiler_window) if advanced_select else latent_full
                    ci, dis_reduce, dis_full = contimpro(embed_reduce, embed_full, label, R=R, metric=DistanceMetric, p=p, Theiler_window=Theiler_window)
                    
                    self.MCI_valmatrix[i][j] = ci
                    if embed_full.shape[1] == embed_reduce.shape[1]:
                        alpha = 1
                    else:
                        _, alpha = self.get_mean_significance(dis_reduce, dis_full, test_name=pair_test)
                        self.MCI_signmatrix[i][j] = alpha

                    delay = []
                    if advanced_select:      
                        W = len(flag)
                        for ii in range(W):
                            if flag[ii] == 1:
                                delay.append(W-ii)
                        self.MCI_delaymatrix[j+1][i] = delay

        return self.MCI_valmatrix, self.MCI_signmatrix       


 
try:     
    from joblib import Parallel,delayed
except Exception as e:
    warnings.warn(str(e))

class ParallelContinuityImprove(StandardContinuityImprove):
    def __init__(
        self,
        n_jobs=1,
        *args,
        **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.n_jobs = n_jobs

    # Override
    def onepair_test(self, cause, effect, condition, R=5, pair_test="ttest", scale=False, \
            advanced_select=True, latent_reduce=None, DistanceMetric='minkowski', p=2, Theiler_window=1):

        if scale == True:
            cause = scale_ts(cause)
            effect = scale_ts(effect)
            condition = [scale_ts(condition[i]) for i in range(len(condition))]

        if latent_reduce is None:
            latent_reduce, label = self.fit_transform(cause, effect, condition)
        else:
            pass
       
        (_, full, label), (_,_,_) = train_valid_split(cause, effect, condition, self.time_window, self.pred_horizon)
        cause_vec = full[:,:,-1]
        if advanced_select:

            embed_reduce,_ = Advanced_Select1(latent_reduce, label, k=R, metric=DistanceMetric, p=p)
        else:
            embed_reduce = latent_reduce
        latent_full = np.concatenate((embed_reduce, cause_vec), axis=1)
        if advanced_select:

            embed_full, flag = Advanced_Select2(embed_reduce, cause_vec, label, k=R, metric=DistanceMetric, p=p)
        else:
            embed_full = latent_full       
        ci, dis_reduce, dis_full = contimpro(embed_reduce, embed_full, label, R=R, metric=DistanceMetric, p=p)

        if embed_full.shape[1] == embed_reduce.shape[1]:
            alpha = 1
        else:
            _, alpha = self.get_mean_significance(dis_reduce, dis_full, test_name=pair_test)
        delay = []
        if advanced_select:
            # flag = flag[embed_reduce.shape[1]:]
            W = len(flag)
            for i in range(W):
                if flag[i] == 1:
                    delay.append(W-i)

        return ci, alpha, dis_reduce, dis_full, delay


    def run_mci(self, data, advanced_select=True, R=5, pair_test='ttest', scale=False, DistanceMetric='minkowski', p=2, Theiler_window=1):
        '''
        data: list or array-like, 
              e.g. list [x1, x2, ..., xn]  xi is ndarray type shape (T,) or ndarray shape (T, N)
                   T -- the length of time series, N -- the number of variables 
        '''


        def makecondition(data, ll, j):
            # ll = ll.copy()
            ll.remove(j)
            return [data[k] for k in ll]

        if scale == True:
            if type(data) is list:
                data = [scale_ts(data[i]) for i in range(len(data))]
            else:
                data = scale_ts(data)


        if type(data) is not list:
            data = data.transpose()
        N = len(data)
        self.MCI_valmatrix = np.zeros((N,N))
        self.MCI_signmatrix = np.ones((N,N))
        self.MCI_delaymatrix = {i: [[] for j in range(N)] for i in range(1,N+1)}
        if pair_test:
            surrogate_num = 0   # use the Pair-Sample T test, ignoring the execution of surrogates

        for j in trange(N):

            lis = [k for k in range(N)]
            lis.remove(j)
            lis.remove((j+1)%N)

            ll = [k for k in range(N)]#
            ll.remove(j)
            with Parallel(n_jobs=self.n_jobs) as parallel:
                results = parallel(delayed(self.onepair_test)(data[i], data[j], condition=makecondition(data, ll.copy(), i), \
                                            R=R, DistanceMetric=DistanceMetric, p=p, pair_test=pair_test, advanced_select=advanced_select, \
                                                latent_reduce=None, Theiler_window=Theiler_window)
                                                    for i in ll)

                for i in ll:
                    if j<i:
                        ii = i-1
                    elif j>i:
                        ii = i
                    else:
                        continue
                    self.MCI_valmatrix[i][j] = results[ii][0]
                    self.MCI_signmatrix[i][j] = results[ii][1]
                    self.MCI_delaymatrix[j+1][i] = results[ii][-1]
    
        return self.MCI_valmatrix, self.MCI_signmatrix  
