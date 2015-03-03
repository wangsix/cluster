
import numpy as np
import scipy.cluster.vq as vq
import scipy.stats as stats
from scipy.cluster.hierarchy import * #@UnusedWildImport
from scipy.spatial.distance import * #@UnusedWildImport
from sklearn.mixture import GMM
import pickle, scipy
from cluster import segmenter

EPS = 2.0**(-20.0)
NORM = stats.norm
CONFIDENCE_1_TO_10 = 0.926
CONFIDENCE_11_TO_20 = 0.969
CONFIDENCE_21_TO_50 = 1.021
CONFIDENCE_51_TO_100 = 1.047
CONFIDENCE_101_TO_500 = 1.092
CONFIDENCE_OVER_501 = 1.13
FEATURE_LIST = []

def FEATURE_LIST_export(file_path):
    f = open(file_path, 'w')
    pickle.dump(FEATURE_LIST, f)

def FEATURE_LIST_import(feature_list_dir):
    f = open(feature_list_dir, 'r')
    _feature_list = pickle.load(f)
    return _feature_list

def cluster_export(file_path, cluster_object):
    _save_file_path = file_path+cluster_object.name+'.clst' 
    f = open(_save_file_path, 'w')
    pickle.dump(cluster_object, f)
    
def cluster_import(cluster_object_dir):
    f = open(cluster_object_dir, 'r')
    _cluster = pickle.load(f)
    return _cluster

def _co_table(fclust, fclust2):
    ''' co-occurrences table
    '''
    _table = np.zeros((np.max(fclust),np.max(fclust2)))
    for i in range(len(fclust)):
        _table[fclust[i]][fclust2[i]] +=1
    return _table

def _a(f, f2):
    return (np.sum(_co_table(f, f2)**2) - 
            len(f))/2
    
def _b(f, f2):
    return (np.sum(np.sum(_co_table(f,f2),1)**2) - 
            np.sum(_co_table(f, f2)**2))/2

def _c(f, f2):
    return (np.sum(np.sum(_co_table(f,f2),0)**2) - 
            np.sum(_co_table(f, f2)**2))/2

def _d(f, f2):
    return (np.sum(_co_table(f, f2)**2) +
            len(f)**2 - 
            np.sum(np.sum(_co_table(f,f2),1)**2) - 
            np.sum(np.sum(_co_table(f,f2),0)**2))/2 
            
def ARI_ha(f, f2):
    N = len(f)
    a = _a(f,f2); b = _b(f,f2); c = _c(f,f2); d = _d(f,f2)
    return ((binomial_coeff(N,2)*(a+d) -
             (a+b)*(a+c) - 
             (c+d)*(b+d))/
            (binomial_coeff(N,2)**2 - 
             (a+b)*(a+c) - 
             (c+d)*(b+d)))
    
def binomial_coeff(n,k):
    return fac(n)/(fac(n-k)*fac(k))

def fac(n):
    if n <= 1:
        return 1
    else:
        return n*fac(n-1)

def split_centroid(obs, centroid):
    
    _centroid = centroid
    _obs = obs
    project, V = _pca(_obs)
    del _obs
    _centroid_one = _centroid + np.multiply(project.transpose()[0],np.sqrt((2.0*V[0])/np.pi))
    _centroid_two = _centroid - np.multiply(project.transpose()[0],np.sqrt((2.0*V[0])/np.pi))
    del _centroid 

    return np.vstack((_centroid_one,_centroid_two))

def _pca(X):
    # Principal Component Analysis
    # input: X, matrix with training data as flattened arrays in rows
    # return: projection matrix (with important dimensions first),
    # variance and mean

    # get dimensions
    _X = X
    try:
        num_data,dim = _X.shape
    except ValueError:
        num_data = 1
        dim = _X.shape[0]

    #center data
    _X = np.subtract(_X,np.mean(_X,0))
    
    _cov = np.cov(_X.transpose())
    _evalue,_evector = np.linalg.eigh(_cov)
    _argsorted = np.argsort(_evalue)[::-1]
    _sorted_evector = np.zeros(_evector.shape)
    _sorted_evalue = np.zeros(_evalue.shape)
    for i in range(len(_evalue)):
        _sorted_evector[:,i] = _evector[:,_argsorted[i]]
        _sorted_evalue[i] = _evalue[_argsorted[i]]
    return _sorted_evector,_sorted_evalue 

def get_child_obs(parent_obs, code_num, code):    
    _child_obs = []
    _count = 0
    for i in range(len(code_num)):
        if code_num[i] == code:
            _child_obs.insert(0,parent_obs[i])
            _count += 1
    _child_obs = np.array(_child_obs)  
    return _child_obs,_count 

def _get_BIC(obs, centroids, VERBOSE = False):
    _centroids = centroids
    _obs = obs
    try:
        _dim = _obs.shape[1]
        _R = _obs.shape[0]
    except IndexError:
        _dim = len(_centroids)
        _obs = np.array([obs])
        _R = _obs.shape[0]
        
    try:
        _dim = _centroids.shape[1]
        _K = _centroids.shape[0]
    except IndexError: 
        _dim = len(_centroids)
        _K = 1
        _centroids = np.array([centroids])
        
    _parameter_num = (_K-1.0) + _dim*_K + 1.0
                
    if _K == 1:
        if _R == 0:
            _tmp_BIC = 0
        else:
            _tmp_points_count = _R
            _tmp_squared_sum = np.sum(np.square(np.subtract(_obs,_centroids)))
            if _R == _K:
                _variance = 0
                _r = _R
                _log_likelihood = ((-_r/2.0)*np.log(2.0*np.pi)) - ((_r-_K)/2.0) + (_r*np.log(_r)) - (_r*np.log(_R)) 
            else:
                _variance = np.multiply((1.0/(_R-_K)),_tmp_squared_sum)
                _r = _R
                _log_likelihood = ((-_r/2.0)*np.log(2.0*np.pi)) - ((_r*_dim/2.0)*np.log(_variance)) \
                                    - ((_r-_K)/2.0) + (_r*np.log(_r)) - (_r*np.log(_R)) 
            _tmp_BIC = (_log_likelihood - (_parameter_num/2.0)*np.log(_R))
        if VERBOSE:
            print "_get_BIC(one class): ",_tmp_BIC
    else:
        _code,_distance = vq.vq(_obs,_centroids)
        del _obs,_centroids
        _tmp_squared_sum = 0
        _tmp_points_count = np.zeros((_K,1))
        _tmp_log_likelihood = np.zeros((_K,1))
        for i in range(_R):
            _tmp_squared_sum += np.square(_distance[i]) 
            _tmp_points_count[_code[i]] += 1
        if _R == _K:
            _variance = 0
            for i in range(_K):
                _r = _tmp_points_count[i]
                if _r == 0 :
                    _tmp_log_likelihood[i] = 0
                elif  _R == 0 :     
                    _tmp_log_likelihood[i] = 0
                else:
                    _tmp_log_likelihood[i] = ((-_r/2.0)*np.log(2.0*np.pi)) - ((_r-_K)/2.0) + (_r*np.log(_r)) - (_r*np.log(_R)) 
        else:
            _variance = np.multiply((1.0/(_R-_K)),_tmp_squared_sum)
            for i in range(_K):
                _r = _tmp_points_count[i]
                if _r == 0 :
                    _tmp_log_likelihood[i] = 0
                elif  _R == 0 :     
                    _tmp_log_likelihood[i] = 0
                else:
                    _tmp_log_likelihood[i] = ((-_r/2.0)*np.log(2.0*np.pi)) - ((_r*_dim/2.0)*np.log(_variance)) \
                                                - ((_r-_K)/2.0) + (_r*np.log(_r)) - (_r*np.log(_R)) 
        _log_likelihood = np.sum(_tmp_log_likelihood)
        _tmp_BIC = (_log_likelihood - (_parameter_num/2.0)*np.log(_R))
        if VERBOSE:
            print "_get_BIC(two class): ",_tmp_BIC
    return _tmp_BIC

def b_accept(obs, VERBOSE = False):
    _obs = obs
    _R = len(_obs)
    
    if VERBOSE == True:
        print 'number of obs:', _R
    _obs = np.divide(np.subtract(_obs,np.mean(_obs)),np.std(_obs))
    _sorted_obs = np.sort(_obs)        
    _score = 0.0 
    for i in range(_R):
        _score += np.multiply((2*(i+1)-1),(np.log(NORM.cdf(_sorted_obs[i]))+np.log(1-NORM.cdf(_sorted_obs[_R-1-i]))))
    _score = (-(_score)/_R - _R)*(1+(4/_R)-(25/(_R**2)))
    
    if _R <= 10:
        _CRITICAL = CONFIDENCE_1_TO_10    
    elif 10 < _R <= 20:
        _CRITICAL = CONFIDENCE_11_TO_20
    elif 20 < _R <= 50:
        _CRITICAL = CONFIDENCE_21_TO_50
    elif 50 < _R <= 100:
        _CRITICAL = CONFIDENCE_51_TO_100
    elif 100 < _R <= 500:
        _CRITICAL = CONFIDENCE_101_TO_500
    else: 
        _CRITICAL = CONFIDENCE_OVER_501
    
    return (_score < _CRITICAL)

def project_to_vector(data, vector):
    _data = data
    _vector = vector
    _lenth = np.sqrt(np.sum(np.square(_vector)))
    
    return [np.inner(_data[i],_vector)/_lenth for i in range(_data.shape[0])]
    
def KLD(u,v):
    u = u + EPS
    v = v + EPS
    u = np.divide(u,np.sum(u))
    v = np.divide(v,np.sum(v))
    return np.sum(np.multiply(u,np.log(np.divide(u,v)))) 

def JSD(u,v):
    return np.sqrt(0.5*KLD(u,0.5*np.add(u,v))+0.5*KLD(v,0.5*np.add(u,v)))

def _matrix_update(prev,current,p):
    p[prev][current] +=1 

def set_transition(charm_vec,HMM_vec):
    for i in range(len(charm_vec)):
        for j in range(len(charm_vec[i].list)):
            HMM_vec[i]._set_transition_hist(charm_vec[i].list[j].feature)
        HMM_vec[i]._set_transition()

def viterbi(n_iter, charm_vec, cluster_obj, HMM_vec, b):
    count = 0
    while(count < n_iter):
        for i in range(len(charm_vec)):
            for j in range(len(charm_vec[i].list)):
                _seq = HMM_vec[i]._viterbi(charm_vec[i].list[j].feature)
                HMM_vec[i]._update_transition_hist(_seq, update_init = b)
                print "Update with Viterbi: ",charm_vec[i].piece,',',charm_vec[i].list[j].performer,' ',charm_vec[i].list[j].date," - done."
                if count == (n_iter-1):
                    charm_vec[i].set_performance_label(j,_seq)
                    charm_vec[i].set_label_hist(j,cluster_obj.final_k)
                    FEATURE_LIST.extend(charm_vec[i].list[j].feature_labeled)
            if count == (n_iter-1):
                charm_vec[i].set_cluster_dist(cluster_obj.final_k)
        print "HMM learning with Viterbi progress: ",((float)(count+1)/n_iter)*100,"%."
                     
        HMM_vec[i]._update_transition(update_init = False)
        count +=1               
        
        
class spectral_clustering(object):        
    
    def __init__(self, obs, self_sim = None, metric = 'euclidean'):
        self.obs = obs
        self.N, self.dim  = obs.shape
        self.metric = metric
        if self_sim == None:
            self._get_self_sim()
        else:
            self.self_sim = self_sim
        
    def _get_self_sim(self):
        Y = pdist(self.obs, self.metric)
        self.self_sim = squareform(Y)
#         self.self_sim = scipy.exp(-A / np.std(Y)**2)
    
    def get_laplacian(self):
        # Get the inverse of the degree matrix 
        d_mat = np.sum(self.self_sim, axis = 1)**-1.0
        # Set infinite terms to 1.0
        d_mat[~np.isfinite(d_mat)] = 1.0
        # Use the square root of the degree matrix to set up the diagonal matrix
        d_mat = np.diag(d_mat**0.5)
        # Calculate the normalized graph laplacian
        self.laplacian = np.eye(self.N) - d_mat.dot(self.self_sim.dot(d_mat))
        
    def eigen_decompose(self):
        eigen_vals, eigen_vecs = scipy.linalg.eig(self.laplacian)
        eigen_vals = eigen_vals.real
        eigen_vecs = eigen_vecs.real
        ind = np.argsort(eigen_vals)
        
        eigen_vals = eigen_vals[ind]
        eigen_vecs = eigen_vecs[:,ind]
    
        self.eigen_vals = eigen_vals
        self.eigen_vecs = eigen_vecs.T
        
                
class X_means(object):
    
    default_init_k = 1
    default_limit_k = 512
    default_name = 'xmeans'
    
    valid_keywords = ['init_k',
                      'limit_k',
                      'name'
                      ]

    def __init__(self, obs, **keywords):
        
        self._obs = obs;
        self._dim = self._obs.shape[1]
        self._init_k = None
        self._limit_k = None
        self.final_k = None
        self.centroids = None
        self.name = None
         
        for key in keywords:
            if not key in X_means.valid_keywords:
                raise Hell("%s is not a valid keyword!"%key)

        if 'init_k' in keywords:
            self._init_k = keywords['init_k']
        else:
            self._init_k = X_means.default_init_k

        if 'limit_k' in keywords:
            self._limit_k = keywords['limit_k']
        else:
            self._limit_k = X_means.default_limit_k
            
        if 'name' in keywords:
            self.name = keywords['name']
        else:
            self.name = X_means.default_name
                    
        self.k = self._init_k;
        self.BIC = 0.0;
        _centroid = np.array([self._obs.mean(0)])
        self.centroids,self.code = vq.kmeans2(self._obs, _centroid ,minit = 'matrix')
        _obs = self._obs
        _centroid = self.centroids
        self.BIC = _get_BIC(_obs, _centroid)
        _current_k = self.k;
        _current_centroid = self.centroids;
        _current_BIC = self.BIC;
        while self.k <= self._limit_k:
            self.centroids = self.improve_struct()
            self.k = self.centroids.shape[0]
            _obs = self._obs
            _centroid = self.centroids
            self.BIC = _get_BIC(_obs, _centroid)
            if self.BIC > _current_BIC:
                _current_BIC = self.BIC
                _current_k = self.k
                _current_centroid = self.centroids 
                _obs = self._obs
                _centroid = self.centroids
                _tmp_centroids,self.code = vq.kmeans2(_obs,_centroid,minit = 'matrix')
                self.centroids = _tmp_centroids
            else:
                _obs = self._obs
                _centroid = self.centroids
                _tmp_centroids,self.code = vq.kmeans2(_obs,_centroid,minit = 'matrix')
                self.centroids = _tmp_centroids
                break
#            print _current_k
                
        self.centroids = _current_centroid
        self.final_k = _current_k 
        
        self.histogram = []
        for i in range(self.final_k):
            _obs = self._obs
            _code = self.code
            self.histogram.append(get_child_obs(_obs,_code,i)[1])
        
    def improve_struct(self):
        _tmp = self.centroids
        _K = _tmp.shape[0]
        _dim = _tmp.shape[1]
        
        if _K != 1:
            _first_centroid = _tmp[0]
        else:
            _first_centroid = _tmp
        
        _tmp_obs_self = self._obs
        _child_obs, _count = get_child_obs(_tmp_obs_self, self.code, 0) 
        
        if _count == 1:
            _centroids = np.array([_first_centroid])
        elif _count == 0:
            _centroids = np.array([])
        else:
            _tmp_obs = _child_obs
            _BIC = _get_BIC(_tmp_obs, _first_centroid)
            _first_centroid_two = split_centroid(_tmp_obs,_first_centroid)
            _tmp_obs_self = self._obs
            _child_obs, _count = get_child_obs(_tmp_obs_self, self.code, 0)
            _tmp_centroid_two,_tmp_code = vq.kmeans2(_child_obs,_first_centroid_two, minit = 'matrix')
            _tmp_obs = _child_obs
            _BIC2 = _get_BIC(_tmp_obs, _tmp_centroid_two)
            if _BIC2 > _BIC:
                _centroids = _tmp_centroid_two
            else:
                _centroids = np.array([_tmp[0]])
        
        for i in range(1,_K):
            _tmp_centroid = _tmp[i]
            _tmp_obs_self = self._obs
            _child_obs, _count = get_child_obs(self._obs, self.code, i)
            _tmp_obs = _child_obs
            if _count == 1:
                _centroids = np.insert(_centroids, 0, _tmp_centroid,0)
            else:
                _BIC = _get_BIC(_tmp_obs, _tmp_centroid) 
                _tmp_centroid_two = split_centroid(_tmp_obs,_tmp_centroid)
                _tmp_obs_self = self._obs
                _child_obs, _count = get_child_obs(_tmp_obs_self, self.code, i) 
                _tmp_new_centroid_two,_tmp_code = vq.kmeans2(_child_obs,_tmp_centroid_two,iter = 20 , minit = 'matrix')
                _tmp_obs = _child_obs
                _BIC2 = _get_BIC(_tmp_obs, _tmp_new_centroid_two)
                if _BIC2 > _BIC:
                    for i in range(_tmp_new_centroid_two.shape[0]):
                        _centroids = np.insert(_centroids, 0,_tmp_new_centroid_two[i],0)
                else:
                    _centroids = np.insert(_centroids, 0, _tmp_centroid,0)
        return _centroids        
    
class G_means(object):
    
    default_init_k = 1
    default_limit_k = 512
    default_name = 'gmeans'
    default_VERBOSE = False

    valid_keywords = ['init_k',
                      'limit_k',
                      'name',
                      'VERBOSE']

    def __init__(self, obs, **keywords):

        self._obs = obs
        self._init_k = None
        self._limit_k = None
        self.final_k = None
        self.centroids = None
        self.name = None
        
        for key in keywords:
            if not key in G_means.valid_keywords:
                raise Hell("%s is not a valid keyword!"%key)
            
        if 'init_k' in keywords:
            self._init_k = keywords['init_k']
        else:
            self._init_k = G_means.default_init_k

        if 'limit_k' in keywords:
            self._limit_k = keywords['limit_k']
        else:
            self._limit_k = G_means.default_limit_k
            
        if 'name' in keywords:
            self.name = keywords['name']
        else:
            self.name = G_means.default_name
            
        if 'VERBOSE' in keywords:
            self.VERBOSE = keywords['VERBOSE']
        else:
            self.VERBOSE = G_means.default_VERBOSE

        self.k = self._init_k;
        _obs = self._obs
        if self.k == 1:
            _centroid = np.array([self._obs.mean(0)])
            self.centroids,self.code = vq.kmeans2(_obs, _centroid ,minit = 'matrix')
        else:
            self.centroids,self.code = vq.kmeans2(_obs, self.k, minit = 'points')
        _centroid = self.centroids
        _current_k = self.k;
        _current_centroid = self.centroids;
        while self.k <= self._limit_k:
            self.centroids = self.improve_struct()
            self.k = self.centroids.shape[0]
            if self.k > _current_k:
                _current_k = self.k
                _current_centroid = self.centroids 
                _obs = self._obs
                _centroid = self.centroids                
                _tmp_centroids,self.code = vq.kmeans2(_obs,_centroid,minit = 'matrix')
                self.centroids = _tmp_centroids
            else:
                _obs = self._obs
                _centroid = self.centroids
                _tmp_centroids,self.code = vq.kmeans2(_obs,_centroid,minit = 'matrix')
                self.centroids = _tmp_centroids
                break
            if self.VERBOSE == True:
                print _current_k 

        self.centroids = _current_centroid
        self.final_k = _current_k 

        self.histogram = []
        for i in range(self.final_k):
            _obs = self._obs
            _code = self.code
            self.histogram.append(get_child_obs(_obs,_code,i)[1])
        
    def improve_struct(self):
        _tmp = self.centroids
        _K = _tmp.shape[0]
        _dim = _tmp.shape[1]
        
        if _K != 1:
            _first_centroid = _tmp[0]
            if self.VERBOSE == True:
                print 'improve_struct: get first centroid from centroids'
                print _first_centroid
        else:
            _first_centroid = _tmp
            if self.VERBOSE == True:
                print 'improve_struct: only one centroid'
                print _first_centroid
       
        _tmp_obs_self = self._obs
        _child_obs, _count = get_child_obs(_tmp_obs_self, self.code, 0) 
        
        if _count == 1:
            _centroids = _first_centroid
            _centroids = np.array([_tmp[0]])
            if self.VERBOSE == True:
                print 'improve_struct: only one point'
        elif _count == 0:
            _centroids = np.array([])
            if self.VERBOSE == True:
                print 'improve_struct: no centroid assigned'
        else:
            _tmp_obs = _child_obs            
            _first_centroid_two = split_centroid(_tmp_obs,_first_centroid)
            _tmp_obs_self = self._obs
            _child_obs, _count = get_child_obs(_tmp_obs_self, self.code, 0) 
            _tmp_centroid_two,_tmp_code = vq.kmeans2(_child_obs,_first_centroid_two, minit = 'matrix')
            _target_vector = np.subtract(_tmp_centroid_two[0],_tmp_centroid_two[1])
            _tmp_obs_self = self._obs
            _child_obs, _count = get_child_obs(_tmp_obs_self, self.code, 0) 
            _projected = project_to_vector(_child_obs, _target_vector)
            if b_accept(_projected, self.VERBOSE):
                _centroids = np.array([_tmp[0]])
            else:
                _centroids = _tmp_centroid_two
        
        for i in range(1,_K):
            _tmp_centroid = _tmp[i]
            if self.VERBOSE == True:
                print 'improve_struct: get '+ str(i) +'th centroid from centroids'
            _tmp_obs_self_b = self._obs
            _child_obs, _count = get_child_obs(_tmp_obs_self_b, self.code, i)
            if _count == 1:
                _centroids = np.insert(_centroids, 0, _tmp_centroid,0)
                if self.VERBOSE == True:
                    print 'improve_struct: only one point'
            elif _count == 0:
                _centroids = np.array([])  
                if self.VERBOSE == True:
                    print 'improve_struct: no centroid assigned'
            else:
                _tmp_obs = _child_obs
                _tmp_centroid_two = split_centroid(_tmp_obs,_tmp_centroid)
                _tmp_obs_self_b = self._obs
                _child_obs, _count = get_child_obs(_tmp_obs_self_b, self.code, i)
                _tmp_new_centroid_two,_tmp_code = vq.kmeans2(_child_obs,_tmp_centroid_two,iter = 20 , minit = 'matrix')
                _target_vector = np.subtract(_tmp_new_centroid_two[0],_tmp_new_centroid_two[1])
                _tmp_obs_self_b = self._obs 
                _child_obs, _count = get_child_obs(_tmp_obs_self_b, self.code, i)
                _projected = project_to_vector(_child_obs, _target_vector)
                if b_accept(_projected):
                    _centroids = np.insert(_centroids, 0, _tmp[i],0)
                    if self.VERBOSE == True:
                        print 'b_accept: no split'
                else:
                    for i in range(_tmp_new_centroid_two.shape[0]):
                        _centroids = np.insert(_centroids, 0, _tmp_new_centroid_two[i],0)
                    if self.VERBOSE == True:
                        print 'b_accept: centroids splited'
        if self.VERBOSE == True:
            print 'improve_struct: current centroids'
            print _centroids
        return _centroids       
    
class HMMClassfier(object):
    
    default_transition = None
    default_num_states = 32

    valid_keywords = ['_obs',
                      'num_states',
                      'transition',
                      'obs_dist',
                      'centroids',
                      'init_dist']

    def __init__(self, **keywords):
        
        self._obs = None
        self.transition_hist = None
        self.num_states = None
        self.obs_dist = None
        self.transition = None
        self.init_dist = None
        
        for key in keywords:
            if not key in HMMClassfier.valid_keywords:
                raise Hell("%s is not a valid keyword!"%key)
            
        if '_obs' in keywords:
            self._obs = keywords['_obs']
        else:
            raise Hell("Have to provide training data!!")
            
        if 'num_states' in keywords:
            self.num_states = keywords['num_states']
        else:
            self.num_states = HMMClassfier.default_num_states
        
        if 'transition' in keywords:
            self.transition = keywords['transition']
        else:
            self.transition = np.zeros((self.num_states,self.num_states))
        
        if 'centroids' in keywords:
            self._centroids = keywords['centroids']
        else:
            raise Hell("Have to provide cluster centroids!!")
        
        if 'obs_dist' in keywords:
            if type(keywords['obs_dist']) == GMM:
                self.obs_dist = keywords['obs_dist']
            else:
                raise Hell("obs_dist is not a sklearn.mixture.GMM object!")
        else:
            self.obs_dist = GMM(n_components = self.num_states, 
                                n_init = 10, 
                                params = 'wc',
                                init_params = 'wc')
            self.obs_dist.means_ = self._centroids
            self.obs_dist.fit(self._obs)

        if 'init_dist' in keywords:
            self.init_dist = keywords['init_dist']
        else:
            self.init_dist = np.zeros((1,self.num_states))
        
        self.transition_hist = np.zeros((self.num_states,self.num_states))
        self.transition_hist_new = np.zeros((self.num_states,self.num_states))
        self.init_dist = self.obs_dist.weights_
        self.init_dist_tmp = np.zeros((1,self.num_states))
        
    def _set_transition_hist(self, obs):
        '''
        Construct a transition histogram by reading in observation sequence.
        When an observation sequence is read, clusters are assigned based on
        predicions made by the estimated distribution of the whole observation
        set.
        '''

        _tmp_seq = self.obs_dist.predict(obs)
        _tmp_matrix = np.zeros((self.num_states,self.num_states))
        for i in range(len(_tmp_seq)-1):
            _prev = _tmp_seq[i]
            _current = _tmp_seq[i+1]
            _matrix_update(_prev, _current, _tmp_matrix)
        self.transition_hist = self.transition_hist + _tmp_matrix  
    
    def _set_transition(self):
        '''Transform the transition histogram into the transition probability matrix'''    

        for i in range(self.transition.shape[0]):
            for j in range(self.transition.shape[1]):
                if self.transition_hist[i][j] == 0:
                    self.transition_hist[i][j] = EPS
                     
        for rnum in range(self.transition.shape[0]):            
            self.transition[rnum][:] = self.transition_hist[rnum][:]/sum(self.transition_hist[rnum][:])

    def _viterbi(self,obs):
        '''
        Use the Viterbi algorithm to re-estimate the clusters assigned to the observation
        sequences.
        '''    

        _viterbi_mat = np.zeros((obs.shape[0],self.num_states))
        _path = np.zeros((obs.shape[0],self.num_states)) 
        _init_dist = np.log(self.init_dist)
        _viterbi_mat[0] = (np.log(self.obs_dist.predict_proba(obs[0]) + EPS) + _init_dist)[0]
        _path[0] = np.argmax(_viterbi_mat[0])
        
        for i in range(1,obs.shape[0]):
            for j in range(self.num_states):
                _eval = (_viterbi_mat[i-1] + 
                         np.log(self.transition.transpose()[j]) + 
                         np.log(self.obs_dist.predict_proba(obs[i])[0][j] + 
                                EPS))
                _viterbi_mat[i][j] = np.max(_eval)
#                _path[i][j] = np.argmax(_eval)
                _path[i-1][j] = np.argmax(_eval)
        

#        for j in range(self.num_states):
        _path[-1] = np.argmax(_viterbi_mat[-1])
        _optimal = _path.transpose()[np.argmax(_viterbi_mat[-1])]
        return _optimal #,_viterbi_mat,_path
    
    def _update_transition_hist(self, seq, update_init = False):
        '''
        Update then create a new transition histogram based on the result
        from the Viterbi algorithm.
        '''

        _tmp_seq = seq
        _tmp_matrix = np.zeros((self.num_states,self.num_states))
        for i in range(len(_tmp_seq)-1):
            _prev = _tmp_seq[i]
            _current = _tmp_seq[i+1]
            _matrix_update(_prev, _current, _tmp_matrix)
            if update_init == True:
                if i == 0:
                    self.init_dist_tmp[0][_prev] +=1
        self.transition_hist_new = self.transition_hist_new + _tmp_matrix             
                
    def _update_transition(self, update_init = False):
        for i in range(self.transition.shape[0]):
            for j in range(self.transition.shape[1]):
                if self.transition_hist_new[i][j] == 0:
                    self.transition_hist_new[i][j] = EPS
                     
        for rnum in range(self.transition.shape[0]):            
            self.transition[rnum][:] = self.transition_hist_new[rnum][:]/sum(self.transition_hist_new[rnum][:])
        
        if update_init == True:
            for i in range(self.num_states):
                if self.init_dist_tmp[0][i] == 0:
                    self.init_dist_tmp[0][i] = EPS
            self.init_dist_tmp = self.init_dist_tmp/sum(self.init_dist_tmp)
            self.init_dist = self.init_dist_tmp
            self.init_dist_tmp = np.zeros((1,self.num_states))      
        

class Hell(BaseException):
    pass
