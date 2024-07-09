import copy
import anndata
import scipy.sparse
import numpy as np
import scanpy as sc
import numba
import logging as logg
from . import plot_functions
np.random.seed(1234)
def add_binary_layer(adata, binary_layer_key="binary"):
    '''
    Convert the count matrix associated with the AnnData object to binary and adds it as a new layer.

    This function converts the count matrix in the AnnData object to binary, where non-zero values are set to 1.
    
    The resulting binary matrix is added as a new layer in the AnnData object using the specified key.

    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - binary_layer_key (str, optional): The key for the binary layer to be added. Default is "binary".

    Returns:
    
    - AnnData: The AnnData object with the binary layer added.

    '''
    
    if not 'binary' in adata.layers:
        binary_matrix = copy.deepcopy(adata.X)
        binary_matrix[binary_matrix != 0] = 1
        adata.layers["binary"] = binary_matrix
    return adata


def preproces_sc_matrix(adata,cell_cutoff=1000, cell_cutoff_max=80000, feature_cutoff=3, remove_chrY = True, var_key = 'cCREs', copy=False):

    '''
    Preprocess a sc count matrix in AnnData format.
    
    This function preprocesses a single-cell count matrix in AnnData format by applying the following steps:
    
    1. Filters cells based on the number of features per cell using the specified cutoffs.
    
    2. Filters features based on the number of cells per feature using the specified cutoff.
    
    3. Optionally removes features associated with chromosome Y.
    
    If `copy` is True, a new AnnData object with the preprocessed data is returned, leaving the original AnnData object unchanged.
    If `copy` is False, the original AnnData object is modified in place, and the preprocessed AnnData object is returned.

    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - cell_cutoff (int, optional): Minimum number of features required per cell. Default is 1000.
    - cell_cutoff_max (int, optional): Maximum number of features allowed per cell. Default is 80000.
    - feature_cutoff (int, optional): Minimum number of cells required per feature. Default is 3.
    - remove_chrY (bool, optional): Whether to remove features associated with chromosome Y. Default is True.
    - var_key (str, optional): Key for accessing feature information in AnnData.var. Default is 'cCREs'.
    - copy (bool, optional): If True, a copy of the AnnData object is returned; if False, the original AnnData object is modified. Default is False.

    Returns:
    
    - AnnData: The preprocessed AnnData object.
    '''
    
    if  'num_cell_per_feature' not in adata.varm_keys() and 'num_feature_per_cell' not in adata.obs_keys():
        cell_feature_statistics(adata)
    if copy==False:
        adata = adata[adata.obsm['num_feature_per_cell'] > cell_cutoff]
        adata = adata[adata.obsm['num_feature_per_cell'] < cell_cutoff_max]
        adata = adata[:,adata.varm['num_cell_per_feature'] > feature_cutoff]
        if remove_chrY==True:
            filtered_chrY= [a for a in adata.var_names if not a.startswith("chrY")]
            adata=adata[:,filtered_chrY]
        return adata
    else:
        adata2 = adata[adata.obsm['num_feature_per_cell'] > cell_cutoff].copy()
        adata2 = adata2[adata2.obsm['num_feature_per_cell'] < cell_cutoff_max]
        adata2 = adata2[:,adata2.varm['num_cell_per_feature'] > feature_cutoff]
        if remove_chrY==True:
            filtered_chrY= [a for a in adata2.var_names if not a.startswith("chrY")]
            adata2=adata2[:,filtered_chrY]
        return adata2
    

def overlap_vars(adata1, adata2):
    '''
    Generic function to get shared variables between two AnnData objects.

    Parameters:
    
    - adata1 (AnnData): An AnnData object containing the sc count matrix.
    - adata2 (AnnData): An AnnData object containing the sc count matrix.

    Returns:
    
    - List: List of shared variables. 
    
    '''
    common_vars = sorted(list(set(list(adata1.var_names)) & set(list(adata2.var_names))))
    return common_vars

# filter adata by var or obs list

def subset_adata_vars(adata, vars_list, copy_=True):
    '''
    Subset the variables (features) of an AnnData object based on a specified list.

    The resulting AnnData object includes only the variables specified in the `vars_list`.

    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - vars_list (list): A list of variable names (features) to retain in the subset.
    - copy_ (bool, optional): If True, a `copy` of the AnnData object is returned; if False, the original AnnData object is modified. Default is True.

    Returns:
    
    - AnnData: The AnnData object with a subset of variables.
    '''
    if copy_==False:
        adata = adata[:, sorted(vars_list)]
        return adata
    else:
        adata2 = adata[:, sorted(vars_list)].copy()
        return adata2
        
def subset_adata_obs(adata, obs_list, copy_=True):
    '''
    This function subsets the observations (cells) of the AnnData object based on the provided list.
    
    The resulting AnnData object includes only the variables specified in the `vars_list`.

    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - vars_list (list): A list of observations names (cells) to retain in the subset.
    - copy_ (bool, optional): If True, a `copy` of the AnnData object is returned; if False, the original AnnData object is modified. Default is True.

    Returns:
    
    - AnnData: The AnnData object with a subset of variables.
    '''
    if copy_==False:
        adata = adata[sorted(obs_list)]
        return adata
    else:
        adata2 = adata[sorted(obs_list)].copy()
        return adata2


def apply_TFIDF_sparse(adata, binary_layer_key='binary', TFIDF_key='TF_logIDF' ):
    '''
    Apply Term Frequency - Inverse Document Frequency TF-log(IDF) normalization to the binary layer of the AnnData object.
    
    If the binary layer is not present, it calculates and adds the binary layer using the specified key.
    
    Additionally, if cell and feature statistics are not available, it calculates them using the binary layer.

    Parameters:
    
    - adata (AnnData):  An AnnData object containing the sc count matrix.
    - binary_layer_key (str): The key for accessing the binary layer. Default is "binary".
    - TFIDF_key (str): The key for the TFIDF normalized matrix layer to be added. Default is "TF_logIDF".

    Returns:
    
    - AnnData: The AnnData object with the TF-log(IDF) normalized layer added.
    '''
    if not binary_layer_key in adata.layers:
        add_binary_layer(adata, binary_layer_key=binary_layer_key)
    if  'num_cell_per_feature' not in adata.varm_keys() and 'num_feature_per_cell' not in adata.obs_keys():
        print("calculating cell and feature statistics")
        cell_feature_statistics(adata, binary_layer_key =binary_layer_key)
    
    TF= adata.layers[binary_layer_key].T.tocsr() * scipy.sparse.diags(1/adata.obsm['num_feature_per_cell'])
    IDF = adata.n_vars / adata.varm['num_cell_per_feature']
    TF_logIDF = TF.multiply(np.log(1+IDF.reshape(IDF.shape[0],1)))
    TF_logIDF_t= TF_logIDF.transpose().tocsr()
    adata.layers[TFIDF_key] = TF_logIDF_t
    return adata

#########
# wrapper function for scanpy's pca

def apply_PCA(adata, layer_key = "TF_logIDF", svd_solver = 'arpack', random_state=0):
    '''
    Wrapper around scanpy.tl.pca to enable applying scanpy.tl.pca function to a specified layer.
    
    Adds the ``X_pca``, ``PCs`` (components), ``variance``, and ``variance_ratio`` to the adata object.

    See scanpy documentation for details: https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.pca.html#scanpy-tl-pca
    
    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - layer_key (str): The key for accessing the layer to which PCA is applied. Default is "TF_logIDF". 

    Returns:
    
    - AnnData: The AnnData object with the PCA results added.
    '''
    pca__ = sc.tl.pca(adata.layers[layer_key], svd_solver= svd_solver, return_info=True,random_state=0)
    # define ins annotations for pca
    adata.uns['pca'] = {}
    # X_pca
    adata.obsm['X_pca'] = pca__[0]
    # components_
    adata.varm['PCs'] = pca__[1].T
    # explained_variance_
    adata.uns['pca']['variance'] = pca__[2] 
    # explained_variance_ratio_
    adata.uns['pca']['variance_ratio'] = pca__[3] 
    
    return adata


#################################################
# normalization 
def preprocessing_libsize_norm_log2(adata):
    '''
    Perform libray-size normalization & log2 transformation on the Anndata object.
    
    Normalized and log2 transformed matrix is added as a layer with keyword "libsize_norm_log2". 

    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.

    Returns:
    
    - AnnData: The AnnData object with the libsize_norm_log2 normalized layer added.

    '''
    libsize = adata.X.sum(1)
    countdata_matrix = adata.X.todense()
    adata_X_libsize_normalized_log2 = np.log2((countdata_matrix/libsize *1000000)+1)
    adata.layers["libsize_norm_log2"] = np.array(adata_X_libsize_normalized_log2)
    return adata


# this function is copied from scanpy 
def _get_mean_var(X, *, axis=0):
    if scipy.sparse.issparse(X):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = np.mean(X, axis=axis, dtype=np.float64)
        mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var

# this function is copied from scanpy 

def sparse_mean_variance_axis(mtx: scipy.sparse.spmatrix, axis: int):
    """
    This code and internal functions are based on sklearns
    `sparsefuncs.mean_variance_axis`.

    Modifications:
    
    * allow deciding on the output type, which can increase accuracy when calculating the mean and variance of 32bit floats.
    * This doesn't currently implement support for null values, but could.
    * Uses numba not cython
    """
    assert axis in (0, 1)
    if isinstance(mtx, scipy.sparse.csr_matrix):
        ax_minor = 1
        shape = mtx.shape
    elif isinstance(mtx, scipy.sparse.csc_matrix):
        ax_minor = 0
        shape = mtx.shape[::-1]
    else:
        raise ValueError("This function only works on sparse csr and csc matrices")
    if axis == ax_minor:
        return sparse_mean_var_major_axis(
            mtx.data, mtx.indices, mtx.indptr, *shape, np.float64
        )
    else:
        return sparse_mean_var_minor_axis(mtx.data, mtx.indices, *shape, np.float64)


def preprocessing_standardization(adata, input_layer_key="libsize_norm_log2", output_layer_key = "libsize_norm_log2_std", std_key= None,  mean_key=None, std_ = None, mean_= None, zero_center=True):
    '''
    Perform z-normalization at the feature level. If the standard deviation (``std``) and ``mean`` are already included in the AnnData (adata), the function applies normalization directly. In the absence of these variables, it calculates and adds the standard deviation and mean to the AnnData using the specified layer key (layer_key). Subsequently, it performs z-normalization.

    Additionally, if alternative ``std_`` and ``mean_`` matrices/arrays are provided, these values are utilized for the calculations instead of assuming zero mean and unit variance.

    Parameters:
   
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - input_layer_key (str): The key for accessing the layer to which standardization is applied. Default is "libsize_norm_log2".
    - output_layer_key (str): The key for the standardized layer to be added. Default is "libsize_norm_log2_std".
    - std_key (str): The key for the standard deviation to be added. If None, ``feature_std`` is added as key.
    - mean_key (str): The key for the ``mean`` to be added. If None, ``feature_mean`` is added as key.
    - std_ (numpy array): The key for accessing the standard deviation. If specified, it is utilized for the z-score calculations instead of assuming zero mean and unit variance. Default is None.
    - mean_ (numpy array): The key for accessing the ``mean``. If specified, it is utilized for the z-score calculations instead of assuming zero mean and unit variance. Default is None.

    Returns:
   
    - AnnData: The AnnData object with the libsize_norm_log2_std standardized layer added.

    '''
    try:

        print(adata.var[std_key])

    except (KeyError,AttributeError) as e :

        print("adding std with default keywords")
    try:

        print(adata.var[mean_key])

    except (KeyError, AttributeError) as e:

        print("adding mean with default keywords")
              
    if std_key is None:
        std_key = "feature_std"
    if mean_key is None:
        mean_key = "feature_mean"
    if input_layer_key is not None:
        if output_layer_key is None:
            output_layer_key = str(input_layer_key) + "_std"
        
        if np.issubdtype(adata.layers[input_layer_key].dtype, np.integer):
            logg.info(
                '... as scaling leads to float results, integer '
                'input is cast to float, returning copy.')
            adata.layers[input_layer_key] = adata.layers[input_layer_key].astype(float)

        mean, var = _get_mean_var(adata.layers[input_layer_key])
        std = np.sqrt(var)
        std[std == 0] = 1
        adata.var[std_key] = std
        adata.var[mean_key] = mean

        if zero_center:
            scaled_X  = (adata.layers[input_layer_key] -  mean) 
            scaled_X /= std
        else:
            logg.info(
            '... using the user given mean and std')
            if mean_ is None or std_ is None:
                print("mean_ or std_ is not given as an input "
                     "activating zero scaling ")
                scaled_X  = (adata.layers[input_layer_key] -  mean) 
                scaled_X /= std


            scaled_X  = (adata.layers[input_layer_key] -  np.array(mean_)) 
            std_ = np.array(std_)
            std_[std_ == 0] = 1
            scaled_X /=  np.array(std_)
    else:
        if output_layer_key is None:
            output_layer_key = "std_matrix"

        if np.issubdtype(adata.X.dtype, np.integer):
            logg.info(
                '... as scaling leads to float results, integer '
                'input is cast to float, returning copy.')
            adata.layers[input_layer_key] = adata.layers[input_layer_key].astype(float)

        mean, var = _get_mean_var(X)
        std = np.sqrt(var)
        std[std == 0] = 1
        adata.var[std_key] = mean
        adata.var[mean_key] = std

        if zero_center:
            scaled_X  = (adata.X -  mean) 
            scaled_X /= std
        else:
            logg.info(
            '... using the user given mean and std')
            if mean_ is None or std_ is None:
                print("mean_ or std_ is not given as an input "
                     "activating zero scaling ")
                scaled_X  = (adata.X -  mean) 
                scaled_X /= std

            scaled_X  = (adata.layers[input_layer_key] -  np.array(mean_))
            std_ = np.array(std_)
            std_[std_ == 0] = 1
            scaled_X /=  np.array(std_)
        
    adata.layers[output_layer_key] = scaled_X
    return adata    
