import pandas as pd
import anndata
import numpy as np 
import scipy.sparse
np.random.seed(1234)
def generate_bulk_AnnData(bulk_df):
    '''
    Generate AnnData object from DataFrame.

    Parameters:
    
    - bulk_df (DataFrame): Dataframe of prototype data:
        - columns: cell types / samples
        - rows: featues (cCREs)
        
    Returns:
    
    - AnnData of prototypes.
        
    '''
    features = pd.DataFrame(bulk_df.index)
    features.columns=['cCREs']
    features.index = features['cCREs']
    cell_types = pd.DataFrame(bulk_df.columns)
    cell_types.columns=['cell_types']
    cell_types.index = cell_types['cell_types']
    bulk_complete_adata = anndata.AnnData(np.array(bulk_df.T, dtype= np.float32), var=features, obs=cell_types)
    return bulk_complete_adata

def generate_bulk_sparse_AnnData(bulk_df, var_key='cCREs', obs_key='cell_types'):
    '''
    Generate AnnData object from DataFrame. The count matrix is sparse. 

    Parameters:
    
    - bulk_df (DataFrame): Dataframe of prototype data:
        - columns: cell types / samples
        - rows: featues (cCREs)
        
    Returns:
    
        - AnnData of prototypes.
        
    '''
    features = pd.DataFrame(bulk_df.index)
    features.columns=[var_key]
    features.index = features[var_key]
    cell_types = pd.DataFrame(bulk_df.columns)
    cell_types.columns=[obs_key]
    cell_types.index = cell_types[obs_key]
    bulk_csr = scipy.sparse.csr_matrix(np.matrix(bulk_df.T.values,dtype=np.float32))
    bulk_complete_adata = anndata.AnnData(bulk_csr, var=features, obs=cell_types)
    return bulk_complete_adata


def preprocess_bulk_adata(bulk_adata, remove_chrY=True, var_key = 'cCREs', copy=False):
    '''
    Preprocess a prototype count matrix in AnnData format.
    This function preprocesses a prototype count matrix in AnnData format by optionally removing features associated with chromosome Y.

    If `copy` is True, a new AnnData object with the preprocessed data is returned, leaving the original AnnData object unchanged.
    If `copy` is False, the original AnnData object is modified in place, and the preprocessed AnnData object is returned.
    
    Parameters:
    
    - bulk_adata (AnnData): An AnnData object containing the prototype count matrix.
    - remove_chrY (bool, optional): Whether to remove features associated with chromosome Y. Default is True.
    - var_key (str, optional): Key for accessing feature information in AnnData.var. Default is 'cCREs'.
    - copy (bool, optional): If True, a copy of the AnnData object is returned; if False, the original AnnData object is modified. Default is False.

    Returns:
    
    - AnnData: The preprocessed AnnData object.
    '''
    feature_coverage = bulk_adata.X.sum(0)
    if copy==False:
        bulk_adata = bulk_adata[:,feature_coverage>0]
        if remove_chrY==True:
            filtered_chrY= [a for a in bulk_adata.var_names if not a.startswith("chrY")]
            bulk_adata=bulk_adata[:,sorted(filtered_chrY)]
        return bulk_adata
    else:
        bulk_adata2 = bulk_adata[:,feature_coverage>0].copy()
        if remove_chrY==True:
            filtered_chrY= [a for a in bulk_adata2.var_names if not a.startswith("chrY")]
            bulk_adata2=bulk_adata2[:,sorted(filtered_chrY)]
        return bulk_adata2
