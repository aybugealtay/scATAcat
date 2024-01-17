import scanpy as sc
import scipy.sparse
import pandas as pd
import anndata
import numpy as np 
import collections
np.random.seed(1234)


def get_pseudobulk_matrix(adata, cluster_key = 'leiden', method = 'sum'):
    '''
    Constructs pseudobulk by features matrix given the cluster key. 
    
    Parameters:
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - cluster_key (str, optional): The key for the cluster key from which the pseudobulk matrix is constructed. Default is "leiden".
    - method: method to aggregate the cells:
        - sum: sums the feature counts across cells
        - mean: takes the mean of the feature counts across cells
        
    Returns:
    - Pandas dataframe in th shape of pseudobulk x feature. 
        
    '''
    pseudobulk_by_feature_df = pd.DataFrame(index =  np.squeeze(np.asarray(adata.var.values)))
    #iterate over the clusteter ids, subset the matrix per clusters, and sum/mean/median the values
    for clust_id in set(adata.obs[cluster_key].values):
        if method == "sum":
            rname = "clust_" + str(clust_id)
            adata.X[adata.obs[cluster_key] == clust_id].sum(0)
            pseudobulk_by_feature_df[rname]= np.squeeze(np.asarray(adata.X[adata.obs[cluster_key] == clust_id].sum(0)))   
        if method == "mean":
            rname = "clust_" + str(clust_id)
            adata.X[adata.obs[cluster_key] == clust_id].sum(0)
            pseudobulk_by_feature_df[rname]= np.squeeze(np.asarray(adata.X[adata.obs[cluster_key] == clust_id].mean(0)))
            
    return pseudobulk_by_feature_df

def get_pseudobulk_matrix_ext(adata_to_subset, adata_to_get_clusters, cluster_key="leiden" , method = 'sum'):
    '''
    Constructs pseudobulk by features matrix given the cluster key and AnnData objects. 
    
    Parameters:
    - adata_to_subset (AnnData): An AnnData object containing the sc count matrix.
    - adata_to_get_clusters (AnnData): An AnnData object containing the clusterong information for give  cluster_key.
    - cluster_key (str, optional): The key for the cluster key from which the pseudobulk matrix is constructed. Default is "leiden".
    - method: method to aggregate the cells:
        - sum: sums the feature counts across cells
        - mean: takes the mean of the feature counts across cells
        
    Returns:
    - Pandas dataframe in th shape of pseudobulk x feature. 
        
    '''
    pseudobulk_by_feature_df = pd.DataFrame(index =  np.squeeze(np.asarray(adata_to_subset.var.values)))
    #iterate over the clusteter ids, subset the matrix per clusters, and sum/mean/median the values
    for clust_id in set(adata_to_get_clusters.obs[cluster_key].values):
        if method == "sum":
            rname = "clust_" + str(clust_id)
            adata_to_subset.X[adata_to_get_clusters.obs[cluster_key] == clust_id].sum(0)
            pseudobulk_by_feature_df[rname]= np.squeeze(np.asarray(adata_to_subset.X[adata_to_get_clusters.obs[cluster_key] == clust_id].sum(0)))   
        if method == "mean":
            rname = "clust_" + str(clust_id)
            adata_to_subset.X[adata_to_get_clusters.obs[cluster_key] == clust_id].sum(0)
            pseudobulk_by_feature_df[rname]= np.squeeze(np.asarray(adata_to_subset.X[adata_to_get_clusters.obs[cluster_key] == clust_id].mean(0)))
            
    return pseudobulk_by_feature_df

def get_closest_prototype_to_pseudobulk(pseudobulk_prototype_centroid_euclidean_dis_df):
    '''
    Calculates the distances between prototypes and pseudobulks and returns the closest prototype to a pseudobulk.
    
    Paramaters:
    - pseudobulk_prototype_centroid_euclidean_dis_df (Pandas Dataframe): square matrix of pairwise distances between prorootype centroids and pseudobulk samples. Can be obtained by running plot_pca_dist_cent_heatmap() function.

    Returns:
     - {pseudobulk:closest_prototype} dictionary  
    '''

    # define a dict to keep pbulk: closest bulk pairs
    pbulk_closest_prototype_dict = {}
    # we hardcoded "pbulk" suffix for centorid matrix. now first subset the df to pbulks
    for pbulk in pseudobulk_prototype_centroid_euclidean_dis_df[pseudobulk_prototype_centroid_euclidean_dis_df.columns.str.endswith("pbulk") == True].index:
        # order by the smallest distance
        smallest_dist_ordered= pseudobulk_prototype_centroid_euclidean_dis_df.nsmallest(pseudobulk_prototype_centroid_euclidean_dis_df.shape[0], pbulk)
        # get the closest protoype that is not pbulk
        closest_prototype_id = smallest_dist_ordered.iloc[smallest_dist_ordered.index.str.endswith("pbulk") == False,:].index[0]
        pbulk_closest_prototype_dict[pbulk]=closest_prototype_id
        
    return(dict(sorted(pbulk_closest_prototype_dict.items())))

def get_closest_pseubulk_to_prototype(pseudobulk_prototype_centroid_euclidean_dis_df):
    '''
    Calculates the distances between pseudobulks and prototypes and returns the closest pseudobulk to a prototype.
        
    Paramaters:
    - pseudobulk_prototype_centroid_euclidean_dis_df (Pandas Dataframe): square matrix of pairwise distances between prorootype centroids and pseudobulk samples. Can be obtained by running plot_pca_dist_cent_heatmap() function.
    
    Returns:
        - {prototype:closest_pseudobulk} dictionary  
    '''

    # define a dict to keep pbulk: closest bulk pairs
    prototype_closest_pbulk_dict = {}
    # we hardcoded "pbulk" suffix for centorix matrix. now first subset the df to pbulks
    for prototype in pseudobulk_prototype_centroid_euclidean_dis_df[pseudobulk_prototype_centroid_euclidean_dis_df.columns.str.endswith("pbulk") == False].index:
        # order by the smallest distance
        smallest_dist_ordered= pseudobulk_prototype_centroid_euclidean_dis_df.nsmallest(pseudobulk_prototype_centroid_euclidean_dis_df.shape[0], prototype)
        # get the closest prototype that is not pbulk
        closest_pbulk_id = smallest_dist_ordered.iloc[smallest_dist_ordered.index.str.endswith("pbulk") == True,:].index[0]
        prototype_closest_pbulk_dict[bulk]=closest_pbulk_id
        
    return(dict(sorted(prototype_closest_pbulk_dict.items())))

def get_pseudobulk_to_prototype_distance(pseudobulk_prototype_centroid_euclidean_dis_df, pbulk_to_prototype=True):
    '''
    Transfers Euclidean distances to scaled similarities based on pseudobulk and bulk samples' perspectives.
    This function takes a square matrix of pairwise Euclidean distances between bulk centroids and pseudobulk samples.
    It then scales the distances to the minimum and returns a DataFrame representing the percentage contributions for each sample.
    
    Parameters:
    - pseudobulk_prototype_centroid_euclidean_dis_df (DataFrame): A square matrix of pairwise distances between bulk centroids and pseudobulk samples.
    - pbulk_to_prototype (bool, optional): If True, the distances are determined by the prorotypes' perspective.
                                           If False, the distances are determined by the pseudobulk samples' perspective. Default is True.

    '''
    if pbulk_to_prototype==False:
        per_cont_df = pd.DataFrame()
        per_cont_df.index = pseudobulk_prototype_centroid_euclidean_dis_df[pseudobulk_prototype_centroid_euclidean_dis_df.columns.str.endswith("pbulk") == True].index
        for prototype in pseudobulk_prototype_centroid_euclidean_dis_df[pseudobulk_prototype_centroid_euclidean_dis_df.columns.str.endswith("pbulk") == False].index:
            sub_df =  pseudobulk_prototype_centroid_euclidean_dis_df[[prototype]].iloc[pseudobulk_prototype_centroid_euclidean_dis_df.index.str.endswith("pbulk") == True,:]
            sub_df_scaled = sub_df.max() -sub_df
            sub_df_scaled_perc = sub_df_scaled.div(sub_df_scaled.sum(axis=0), axis=1)*100
            per_cont_df = pd.concat([per_cont_df,sub_df_scaled_perc], axis=1, join="inner")
    if pbulk_to_prototype ==True:
        per_cont_df = pd.DataFrame()
        per_cont_df.index = pseudobulk_prototype_centroid_euclidean_dis_df[pseudobulk_prototype_centroid_euclidean_dis_df.columns.str.endswith("pbulk") == False].index
        for prototype in pseudobulk_prototype_centroid_euclidean_dis_df[pseudobulk_prototype_centroid_euclidean_dis_df.columns.str.endswith("pbulk") == True].index:
            sub_df =  pseudobulk_prototype_centroid_euclidean_dis_df[[prototype]].iloc[pseudobulk_prototype_centroid_euclidean_dis_df.index.str.endswith("pbulk") == False,:]
            sub_df_scaled = sub_df.max() -sub_df
            sub_df_scaled_perc = sub_df_scaled.div(sub_df_scaled.sum(axis=0), axis=1)*100
            per_cont_df = pd.concat([per_cont_df,sub_df_scaled_perc], axis=1, join="inner")
    return per_cont_df