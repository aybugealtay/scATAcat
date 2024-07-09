import numpy as np 
import matplotlib 
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.decomposition import PCA
import sklearn
import pandas as pd
import anndata
from scipy.spatial.distance import cdist, pdist, squareform
import seaborn as sns; sns.set_theme(color_codes=True)
import scanpy as sc
from mycolorpy import colorlist as mcp
np.random.seed(1234)

def cell_feature_statistics(adata, binary_layer_key ='binary'):
    '''
    Calculates the cell and feature statistics and adds them to AnnData.
        
    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - binary_layer_key (str, optional): The key for accessing the layer to calculates the cell and feature statistics, Default 'binary'.

    Returns:
    
    AnnData with the following features:
        - num_cell_per_feature: how many cells have a count for a feature? / number of cells sharing a feature
        - num_feature_per_cell : how many features are open for a cell? / number of features in a cell
    
    '''
    if not binary_layer_key in adata.layers:
        add_binary_layer(adata, binary_layer_key=binary_layer_key)
    num_cell_per_feature = adata.layers[binary_layer_key].sum(0)
    num_feature_per_cell = adata.layers[binary_layer_key].sum(1)
    # cast the results to array to match the original matrix size
    adata.varm['num_cell_per_feature'] = np.squeeze(np.asarray(num_cell_per_feature))
    adata.obsm['num_feature_per_cell'] = np.squeeze(np.asarray(num_feature_per_cell))
    return adata
    
def plot_cell_statistics(adata, binary_layer_key ='binary', color=None, edgecolor=None, bins=None, xlabel=None, ylabel=None, title=None, threshold=None, save=False, save_dir=None, dpi=300):
    '''
    Plots the cell statistics. In simpler terms, this method shows how densely features are occupied by cells.
    
    It provides a visual representation of the distribution and concentration of these cells within the features.

    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - binary_layer_key (str, optional): The key for accessing the layer to calculate the cell and feature statistics, Default 'binary'.
    - save (bool, optional): Whether or not to save the figure, Default False. 
    - save_dir (str, optional): Directory to sace the figure. Default None and saves to current directory. 
    - dpi (str, optional): resolution of the figure to save in dots per inch
    - kwds
        Are passed to :func:`matplotlib.pyplot.hist`.

    Returns:
    
    - AnnData obejct with following features: 
        - num_cell_per_feature (int): how many cells have a count for a feature? / number of cells sharing a feature
        - num_feature_per_cell : how many features are open for a cell? / number of features in a cell
        
    - Cell statistic plot

    '''
    sns.set_style('whitegrid')
    
    if  'num_cell_per_feature' not in adata.varm_keys() and 'num_feature_per_cell' not in adata.obs_keys():
        print("calculating cell and feature statistics")
        cell_feature_statistics(adata, binary_layer_key =binary_layer_key)
    # plotting settings
    if xlabel ==None:
        plt.xlabel('cells')
    else:
        plt.xlabel(xlabel)

    if ylabel ==None:
        plt.ylabel('frequency of features')
    else:
        plt.ylabel(ylabel)

    if title !=None:
        plt.title(title)

    if color == None:
        color='c'
    if edgecolor == None:
        edgecolor='k'

    if bins == None:
        bins=50
    
    fig = plt.hist(adata.obsm['num_feature_per_cell'], bins, color=color, edgecolor=edgecolor)

    if threshold != None:
        plt.axvline(x=threshold, color='r', linestyle='--')
        
    if save == True:
        if save_dir is not None:
            plt.savefig(save_dir, dpi=dpi)
        else:
            plt.savefig("./",dpi=dpi)
    return adata



def plot_feature_statistics(adata, binary_layer_key ='binary', color=None, edgecolor=None, bins=None, xlabel=None, ylabel=None, title=None, threshold=None, save=False, save_dir=None, dpi=300, fig_size_inches= (15,15)):
    '''
    Plots the feature statistics. In simpler terms, this method shows how densely cells are occupied by features.
    
    It provides a visual representation of the distribution and concentration of these features within the cells.

    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - binary_layer_key (str, optional): The key for accessing the layer to calculate the cell and feature statistics, Default 'binary'.
    - save (bool, optional): Whether or not to save the figure, Default False. 
    - save_dir (str, optional): Directory to sace the figure. Default None and saves to current directory. 
    - dpi (str, optional): resolution of the figure to save in dots per inch
    - kwds
        Are passed to :func:`matplotlib.pyplot.hist`.

    Returns:
    
    - AnnData obejct with following features: 
        - num_cell_per_feature (int): how many cells have a count for a feature? / number of cells sharing a feature
        - num_feature_per_cell : how many features are open for a cell? / number of features in a cell
        
    - Feature statistic plot 
    '''
    sns.set_style('whitegrid')
    
    if  'num_cell_per_feature' not in adata.varm_keys() and 'num_feature_per_cell' not in adata.obs_keys():
        print("calculating cell and feature statistics")
        cell_feature_statistics(adata, binary_layer_key =binary_layer_key)
    # plotting settings
    if xlabel ==None:
        plt.xlabel('features')
    else:
        plt.xlabel(xlabel)

    if ylabel ==None:
        plt.ylabel('frequency of cells')
    else:
        plt.ylabel(ylabel)

    if title !=None:
        plt.title(title)

    if color == None:
        color='c'
    if edgecolor == None:
        edgecolor='k'

    if bins == None:
        bins=50
    
    fig = plt.hist(adata.varm['num_cell_per_feature'], bins, color=color, edgecolor=edgecolor)

    if threshold != None:
        plt.axvline(x=threshold, color='r', linestyle='--')
        
    if save == True:
        if save_dir is not None:
            plt.savefig(save_dir, dpi=dpi)
        else:
            plt.savefig("./",dpi=dpi)    
    return adata


# 3D projection

def projection(prototype_adata, pseudobulk_adata, prototype_layer_key = "libsize_norm_log2_std", pseudobulk_layer_key = "libsize_norm_log2_bulk_scaled", prototype_label_font_size = 18, pseudobulk_label_font_size = 18, prototype_colors = None, cmap = None, pseudobulk_colors = None, color_key = "clustering_color", pseudobulk_point_size=180, prototype_point_size=150, pseudobulk_point_alpha=0.8, prototype_point_alpha=0.7, prototype_legend = True, pseudobulk_legend = True, save_path = None, dpi = 300, fig_size_inches = (15,15)):
    '''
    Custom 3D PCA projection of prototypes and pseudobulks.  

    Parameters:
    
    - prototype_adata (AnnData): An AnnData object containing the prototype count matrix.
    - pseudobulk_adata(AnnData): An AnnData object containing the pseudobulk count matrix.
    - prototype_layer_key (str): The key for accessing the prototype layer for projection. Default 'libsize_norm_log2_std'.
    - pseudobulk_layer_key (str): The key for accessing the pseudobulk layer for projection. Default 'libsize_norm_log2_bulk_scaled'
    - prototype_label_font_size (int): Font size of the prototype labels on the PCA projection. If set to 0, no labels will be plotted. Default 18. 
    - pseudobulk_label_font_size (int): Font size of the pseudobulk labels on the PCA projection. If set to 0, no labels will be plotted. Default 18.    
    - prototype_colors (List[str ]or None): A list of color codes to be used for plotting prototypes. If `None`, colors will be chosen by `cmap` parameter.
    - cmap (str): Matplotlib colormap used to colorcode the prototypes if prototype_colors is None.
    - pseudobulk_colors (List[str] or None): A list of color codes to be used for plotting pseudobulks. If `None`, colors will be determined by `color_key` parameter
    - color_key (str, optional): The key for accessing the cluster colors in the sc data. If provided, the pseudobulk points will be colored based on the cluster colors they originated from.
    - pseudobulk_point_size (int): Size of the pseudobulk point displayed on the plot. Default 180.
    - prototype_point_size (int): Size of the prototype point displayed on the plot. Default 200.
    - pseudobulk_point_alpha (float): Parameter controling the transparency of the plotted pseudobulks. Ranges between 0 (transparent) and 1 (opaque). Default 0.8.
    - prototype_point_alpha (float): Parameter controling the transparency of the plotted prototypes. Ranges between 0 (transparent) and 1 (opaque). Default 0.7.
    - prototype_legend (bool): A boolean value indicating whether or not to include prototype-related items in the legend. Deault True.
    - pseudobulk_legend(bool): A boolean value indicating whether or not to include pseudobulk-related items in the legend. Deault True.
    - save_path (str or None): Path where the plot should be saved. If `None`, the plot is not saved.
    - dpi (int): The resolution in dots per inch. Default 300.
    - fig_size_inches (tuple): A tuple representing the size (width, height) of the figure in inches. Default (15,15).

    Returns: 
    
    - 3D PCA projection figure.
    - PCA transformed values of prototypes.
    - PCA transformed values of pseudbulks.

    '''

    num_of_prototype_samples = prototype_adata.n_obs
    if num_of_prototype_samples >30:
        n_comp = 30
    else:
        n_comp = num_of_prototype_samples
    pca_prototype = PCA(n_components=n_comp)
    pca_prototype_train = pca_prototype.fit_transform(prototype_adata.layers[prototype_layer_key])
    PCs_pseudobulk_projection = pca_prototype.transform(pseudobulk_adata.layers[pseudobulk_layer_key])


    trained_prototype_pca_df = pd.DataFrame(data = pca_prototype_train
                 , columns = ["principal component " + str(i) for i in range(n_comp)])
    trained_prototype_pca_df["targets"] = prototype_adata.obs.index

    projected_pseudobulk_pca_df = pd.DataFrame(data = PCs_pseudobulk_projection
                 , columns = ["principal component " + str(i) for i in range(n_comp)])
    projected_pseudobulk_pca_df["targets"] = pseudobulk_adata.obs.index
    projected_pseudobulk_pca_df

    # sort pseudobulk adata so that ther cluster order matches the cluster color order
    projected_pseudobulk_pca_df["sort_index"] = projected_pseudobulk_pca_df["targets"].str.extract('(\d+)')
    projected_pseudobulk_pca_df["sort_index"] = projected_pseudobulk_pca_df["sort_index"].astype(int)
    projected_pseudobulk_pca_df.sort_values(by=["sort_index"], inplace=True)

    trained_prototype_pca_df_w_labels = trained_prototype_pca_df.copy()
    trained_prototype_pca_df_w_labels["cell_type"] = trained_prototype_pca_df_w_labels['targets'].apply(lambda r: '_'.join(r.split('_')[:-1]))
    my_color=pd.Series(pd.Categorical(trained_prototype_pca_df_w_labels["cell_type"])).cat.codes
    trained_prototype_pca_df_w_labels["color_id"] = my_color
    # plot
    fig = plt.figure()
    ax = plt.axes(projection='3d',computed_zorder=False)

    PC1 = trained_prototype_pca_df_w_labels['principal component 0'].values
    PC2 = trained_prototype_pca_df_w_labels['principal component 1'].values
    PC3 = trained_prototype_pca_df_w_labels['principal component 2'].values

    CELLTYPES = trained_prototype_pca_df_w_labels["cell_type"].values
    CELLTYPES_ = np.unique(CELLTYPES)
    # colors for prototype samples 
    if prototype_colors is None:
        if cmap is None:
            cmap="twilight_shifted"
        else:
            cmap=cmap
        COLORS = mcp.gen_color(cmap=cmap,n=len(CELLTYPES_))
    else:
        COLORS = prototype_colors
    for cell_type1, color in zip(CELLTYPES_, COLORS):
        idxs = np.where(CELLTYPES == cell_type1)
        # control the legend
        if prototype_legend == True:
            ax.scatter(PC1[idxs,], PC2[idxs,],PC3[idxs,], label=cell_type1,
                       s=prototype_point_size, color=color, alpha=prototype_point_alpha, marker = "v",zorder=4.4)
        else:
            ax.scatter(PC1[idxs,], PC2[idxs,],PC3[idxs,], s=prototype_point_size, color=color, alpha=prototype_point_alpha, marker = "v",zorder=4.4)

    # prototype labels on the plot
    m= np.array([list(PC1),list(PC2),list(PC3)])
    for i in range(len(m[0])): #plot each point + it's index as text above
        ax.text(m[0,i],m[1,i],m[2,i],  '%s' % ('  ' +CELLTYPES[i]), fontsize=prototype_label_font_size)

    pseudobulk_CELLTYPES = projected_pseudobulk_pca_df["targets"].values
    CELLTYPES_ = pseudobulk_CELLTYPES
    PC1_ = projected_pseudobulk_pca_df['principal component 0'].values
    PC2_ = projected_pseudobulk_pca_df['principal component 1'].values
    PC3_ = projected_pseudobulk_pca_df['principal component 2'].values
    if pseudobulk_colors is None:
    # get the color code from pseudobulk matrix:
        COLORS_ = pseudobulk_adata.uns[color_key]
    else: 
        COLORS_ = pseudobulk_colors
    for pseudobulk_cell_type, color in zip(CELLTYPES_, COLORS_):
        idxs = np.where(pseudobulk_CELLTYPES == pseudobulk_cell_type)
        # control the legend
        if pseudobulk_legend == True:
            ax.scatter(PC1_[idxs,], PC2_[idxs,],PC3_[idxs,], label=pseudobulk_cell_type,
                       s=pseudobulk_point_size, color=color, alpha=pseudobulk_point_alpha, zorder=4.5)
        else:
            ax.scatter(PC1_[idxs,], PC2_[idxs,],PC3_[idxs,],
                       s=pseudobulk_point_size, color=color, alpha=pseudobulk_point_alpha, zorder=4.5)            
        
    # pseudobulk labels on the plot
    m= np.array([list(PC1_),list(PC2_),list(PC3_)])
    for i in range(len(m[0])):
        ax.text(m[0,i],m[1,i],m[2,i],  '%s' % ('  ' +pseudobulk_CELLTYPES[i]), fontsize=pseudobulk_label_font_size)

    #fig
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_facecolor('white')
    if not (prototype_legend == False and pseudobulk_legend == False):
        ax.legend(ncol=2,handleheight=2.4, labelspacing=0.05)
    figure = plt.gcf() 
    figure.set_size_inches(fig_size_inches)

    # save
    if save_path is not None:
        fig.savefig(save_path)
    return fig, trained_prototype_pca_df_w_labels,projected_pseudobulk_pca_df 

# euclidean distance heatmap

def plot_pca_dist_heatmap(trained_bulk_pca_df_w_labels,projected_pseudobulk_pca_df, cmap='Blues_r'):
    '''
    Plot a heatmap visualizing the pairwise Euclidean distances of prototypes and pseudobulks.
    
    This function combines PCA components of projected pseudobulk data and trained prototype data, calculates the pairwise Euclidean distances, and plots a heatmap using Seaborn's clustermap.

    Parameters:
    
    - trained_bulk_pca_df_w_labels (DataFrame): DataFrame containing PCA components of trained prototypes with labels.
    - projected_pseudobulk_pca_df (DataFrame): DataFrame containing PCA components of projected pseudobulk data.
    - cmap (str, optional): Colormap for the heatmap. Default is 'Blues_r'.

    Returns:
    
    - tuple: A tuple containing:
        - sns.ClusterGrid: Seaborn ClusterGrid object representing the heatmap.
        - DataFrame: DataFrame containing the pairwise Euclidean distances.
    '''
    n_PC= projected_pseudobulk_pca_df.shape[1] -2
    #combine PCs
    pbulk_bulk_combined_pca_df = pd.concat([projected_pseudobulk_pca_df.iloc[:,0:n_PC], trained_bulk_pca_df_w_labels.iloc[:,0:n_PC]])
    names = list(projected_pseudobulk_pca_df["targets"])+list(trained_bulk_pca_df_w_labels["targets"])
    pbulk_bulk_combined_pca_df.index =names
    euclidean_dis=pdist(pbulk_bulk_combined_pca_df.iloc[:,0:n_PC].to_numpy(), 'euclidean')
    euclidean_dis_df = pd.DataFrame(squareform(euclidean_dis))
    euclidean_dis_df.index = names
    euclidean_dis_df.columns = names
    g = sns.clustermap(euclidean_dis_df,yticklabels=True,xticklabels=True,cmap=cmap)
    return g, euclidean_dis_df

def plot_pca_dist_cent_heatmap(trained_bulk_pca_df_w_labels,projected_pseudobulk_pca_df, cmap='Blues_r'):
    '''
    Plot a heatmap visualizing the pairwise Euclidean distances between centroids of prorotypes and pseudobulks.
    This function combines PCA components of projected pseudobulk data and trained prototype data, calculates centroids for trained prototype data, and plots a heatmap using Seaborn's clustermap.

    Parameters:
    - trained_bulk_pca_df_w_labels (DataFrame): DataFrame containing PCA components of trained bulk data with labels.
    - projected_pseudobulk_pca_df (DataFrame): DataFrame containing PCA components of projected pseudobulk data.
    - cmap (str, optional): Colormap for the heatmap. Default is 'Blues_r'.
    
    Returns:
    - tuple: A tuple containing:
        - sns.ClusterGrid: Seaborn ClusterGrid object representing the heatmap.
        - DataFrame: DataFrame containing the pairwise Euclidean distances.
    ''' 
    n_PC= projected_pseudobulk_pca_df.shape[1] -2
    trained_bulk_pca_df_w_labels_centroid = trained_bulk_pca_df_w_labels.groupby('cell_type')[trained_bulk_pca_df_w_labels.columns[0:n_PC]].agg('mean')
    # combine centrodis and pbulk dfs and calculate pairwise distances
    pbulk_bulk_centrid_combined_pca_df = pd.concat([projected_pseudobulk_pca_df.iloc[:,0:n_PC], trained_bulk_pca_df_w_labels_centroid.iloc[:,0:n_PC]])
    df_names = [name+"_pbulk" for name in list(projected_pseudobulk_pca_df["targets"])] +list(trained_bulk_pca_df_w_labels_centroid.index) 
    pbulk_bulk_centrid_combined_pca_df.index = df_names
    pbulk_bulk_centrid_euclidean_dis=pdist(pbulk_bulk_centrid_combined_pca_df.to_numpy(), 'euclidean')
    pbulk_bulk_centrid_euclidean_dis_df = pd.DataFrame(squareform(pbulk_bulk_centrid_euclidean_dis))
    pbulk_bulk_centrid_euclidean_dis_df.index = df_names
    pbulk_bulk_centrid_euclidean_dis_df.columns = df_names
    g = sns.clustermap(pbulk_bulk_centrid_euclidean_dis_df,yticklabels=True,xticklabels=True, cmap=cmap)
    
    return g, pbulk_bulk_centrid_euclidean_dis_df


def plot_gene_activity_of_UMAP(adata, gene_name, activity_matrix, out_path, point_size=22, cmap=None):
    '''
    Plot UMAP embedding of the given genes' activity across single cells. 
    
    This function saves a Matplotlib figure to the specified file path.

    Parameters:
    
    - adata (AnnData): An AnnData object containing the sc count matrix.
    - gene_name (str): Name of the gene.
    - activity_matrix (DataFrame): Gene activity score of the gene across cells. (Rows:cells x columns:genes (str))
    - out_path (str): The path to the output directory. 
    - point_size: Size of the cell points displayed on the UMAP.
    - cmap: Color map object passed to sc.pl.umap()

    Returns:
    
    - None
    
    '''
    if gene_name not in activity_matrix.index:
        print("this gene is not included in the gene activity matrix")
        return
    else:
        # scale the gene 0-1
        gene_df = pd.DataFrame(activity_matrix.loc[gene_name,:])
        x = gene_df.values #returns a numpy array
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        gene_df_scaled = pd.DataFrame(x_scaled)
        gene_df_scaled.index = gene_df.index
        adata.obs[gene_name] = gene_df_scaled.loc[adata.obs.index,]
        if cmap == None:
            blues_cmap = matplotlib.cm.Blues(np.linspace(0,1,10))
            custom_blues_cmap = sns.blend_palette(np.insert(blues_cmap[3:10], 0, matplotlib.colors.to_rgba("lightgray", alpha=None), axis=0), as_cmap=True)
            cmap=custom_blues_cmap
        else:
            cmap=cmap
        with plt.rc_context():  # Use this to set figure params like size and dpi
            sc.pl.umap(
            adata,
            color=gene_name,
            cmap=cmap,
            add_outline=False, frameon=False, title ="", save=False, size=point_size,
            legend_fontsize='xx-small’', show=False)
            plt.savefig(out_path, bbox_inches="tight")
        plt.show()
        
