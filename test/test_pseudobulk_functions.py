import unittest
import numpy as np
import pandas as pd
import scATAcat
import anndata

class test_pseudobulk_functions(unittest.TestCase):

    def test_get_pseudobulk_matrix(self):

        # Test Case 1: Method is "sum"
        features = np.array(["region0","region1", "region2", "region3"])
        obs_data = pd.DataFrame(np.array(["cell_0", "cell_1", "cell_2", "cell_3"]), columns=["cells"])
        var_data = pd.DataFrame(features, columns=["regions"])
        count_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [2, 4, 6, 8]])
        adata = anndata.AnnData(X=count_matrix, obs=obs_data, var=var_data)
        adata.obs['leiden'] =[1,1,2,2]
        pseudobulk_df = scATAcat.get_pseudobulk_matrix(adata, cluster_key="leiden", method="sum")

        # Check if pseudobulk matrix is constructed correctly
        expected_pseudobulk_df = pd.DataFrame(
            {
                "clust_1": [6, 8, 10, 12],
                "clust_2": [11, 14, 17, 20]
            },
            index=features
        )
        self.assertTrue(pseudobulk_df.equals(expected_pseudobulk_df))

        # Test Case 2: Method is "mean"
        pseudobulk_df_mean = get_pseudobulk_matrix(adata, cluster_key="leiden", method="mean")

        # Check if pseudobulk matrix is constructed correctly using mean aggregation
        expected_pseudobulk_df_mean = pd.DataFrame(
            {
                "clust_1": [3.0, 4.0, 5.0, 6.0],
                "clust_2": [5.5, 7.0, 8.5, 10.0]
            },
            index=features
        )
        self.assertTrue(pseudobulk_df_mean.equals(expected_pseudobulk_df_mean))

if __name__ == '__main__':
    unittest.main()
