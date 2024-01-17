import unittest
import numpy as np
import scATAcat
import anndata 

class test_bulk_data_functions(unittest.TestCase):

    def test_preprocess_bulk_adata(self):
        # Create a mock AnnData object for testing
        # Test Case 1: Copy is False, remove_chrY is True
        obs_data = pd.DataFrame(["cell_0", "cell_1","cell_2","cell_3"], columns=["cells"])
        obs_data.index = obs_data.cells
        var_data= pd.DataFrame(["chr1_1_3", "chr3_1_5", "chrY_5_6", "chr13_10_100", "chrX_30_50"], columns=["regions"])
        var_data.index = var_data.regions
        count_matrix = np.array([[1, 1, 0, 2, 7], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [7, 0, 0, 4, 4]])
        bulk_adata = anndata.AnnData(np.array(count_matrix, dtype= np.float32), obs=obs_data, var=var_data)
        preprocessed_adata = scATAcat.preprocess_bulk_adata(bulk_adata, remove_chrY=True, copy=False)

        expected_var_names = ["chr1_1_3", "chr3_1_5", "chr13_10_100", "chrX_30_50"]
        self.assertTrue(np.array_equal(preprocessed_adata.var_names, expected_var_names))
        self.assertTrue(np.array_equal(preprocessed_adata.X, np.array([[1, 1, 2, 7], [0, 0, 0, 1], [7, 0, 4, 4]])))

        # Test Case 2: Copy is True, remove_chrY is False
        bulk_adata = anndata.AnnData(np.array(count_matrix, dtype= np.float32), var=var_names, obs=obs_names)
        preprocessed_adata = scATAcat.preprocess_bulk_adata(bulk_adata, remove_chrY=False, copy=True)

        self.assertTrue(np.array_equal(preprocessed_adata.var_names, var_names))
        self.assertTrue(np.array_equal(preprocessed_adata.X, np.array([[1, 1, 0, 2, 7], [0, 0, 0, 0, 1], [7, 0, 0, 4, 4]])))


if __name__ == '__main__':
    unittest.main()
