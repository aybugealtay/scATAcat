import unittest
import numpy as np
import pandas as pd
import scATAcat
import anndata
import scipy.sparse
class test_helper_functions(unittest.TestCase):

    def test_preprocess_sc_matrix(self):

        # Test Case 1: Copy is False, remove_chrY is True
        obs_data = pd.DataFrame(["cell_0", "cell_1", "cell_2"], columns=["cells"])
        obs_data.index = obs_data.cells
        var_data= pd.DataFrame(["chr1_1_3", "chr3_1_5", "chrY_5_6"], columns=["regions"])
        var_data.index = var_data.regions
        count_matrix = np.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]])
        adata = anndata.AnnData(count_matrix, obs=obs_data, var=var_data)
        adata.obsm['num_feature_per_cell'] = np.array([2, 0, 2])
        adata.varm['num_cell_per_feature'] = np.array([2, 0, 2])

        preprocessed_adata = scATAcat.preproces_sc_matrix(adata, cell_cutoff=1, cell_cutoff_max=10, feature_cutoff=1, remove_chrY=True, copy=False)

        expected_count_matrix = np.array([[1], [3]])
        self.assertTrue(np.array_equal(preprocessed_adata.X, expected_count_matrix))

        # Test Case 2: Copy is True, remove_chrY is False
        adata = anndata.AnnData(X=count_matrix, obs=obs_data, var=var_data)
        adata.obsm['num_feature_per_cell'] = np.array([2, 0, 2])
        adata.varm['num_cell_per_feature'] = np.array([2, 0, 2])

        preprocessed_adata = scATAcat.preproces_sc_matrix(adata, cell_cutoff=1, cell_cutoff_max=10, feature_cutoff=1, remove_chrY=False, copy=True)

        expected_count_matrix = np.array([[1, 2], [3, 4]])
        self.assertTrue(np.array_equal(preprocessed_adata.X, expected_count_matrix))

    def test_apply_TFIDF_sparse(self):
        
        # Test Case 1: Binary layer already present
        obs_data = pd.DataFrame(["cell_0", "cell_1", "cell_2"], columns=["cells"])
        obs_data.index = obs_data.cells
        var_data= pd.DataFrame(["region_0", "region_1", "region_2"], columns=["regions"])
        var_data.index = var_data.regions
        count_matrix = np.array([[1, 0, 2], [0, 5, 6], [3, 4, 0]])
        adata = anndata.AnnData(count_matrix, obs=obs_data, var=var_data)
        binary_layer = scipy.sparse.csr_matrix(np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]))
        adata.layers['binary'] = binary_layer

        normalized_adata = scATAcat.apply_TFIDF_sparse(adata, binary_layer_key='binary', TFIDF_key='TF_logIDF')

        # Check if TF-log(IDF) layer is added and calculated correctly
        expected_TFIDF = np.array([[0.69314718, 0., 0.69314718], [0., 0.69314718, 0.69314718], [0.69314718, 0.69314718, 0.]])
        self.assertTrue(np.array_equal(normalized_adata.layers['TF_logIDF'].toarray(), expected_TFIDF))

        # Test Case 2: Binary layer needs to be calculated
        adata = anndata.AnnData(count_matrix, obs=obs_data, var=var_data)
        normalized_adata = scATAcat.apply_TFIDF_sparse(adata, binary_layer_key='binary', TFIDF_key='TF_logIDF')

        # Check if binary layer and TF-log(IDF) layer are added and calculated correctly
        expected_binary = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
        expected_TFIDF = np.array([[0.69314718, 0., 0.69314718], [0., 0.69314718, 0.69314718], [0.69314718, 0.69314718, 0.]])
        self.assertTrue(np.array_equal(normalized_adata.layers['binary'].toarray(), expected_binary))
        self.assertTrue(np.array_equal(normalized_adata.layers['TF_logIDF'].toarray(), expected_TFIDF))


if __name__ == '__main__':
    unittest.main()
