import numpy as np
import random
from scipy.sparse import lil_matrix, coo_matrix

class MaskData(object):
    def __init__(self, mat, k):
        self.mat = mat
        self.k = k
        self.original = mat.copy()
        self.state = 31 # this is the random state for train test split
        self.test_data = None
        self.col_idx = None 
        self.row_idx = None 


    def _mask_clip(self, row_or_col):
        ''' 
        Cuts out items from matrix that do not contain at least k values on axis=0
        '''
        mat = self.mat
        k = self.k
        lil = mat.tolil()
        to_remove = []
        for idx, i in enumerate(lil.rows):
            if len(i) < k:
                to_remove.append(idx)
        lil.rows = np.delete(lil.rows, to_remove)
        lil.data = np.delete(lil.data, to_remove)
        if row_or_col == 'row':
            self.row_idx = np.delete(range(lil.shape[0]), to_remove) 
        elif row_or_col == 'col':
            self.col_idx = np.delete(range(lil.shape[0]), to_remove)
        remaining = lil.shape[0] - len(to_remove)
        lil = lil[:remaining]
        self.mat = lil
        return self 

    def _mask_find_subset(self):
        # Cut out users that don't meet minimum k
        self._mask_clip('row')

        self.mat = self.mat.T

        # Cut out items that don't meet minimum k
        self._mask_clip('col')

        self.mat = self.mat.T
        return self 

    def get_test_data(self):
        self._mask_find_subset()
        state = self.state
        k = self.k
        r_initial_idx = self.row_idx
        c_initial_idx = self.col_idx
        #random.seed(state)
        mat = self.mat
        mat = mat.tocoo()
        mat_row, mat_col = mat.row, mat.col
        # we need to convert these numbers back to the original dataset's column and row indices
        mat_zip = zip(mat_row, mat_col)
        for idx, (row_val, col_val) in enumerate(mat_zip):
            r_val = r_initial_idx[row_val]
            c_val = c_initial_idx[col_val]
            mat_zip[idx] = (r_val, c_val) 

        self.test_data = random.sample(mat_zip, k) 
        return self

    def _convert_test_points(self, zipped_test):
        c_res = np.empty(len(zipped_test))
        r_res = np.empty(len(zipped_test)) 
        x_res = np.empty(len(zipped_test))

        for idx, (r, c) in enumerate(zipped_test):
            c_res[idx] = c
            r_res[idx] = r
            x_res[idx] = self.original.tocsr()[r,c]

        return (r_res.astype(int), c_res.astype(int)), x_res
           
    def mask(self):
        self.get_test_data()
        test_data = self.test_data
        original_data = self.original.tocoo()
        original_data = zip(original_data.row, original_data.col)
        mask = np.ones(self.original.nnz)
        for idx, item in enumerate(original_data):
            if item in test_data:
                mask[idx] = 0
        train_set = self.original.copy()
        train_set.data[mask == 0] = 0 
        train_set.eliminate_zeros()
        test_set = self.original
        print(test_data)
        test_data, test_results = self._convert_test_points(test_data)
        return train_set, self.original, test_data, test_results 
 
