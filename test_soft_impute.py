import numpy as np
from sparse_soft_impute import SoftImpute, SPLR
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, lil_matrix
from sklearn.utils.testing import assert_raises, assert_equal, assert_array_equal
import unittest


class TestPredict(unittest.TestCase):

    def setUp(self):
        row  = np.array([0, 3, 1, 0, 4])
        col  = np.array([0, 3, 1, 2, 5])
        data = np.array([4, -5, 7, -9, 2])
        self.x = csr_matrix((data, (row, col)), shape=(5, 6))
        self.si = SoftImpute(fill_method='sparse')
        self.x_pred = self.si.complete(self.x)
        self.splr = SPLR(self.si.prepare_input_data(self.x))

    def test_prepare_input_data(self):
        '''
        Ensure that prepare_input_data method sets the missing mask
        and clips max_rank back to an acceptable level, if it is outside
        bounds.
        '''
        x_prepared = self.si.prepare_input_data(self.x)
        self.assertTrue(len(self.si.missing_mask) == 3) 
        self.assertTrue(self.si.max_rank == 2)
        self.si.max_rank = 10
        self.si.prepare_input_data(self.x)    
        self.assertFalse(self.si.max_rank == 10)
            
    def test_fill(self):
        '''
        For the sparse setting, fill calls _preprocess_sparse(X). This
        test ensures the preprocessing step works as expected, and outputs
        are of the expected dimensions.
        '''
        rows, cols, shape = self.si.missing_mask
        self.assertTrue(len(rows) == len(cols))
        self.assertTrue(len(shape) == 2)
        U, D_sq, V = self.si.fill(self.x, inplace=True)
        self.assertTrue(U.shape == (shape[0], self.si.max_rank))
        self.assertTrue(np.all(D_sq == np.ones(self.si.max_rank)))
        self.assertTrue(np.all(V == np.zeros((shape[1],self.si.max_rank))))

    def test_UD(self):
        U = np.ones((6,3))
        D = np.ones(3)
        m = 6
        expected = np.ones((6,3))
        self.assertTrue(np.all(expected == self.si._UD(U, D, m)))

    def test_xhat_pred(self):
        '''
        Ensures that the first value outputted by _xhat_pred is 
        equal to the first output from x_result_svd's recomposed 
        output.
        '''
        x_result_svd = self.x_pred
        x_dense = x_result_svd[0].dot(np.diag(x_result_svd[1]).dot(x_result_svd[2].T))
        self.assertAlmostEqual(x_dense[0,0], self.si._xhat_pred(x_result_svd, self.x)[0], places=5)

    def test_als_u_step(self):
        '''
        Tests that the output shapes are correct.
        '''
        V_expected_shape = (6,2)    
        D_sq_expected_length = self.si.max_rank
        output = self.si._als_u_step(self.x_pred, self.splr)
        self.assertTrue(len(output[1]) == D_sq_expected_length)
        self.assertTrue(output[0].shape == V_expected_shape)

    def test_als_v_step(self):
        '''
        Tests that the output shapes are correct.
        '''
        us, d_sqs, vs = (5,2), 2, (6,2)
        output = self.si._als_v_step(self.x_pred, self.splr)
        self.assertTrue(output[0].shape == us)
        self.assertTrue(len(output[1]) == 2)
        self.assertTrue(output[2].shape == vs)
    
    def test_fnorm(self):
        '''
        Calculated Frobenius Norm by hand, ensured matching results 
        '''
        output = self.si._fnorm(self.si.fill(self.x), self.x_pred) 
        expected_output = 120.53 
        self.assertAlmostEqual(expected_output, output, places=2)

    def test_als_cleanup_step(self):
        '''
        Tests that the output shapes are correct.
        '''
        Us, D_sqs, Vs = (5,2), 2, (6,2)
        output = self.si._als_cleanup_step(self.x_pred, self.splr)
        self.assertTrue(output[0].shape == Us)
        self.assertTrue(len(output[1]) == 2)
        self.assertTrue(output[2].shape == Vs)
        
    def test_als_step(self):
        '''
        Ensure that X_fill_svd is updating, and that shapes remain
        correct.
        '''
        output = self.si._als_step(self.x_pred, self.splr)
        Uo, so, Vo = self.x_pred
        U, s, V = output
        self.assertTrue(U.shape == Uo.shape)
        self.assertTrue(V.shape == Vo.shape)
        self.assertFalse(np.all(s == so))
        self.assertFalse(np.all(U == Uo))
        self.assertFalse(np.all(V == Vo))
        
    def test_solve(self):
        '''
        Ensure shape of output is correct.
        '''
        output = self.si.solve(self.x_pred, self.x)
        U, s, V = output
        Uo, so, Vo = self.x_pred

        self.assertTrue(U.shape == Uo.shape)
        self.assertTrue(V.shape == Vo.shape)
        self.assertFalse(np.all(s == so))
        self.assertFalse(np.all(U == Uo))
        self.assertFalse(np.all(V == Vo))
        
    def test_predict_one(self):
        self.assertEqual(self.si.predict(1,1), 7)
        self.assertNotEqual(self.si.predict(0,1), 0)
        with self.assertRaises(ValueError): 
            self.si.predict(5,0)
    
    def test_predict_many(self):

        col_ids = np.array([0, 3, 5, 2])
        row_ids = np.array([0, 3, 4, 0])
        expected = np.array([4, -5, 2, -9])
        self.assertTrue(np.all(self.si.predict_many(row_ids, col_ids) == expected))

        #col_ids = np.array([1,2,1,1,4])
        #row_ids = np.array([2,1,1,3,4])
        #x_result_svd = si.complete(x)
        #x_dense = x_result_svd[0].dot(np.diag(x_result_svd[1]).dot(x_result_svd[2].t))
        #unknown_in = zip(row_ids, col_ids)
        #expected = np.empty(len(unknown_in))
        #for idx, (r, c) in enumerate(unknown_in):
        #    expected[idx] = x_dense[r, c]
        #self.asserttrue(np.all(si.predict_many(row_ids, col_ids) == expected))




if __name__ == '__main__':
    unittest.main()
    '''
# test parameters
sf = SoftImpute(max_rank=3, shrinkage_value=1, init_fill_method='sparse')
# original dataset:
xsc = np.array([[ 0.1390011, -0.09270246, -0.04629866, 0, 0], [-0.4469535, 0,  0.44695354, 0, 0], [-0.8252855, 0, 0,  0.1160078,  0.7092777], [-0.1981945, -0.7993484,  0.16913247,  0.8089969, 0], [ 0.9662344,  0.01088327, -0.56979652, -0.9250004,  0.5176792], [ 0.3651963,  0.86175224, 0, 0, -1.2269486]])
x_orig = csc_matrix(xsc)
# same dataset rounded and in more legible format:
np.array([[ 0.14, -0.09, -0.05,  0.  ,  0.  ],                                         
       [-0.45,  0.  ,  0.45,  0.  ,  0.  ],                                         
       [-0.83,  0.  ,  0.  ,  0.12,  0.71],                                         
       [-0.2 , -0.8 ,  0.17,  0.81,  0.  ],                                         
       [ 0.97,  0.01, -0.57, -0.93,  0.52],                                         
       [ 0.37,  0.86,  0.  ,  0.  , -1.23]]) 
# results from Trevor Hasties (roughly; these were actually my results from my version)
 np.array([[ 0.03,  0.  , -0.02, -0.03,  0.02],                                         
       [-0.18, -0.09,  0.1 ,  0.17,  0.02],                                         
       [-0.26, -0.28,  0.09,  0.18,  0.3 ],                                         
       [-0.31, -0.24,  0.13,  0.25,  0.19],                                         
       [ 0.42,  0.08, -0.26, -0.42,  0.16],                                         
       [ 0.25,  0.42, -0.03, -0.12, -0.55]])  '''
