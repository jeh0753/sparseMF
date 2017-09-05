from sparse_als_soft_impute import SparseSoftImputeALS
import numpy as np
import graphlab
import pandas as pd
from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, issparse
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from bench_plot_sparse_mf import MaskData


def load_movielens(path, leave_n_out=3, sample=1000, seed=0):
    np.random.seed(seed)
    df = pd.read_csv(path).sample(sample, random_state=2)   
    n = len(df)
    row = np.array(df.userId)
    col = np.array(df.movieId)
    data = np.array(df.rating)
    mat = coo_matrix((data, (row, col)))
    masker = MaskData(mat, leave_n_out)
    train, test, test_data_points, test_results = masker.mask()
    return train, test, test_data_points, test_results 


class CustomMixin(TransformerMixin):
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])
        return self

class SparseALS(CustomMixin):
    def __init__(self, shrinkage_value=0.02, max_rank=3):
        self.max_rank = max_rank
        self.shrinkage_value = shrinkage_value
        self.si = None

    def get_params(self, **kwargs):
        params = {'shrinkage_value': self.shrinkage_value}
        params['max_rank'] = self.max_rank
        return params

    def fit(self, X, y):
        data = y
        shape = X[0][2]
        row = np.empty(len(X))
        col = np.empty(len(X))
        for i, (r, c, _) in enumerate(X):
            row[i] = r 
            col[i]  = c
        #import ipdb; ipdb.set_trace()
        train = csr_matrix( (data,(row,col)), shape=shape)
        self.si = SparseSoftImputeALS(max_rank=self.max_rank, shrinkage_value=self.shrinkage_value, max_iters=30, verbose=False)
        self.si.complete(train)
        return self
        #print(si._pred_sparse(test_data_points[0], test_data_points[1], si.X_filled))
    
    def transform(self, X):
        row = np.empty(len(X))
        col = np.empty(len(X))
        for i, (r, c, _) in enumerate(X):
            row[i] = r 
            col[i]  = c
        X = [row, col]
        pred = self.si.predict(X[0], X[1]) 
        return pred
    
    def predict(self, X):
        row = np.empty(len(X))
        col = np.empty(len(X))
        for i, (r, c, _) in enumerate(X):
            row[i] = r 
            col[i]  = c
        X = [row.astype(int), col.astype(int)]
        pred = self.si.predict(X[0], X[1]) 
        return pred

def SIGrid(train):
    params = dict(max_rank=[10, 20], shrinkage_value=[.02, 10])
    grid = GridSearchCV(SparseALS(), cv=3, n_jobs=1, scoring="mean_squared_error", param_grid=params)
    x = zip(train.row, train.col, [train.shape]*len(train.row))
    y = train.data
    grid.fit(x, y)
    return grid.best_params_
      
def GLGrid(train):
    c = coo_matrix(train)
    sf = graphlab.SFrame({'row': c.row, 'col': c.col, 'data': c.data})
    sf_small = sf.dropna('data', how="all")
    folds = graphlab.cross_validation.KFold(sf_small, 3)
    params = dict([('target', 'data'), ('user_id', 'row'), ('item_id', 'col'), ('num_factors', [10, 20]), ('sgd_step_size', [0.02, 10])])
    job = graphlab.grid_search.create(folds, graphlab.factorization_recommender.create, params) 
    params = job.get_best_params() 
    return params

if __name__ == '__main__':
    path = 'movielens.csv'
    train, test, testdatapoints,test_results = load_movielens(path)
    grid =  SIGrid(train)
    
        
