import pandas as pd
import numpy as np
import graphlab
from collections import defaultdict
import gc
from time import time
from imputation import Imputer, FactorizationImputer, BiScaler
from scipy.sparse import coo_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.models import Sequential, Model


def load_movielens(path, seed = 0):
    np.random.seed(0)
    df = pd.read_csv(path)[:100]   
    n = len(df)
    row = np.array(df.userId)
    col = np.array(df.movieId)
    data = np.array(df.rating)
    mat = coo_matrix((data, (row, col)))
    nnz = mat.nnz
    mask = np.random.choice(nnz,int(nnz*.3))
    mat_train = mat.copy()
    mat_train.data[mask] = 0
    train, test = mat_train.toarray(), mat.toarray()
    train, test = np.insert(train, 0, 1, axis=0), np.insert(test, 0, 1, axis=0)
    train, test = np.insert(train, 0, 1, axis=1), np.insert(test, 0, 1, axis=1)
    train[train == 0] = np.nan
    return train, test

def gpredict(train):
    c = coo_matrix(train)
    sf = graphlab.SFrame({'row': c.row, 'col': c.col, 'data': c.data})
    sf_small = sf.dropna('data', how="all")
    m1 = graphlab.factorization_recommender.create(sf_small, user_id='row', item_id='col', target='data')
    spred = m1.predict(sf)
    pred = spred.to_numpy().reshape(train.shape[0],train.shape[1])
    return pred

def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, W_regularizer=l2(reg))(inp)

def neural_net(train):
    small_train = train.copy()
    small_train[[np.isnan(small_train)]] = 0
    c = coo_matrix(small_train)
    n_factors = 50
    n_users = train.shape[0]
    n_movies  = train.shape[1]
    user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
    movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)
    x = merge([u, m], mode='concat')
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(70, activation='relu')(x)
    x = Dropout(0.75)(x)
    x = Dense(1)(x)
    nn = Model([user_in, movie_in], x)
    nn.compile(Adam(0.001), loss='mse')
    nn.fit([c.row, c.col], c.data, batch_size=64, nb_epoch=8)
    d = coo_matrix(train)
    pred = nn.predict_on_batch([d.row, d.col])
    pred = pred.reshape(train.shape[0], train.shape[1])
    return pred

def compute_bench(path, n_iter=3):

    it = 0

    results = defaultdict(lambda: [])

    for i in xrange(n_iter):
        train, test = load_movielens(path, seed=i)
        mask = (test != 0)

        it += 1
        print('====================')
        print('Iteration %03d of %03d' % (i+1, n_iter))
        print('====================')

        gc.collect()
        print("benchmarking FactorizationImputer: ")
        tstart = time()
        imp = FactorizationImputer(normalizer=BiScaler(), verbose=False)#may need to add some variables here
        pred = imp.fit_transform(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['FactorizationImputer'].append((time() - tstart, mse))

        gc.collect()
        print("benchmarking Imputer using mean: ")
        tstart = time()
        imp = Imputer(strategy='mean')
        pred = imp.fit_transform(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['Mean Imputer'].append((time() - tstart, mse))

        gc.collect()
        print("benchmarking Imputer using most frequent value: ")
        tstart = time()
        imp = Imputer(strategy='most_frequent')
        pred = imp.fit_transform(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['Most Frequent Value Imputer'].append((time() - tstart, mse))

        gc.collect()
        print("benchmarking Imputer using Neural Net: ")
        tstart = time()
        imp = Imputer(strategy='median')
        pred = neural_net(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['Neural Network'].append((time() - tstart, mse))

        gc.collect()
        print("benchmarking GraphLab: ")
        tstart = time()
        pred = gpredict(train)
        pred *= mask
        mse = mean_squared_error(test, pred)
        results['GraphLab'].append((time() - tstart, mse))

    return results, train, test

def plot_results(results):
    mse = dict()
    time = dict()
    for k, v in results.iteritems():
        avg_time = []
        avg_mse = []
        for t in v:
            avg_time.append(t[0])
            avg_mse.append(t[1])
        avg_time = np.mean(avg_time)
        avg_mse = np.mean(avg_mse)
        mse[k] = avg_mse
        time[k] = avg_time
    plt.clf()
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.bar(range(len(mse)), mse.values(), align='center')
    ax2.bar(range(len(time)), time.values(), align='center')
    ax1.set_title("Mean Squared Error Each Model")
    ax2.set_title("Time To Run Each Model")
    plt.xticks(range(len(time)), time.keys())
    plt.show()


if __name__ == '__main__':
    results, train, test = compute_bench('movielens.csv', n_iter=3)
    plot_results(results)

    
