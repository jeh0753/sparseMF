# sparseMF

This repo introduces two sparse matrix factorization algorithms. The algorithms were originally introduced by Trevor Hastie et al. in a 2014 paper ["Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares"](https://stanford.edu/~rezab/papers/fastals.pdf) as an extention to SoftImpute, which was introduced in 2009. A sparse implementation of each of these algorithms is introduced here. Both borrow from the [FancyImpute](https://github.com/hammerlab/fancyimpute/tree/master/fancyimpute) python dense implementation of the 2009 SoftImpute algorithm. With large, sparse matrices, this version is significantly faster at predicting ratings for user/item pairs. 

This package includes:

* A new sparse matrix class, entitled Sparse Plus Low Rank (SPLR), as described in the 2009 paper ['Spectral Regularization Algorithms for Learning Large Incomplete Matrices'](https://web.stanford.edu/~hastie/Papers/mazumder10a.pdf).
* An implementation of SoftImpute.
* An implementation of SoftImputeALS. 
* Unit tests for SoftImpute.
* Benchmarking for SoftImputeALS, against GraphLab and the FancyImpute SoftImpute implementation.


## Resources

Here are some helpful resources I found during my research:

1. [A Helpful Introduction to Matrix Factorization Recommenders](http://infolab.stanford.edu/~ullman/mmds/ch9.pdf).
2. [Benchmarks for MovieLens Dataset](https://sites.google.com/site/domainxz/benchmark).
3. [Trevor Hastie's Hybrid Implementation of Soft-Impute and ALS](https://arxiv.org/abs/1410.2596).

