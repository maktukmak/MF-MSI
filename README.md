# MF-MSI

This code is Python implementation of the paper "A probabilistic framework to incorporate mixed-data type features: Matrix factorization with multimodal side information".

# Dependencies

The code requires Numpy and Scipy packages.

# Model

The model performs inference for the user and item latent variables. The inputs are sparse rating matrix, multivariate Gaussian side information matrix one for the users and one for the items, and categorical side information matrix (can include more than one categorical entry) one for the users and one for the items. Variational EM is used to perform inference.

# Datasets

Movielens 100K dataset is provided for demonstration.

