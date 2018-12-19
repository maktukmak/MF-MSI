# MF-MSI

This code is Python implementation of the paper "Matrix Factorization with Multimodal Side Information".

# Installation

The code only requires Numpy and Scipy packages.

# Model

The model performs inference over users and items latent variables for item recommendation purpose. The inputs to the model are sparse rating matrix, one multivariate Gaussian side information vector for each user and item, one categorical side information vector (can include more than one categorical modality) for each user and item. Variational EM is used to make inference.

# Datasets

Three Movielens datasets (100K, 1M and 10M) are provided. The test code runs for 100K for fast presentation.

# Run

To run the model:

pyton MF_MSI_test.py

