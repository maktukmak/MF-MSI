# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:41:59 2018

@author: Mehmet
"""

import numpy as np
from lightfm.datasets import fetch_stackexchange
import time
from utils import softmax
import os
dirname = os.path.dirname(__file__)

def train_test_split(ratedata, I, J, option = 'warm'):
    
    """
        Train\Test splitting of interactions
        Argument 'option' choses between 'warm' or 'cold' start scenario
    """

    ratedata_test = np.empty((0,ratedata.shape[1]))
    ratedata_train = np.empty((0,ratedata.shape[1]))
    
    if option == 'warm':
        for i in range(0, J):
           itemdatarate = len(np.where(ratedata[:,1] == i + 1)[0])
           if itemdatarate < 5:
               ind = np.where(ratedata[:,1] == i+1)[0]
               ratedata_train = np.append(ratedata_train, ratedata[ind, :], axis = 0)
           else:
               itemdatarate_test = int(float(itemdatarate) / 5)
               ind = np.where(ratedata[:,1] == i+1)[0]
               np.random.shuffle(ind)
               ratedata_test = np.append(ratedata_test, ratedata[ind[0:itemdatarate_test], :], axis = 0)
               ratedata_train = np.append(ratedata_train, ratedata[ind[itemdatarate_test:], :], axis = 0)
           
    elif option == 'cold':
        indtest = np.arange(J)
        np.random.shuffle(indtest)
        for i in range(0, J):
            ind = np.where(ratedata[:,1] == indtest[i]+1)[0]
            if i < int(J/5):
                ratedata_test = np.append(ratedata_test, ratedata[ind, :], axis = 0)
            else:
                ratedata_train = np.append(ratedata_train, ratedata[ind, :], axis = 0)
    
    return ratedata_train, ratedata_test

def rate_to_matrix(ratedata, I, J):
    
    R = np.zeros((I,J))
    C = np.zeros((I,J))
    R[ratedata[:,0].astype(int)-1, ratedata[:,1].astype(int)-1] = ratedata[:,2]
    C[ratedata[:,0].astype(int)-1, ratedata[:,1].astype(int)-1] = 1
    
    return R, C

class model_gen_params:	# Fixed system parameters 
    def __init__(self):
        self.Dx = 3
        self.Mx = np.array([5, 3])
        self.Dz = 3
        self.Mz = np.array([5, 3])
        self.I = 300
        self.J = 500
        self.K = 3
        self.lmd_u = 1
        self.lmd_v = 1
        self.cij = 1
        self.miss_frac = 0.95
        
def mm_data_generation(params):
    
    """
        Synthetic data geenration based on params
    """
    
    Dx = params.Dx
    I = params.I
    Mx = params.Mx
    Dz = params.Dz
    J = params.J
    Mz = params.Mz
    lmd_u = params.lmd_u
    lmd_v = params.lmd_v
    K = params.K
        
    mean = np.zeros((K))
    cov = (1/lmd_u) * np.identity(K)
    U = np.random.multivariate_normal(mean, cov, I)
    U = U.T
    
    mean = np.zeros((K))
    cov = (1/lmd_v) * np.identity(K)
    V = np.random.multivariate_normal(mean, cov, J)
    V = V.T
    
    W = np.random.multivariate_normal(np.zeros(K), np.identity(K), Dx)
    #mu_W = np.random.normal(0, 1, Dx)
    mu_W = np.zeros((Dx,1)) 
    Sig_X = np.eye(Dx)
    Xorg = np.zeros((Dx,I))
    mean = W @ U + mu_W
    for i in range(0,I):
        Xorg[:,i] = np.random.multivariate_normal(mean[:,i], Sig_X, 1)
    
    
    H = np.random.multivariate_normal(np.zeros(K), np.identity(K), np.sum(Mx))
    Yorg = np.zeros((np.sum(Mx),I))
    mean = H @ U
    for i in range(0,Mx.shape[0]):
        ind = range(np.sum(Mx[0:i]), np.sum(Mx[0:i+1]))
        prob = softmax(mean[ind, :])
        for j in range(0,I):
            Yorg[ind,j] = np.random.multinomial(1, prob[:,j], size=1)[0, 0:Mx[i]]
            
            
    A = np.random.multivariate_normal(np.zeros(K), np.identity(K), Dz)
    #mu_A = np.random.normal(0, 1, Dz)
    mu_A = np.zeros((Dz,1))
    Sig_Z = np.eye(Dz)
    Zorg = np.zeros((Dz,J))
    mean = A @ V + mu_A
    for j in range(0,J):
        Zorg[:,j] = np.random.multivariate_normal(mean[:,j], Sig_Z, 1)
    
    
    B = np.random.multivariate_normal(np.zeros(K), np.identity(K), np.sum(Mz))
    Porg = np.zeros((np.sum(Mz),J))
    mean = B @ V
    for i in range(0,Mz.shape[0]):
        ind = range(np.sum(Mz[0:i]), np.sum(Mz[0:i+1]))
        prob = softmax(mean[ind, :])
        for j in range(0,J):
            Porg[ind,j] = np.random.multinomial(1, prob[:,j], size=1)[0, 0:Mz[i]]
            
    # Rating Generation
    Rorg = np.zeros((I,J))
    for i in range(0,I):
        for j in range(0,J):
            mean = U[:,i] @ V[:,j]
            Rorg[i,j] = np.random.normal(mean, params.cij, 1)
            
    C = np.random.choice([0, 1], size=(Rorg.shape) , p=[params.miss_frac, 1-params.miss_frac])
    Rorg[np.where(C == 0)] = np.nan
    
    
    mu_R = np.nanmean(Rorg, axis=1)
    Rnorm = Rorg - mu_R[None].T
    std_R = np.nanstd(Rorg, axis=1)
    Rnorm /= std_R[None].T
    
       
    Rate = np.zeros((C.sum(), 3))
    Rate[:,0] = np.where(C==1)[0] + 1
    Rate[:,1] = np.where(C==1)[1] + 1
    Rate[:,2] = Rnorm[np.where(C==1)]
    
    return Xorg, Yorg, Mx, Porg, Zorg, Mz, Rate, I, J

def movie100kprep(option = 'warm'):

    filename = os.path.join(dirname, "Datasets/ml-100k/u.user")
    usrdatac = np.genfromtxt(filename, delimiter='|', usecols= [0, 1])
    usrdatad = np.genfromtxt(filename, delimiter='|', usecols= [2, 3], dtype = str)
    Ddu = []
    ind = np.unique(usrdatad[:,0])
    Ddu.append(ind.shape[0])
    for i in range(0, ind.shape[0]):
        usrdatad[np.where(usrdatad[:,0] == ind[i] ), 0] = i
    ind = np.unique(usrdatad[:,1])
    Ddu.append(ind.shape[0])
    for i in range(0, ind.shape[0]):
        usrdatad[np.where(usrdatad[:,1] == ind[i] ), 1] = i
    Ddu = np.array(Ddu)
    Mu = Ddu - 1
    usrdatad = usrdatad.astype(int)
    
    I = usrdatac.shape[0]
    usrdatacat = np.zeros((I, sum(Ddu)-Ddu.shape[0]))
    temp = np.zeros((I, Mu[0]+1))
    temp[np.arange(I), usrdatad[:,0]-1] = 1
    usrdatacat[:,0:Mu[0]] = temp[:,0:Mu[0]]
    temp = np.zeros((I, Mu[1]+1))
    temp[np.arange(I), usrdatad[:,1]-1] = 1
    usrdatacat[:,Mu[0]:Mu[0] + Mu[1]] = temp[:,0:Mu[1]]
    
    Xorg = usrdatac[:, 1][None]
    Yorg = usrdatacat.T
    Du = 1

    filename = os.path.join(dirname, "Datasets/ml-100k/u.item")
    itmdatac = np.genfromtxt(filename, delimiter='|', usecols= [1], dtype = str)
    itmdatad = np.genfromtxt(filename, delimiter='|', usecols= list(range(5,24)), dtype = str)
    Mi = np.ones(itmdatad.shape[1]).astype(int)
    Porg = itmdatad.astype(int).T
    
    Di = 1
    
    J = itmdatac.shape[0]
    for i in range(0, J):
        itmdatac[i] = itmdatac[i][-5:-1]
    Zorg = itmdatac.astype(float)[None]
    
    filename = os.path.join(dirname, "Datasets/ml-100k/u.data")
    ratedata = np.genfromtxt(filename, delimiter='\t')
    ratedata = np.append(ratedata[:, 0:3], np.zeros((ratedata.shape[0],1)), axis = 1)
    ratedata[np.where(ratedata[:,2] >= 4),3] = 1
    
    # Normalization of Cont. Data
    mu_W = np.mean(Xorg, axis=1)
    Xnorm = Xorg - mu_W[None].T
    std_X = np.std(Xnorm, axis=1)
    Xnorm /= std_X[None].T
    Ynorm = Yorg.copy()
        
    mu_A = np.mean(Zorg, axis=1)
    Znorm = Zorg - mu_A[None].T
    std_Z = np.std(Znorm, axis=1)
    Znorm /= std_Z[None].T
    Pnorm = Porg.copy()
    
    
    ratedata_train, ratedata_test = train_test_split(ratedata, I, J, option)

    Rtrain, Ctrain = rate_to_matrix(ratedata_train, I, J)
    Rtest, Ctest = rate_to_matrix(ratedata_test, I, J)

    Rtrain[(Rtrain < 4) & (Rtrain > 0)] = -1
    Rtrain[Rtrain >= 4] = 1

    Rtest[(Rtest < 4) & (Rtest > 0)] = -1
    Rtest[Rtest >= 4] = 1
    
    
    return Xnorm, Ynorm, Du, Mu, Pnorm, Znorm, Di, Mi, Rtrain, Rtest, Ctrain, Ctest, I, J

def movie1mprep(fetch_from_file = 1, resplit = 0, option = 'warm'):
    
    #start = time.time()
    if fetch_from_file == 1:
        filename = os.path.join(dirname, "Datasets/ml-1m/users.dat")
        usrdatac = np.genfromtxt(filename, delimiter='::', usecols= [2])
        usrdatad = np.genfromtxt(filename, delimiter='::', usecols= [1, 3], dtype = str)
        Ddu = []
        ind = np.unique(usrdatad[:,0])
        Ddu.append(ind.shape[0])
        for i in range(0, ind.shape[0]):
            usrdatad[np.where(usrdatad[:,0] == ind[i] ), 0] = i
        ind = np.unique(usrdatad[:,1])
        Ddu.append(ind.shape[0])
        
        Ddu = np.array(Ddu)
        Mu = Ddu - 1
        usrdatad = usrdatad.astype(int)
        
        I = usrdatac.shape[0]
        usrdatacat = np.zeros((I, sum(Ddu)-Ddu.shape[0]))
        temp = np.zeros((I, Mu[0]+1))
        temp[np.arange(I), usrdatad[:,0]-1] = 1
        usrdatacat[:,0:Mu[0]] = temp[:,0:Mu[0]]
        temp = np.zeros((I, Mu[1]+1))
        temp[np.arange(I), usrdatad[:,1]-1] = 1
        usrdatacat[:,Mu[0]:Mu[0] + Mu[1]] = temp[:,0:Mu[1]]
        Xorg = usrdatac.T[None]
        Yorg = usrdatacat.T
        Mu = np.array(Mu)
        Du = 1
        
        filename = os.path.join(dirname, "Datasets/ml-1m/movies.dat")
        itmdatac = np.genfromtxt(filename, delimiter='::', usecols= [1], dtype = str)
        itmdatad = np.genfromtxt(filename, delimiter='::', usecols= [2], dtype = str)
        itmmap = np.genfromtxt(filename, delimiter='::', usecols= [0], dtype = int)
        J = itmdatac.shape[0]
        for i in range(0, J):
            itmdatac[i] = itmdatac[i][-5:-1]
        Zorg = itmdatac.astype(float)[None]
        
        Genres = []
        for i in range(0, len(itmdatad)):
            Genres = np.append(Genres, itmdatad[i].split("|"))
        ind = np.unique(Genres)
        Ddi = (ind.shape[0])
        Mi = np.ones(Ddi).astype(int)
        itmdatacat = np.zeros((Ddi, J))
        for i in range(0, J):
            for j in range(0, len(itmdatad[i].split("|"))):
                itmdatacat[np.where(ind == itmdatad[i].split("|")[j])[0], i] = 1
        Porg = itmdatacat.astype(int)
        Di = 1
        
        filename = os.path.join(dirname, "Datasets/ml-1m/ratings.dat")
        ratedata = np.genfromtxt(filename, delimiter='::')
        ratedata = np.append(ratedata[:, 0:3], np.zeros((ratedata.shape[0],1)), axis = 1)
        ratedata[np.where(ratedata[:,2] >= 4),3] = 1
        for i in range(0, len(ratedata)):
            ratedata[i, 1] = np.where(itmmap == ratedata[i,1])[0]+1
        
        # Normalization of Cont. Data
        mu_W = np.mean(Xorg, axis=1)
        Xnorm = Xorg - mu_W[None].T
        std_X = np.std(Xnorm, axis=1)
        Xnorm /= std_X[None].T
        Ynorm = Yorg.copy()
            
        mu_A = np.mean(Zorg, axis=1)
        Znorm = Zorg - mu_A[None].T
        std_Z = np.std(Znorm, axis=1)
        Znorm /= std_Z[None].T
        Pnorm = Porg.copy()
        
        ratedata_train, ratedata_test = train_test_split(ratedata, I, J, option)
        Dataset = [Xnorm, Ynorm, Du, Mu, Pnorm, Znorm, Di, Mi, ratedata, ratedata_train, ratedata_test, I, J]
        np.save('movie1m.npy', Dataset)
            
    else:
        Dataset = np.load('movie1m.npy')
        [Xnorm, Ynorm, Du, Mu, Pnorm, Znorm, Di, Mi, ratedata, ratedata_train, ratedata_test, I, J] = Dataset
    #print ("Fetch took", time.time() - start, "seconds.")
    #start = time.time()
    
    if resplit == 1:
        ratedata_train, ratedata_test = train_test_split(ratedata, I, J, option)
    
    Rtrain, Ctrain = rate_to_matrix(ratedata_train, I, J)
    Rtest, Ctest = rate_to_matrix(ratedata_test, I, J)
    
    Rtrain[(Rtrain < 4) & (Rtrain > 0)] = -1
    Rtrain[Rtrain >= 4] = 1
    
    Rtest[(Rtest < 4) & (Rtest > 0)] = -1
    Rtest[Rtest >= 4] = 1
    #print ("Split took", time.time() - start, "seconds.")
    
    return Xnorm, Ynorm, Du, Mu, Pnorm, Znorm, Di, Mi, Rtrain, Rtest, Ctrain, Ctest, I, J

def movie10mprep(fetch_from_file = 0, resplit = 0, option = 'warm'):
    
    #start = time.time()
    if fetch_from_file == 1:
        
        filename = os.path.join(dirname, "Datasets/ml-10M100K/movies.dat")
        itmdatac = np.genfromtxt(filename, delimiter='::', usecols= [1], encoding='utf-8', dtype = str)
        itmdatad = np.genfromtxt(filename, delimiter='::', usecols= [2], encoding='utf-8', dtype = str)
        itmmap = np.genfromtxt(filename, delimiter='::', usecols= [0], encoding='utf-8', dtype = int)
        J = itmdatac.shape[0]
        for i in range(0, J):
            itmdatac[i] = itmdatac[i][-5:-1]
        Zorg = itmdatac.astype(float)[None]
        
        Genres = []
        for i in range(0, len(itmdatad)):
            Genres = np.append(Genres, itmdatad[i].split("|"))
        ind = np.unique(Genres)
        Ddi = (ind.shape[0])
        Mi = np.ones(Ddi).astype(int)
        itmdatacat = np.zeros((Ddi, J))
        for i in range(0, J):
            for j in range(0, len(itmdatad[i].split("|"))):
                itmdatacat[np.where(ind == itmdatad[i].split("|")[j])[0], i] = 1
        Porg = itmdatacat.astype(int)
        Di = 1
        
        filename = os.path.join(dirname, "Datasets/ml-10M100K/ratings.dat")
        ratedata = np.genfromtxt(filename, delimiter='::')
        ratedata = np.append(ratedata[:, 0:3], np.zeros((ratedata.shape[0],1)), axis = 1)
        ratedata[np.where(ratedata[:,2] >= 4),3] = 1
        for i in range(0, len(ratedata)):
            ratedata[i, 1] = np.where(itmmap == ratedata[i,1])[0]+1
        
        mu_A = np.mean(Zorg, axis=1)
        Znorm = Zorg - mu_A[None].T
        std_Z = np.std(Znorm, axis=1)
        Znorm /= std_Z[None].T
        Pnorm = Porg.copy()
        
        I = 71567
        Xnorm = np.zeros((1,I))
        Ynorm = np.zeros((1,I))
        Mu = np.array([1])
        Du = 0
        
        ratedata_train, ratedata_test = train_test_split(ratedata, I, J, option)
        Dataset = [Xnorm, Ynorm, Du, Mu, Pnorm, Znorm, Di, Mi, ratedata.astype(int), ratedata_train.astype(int), ratedata_test.astype(int), I, J]
        if option == 'warm':
            np.save('movie10m_warm.npy', Dataset)
        else:
            np.save('movie10m_cold.npy', Dataset)
            
    else:
        if option == 'warm':
            Dataset = np.load('movie10m_warm.npy')
        else:
            Dataset = np.load('movie10m_cold.npy')
        [Xnorm, Ynorm, Du, Mu, Pnorm, Znorm, Di, Mi, ratedata, ratedata_train, ratedata_test, I, J] = Dataset
    #print ("Fetch took", time.time() - start, "seconds.")
    #start = time.time()
    
    if resplit == 1:
        ratedata_train, ratedata_test = train_test_split(ratedata, I, J, option)
    
    Rtrain, Ctrain = rate_to_matrix(ratedata_train.astype(float), I, J)
    Rtest, Ctest = rate_to_matrix(ratedata_test.astype(float), I, J)
    
    Rtrain[(Rtrain < 4) & (Rtrain > 0)] = -1
    Rtrain[Rtrain >= 4] = 1
    
    Rtest[(Rtest < 4) & (Rtest > 0)] = -1
    Rtest[Rtest >= 4] = 1
    #print ("Split took", time.time() - start, "seconds.")
    
    return Xnorm, Ynorm, Du, Mu, Pnorm, Znorm, Di, Mi, Rtrain, Rtest, Ctrain, Ctest, I, J

