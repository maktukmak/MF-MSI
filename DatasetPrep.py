# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:41:59 2018

@author: Mehmet
"""

import numpy as np
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
    
    """
        Convert rating entries from sparse to dense matrix
    """
    
    R = np.zeros((I,J))
    C = np.zeros((I,J))
    R[ratedata[:,0].astype(int)-1, ratedata[:,1].astype(int)-1] = ratedata[:,2]
    C[ratedata[:,0].astype(int)-1, ratedata[:,1].astype(int)-1] = 1
    
    return R, C


def movie100kprep(option = 'warm'):
    
    """
        Read MovieLens100K dataset and split to train/test sets 
    """

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

    Rtrain, Otrain = rate_to_matrix(ratedata_train, I, J)
    Rtest, Otest = rate_to_matrix(ratedata_test, I, J)

    Rtrain[(Rtrain < 4) & (Rtrain > 0)] = -1
    Rtrain[Rtrain >= 4] = 1

    Rtest[(Rtest < 4) & (Rtest > 0)] = -1
    Rtest[Rtest >= 4] = 1
    
    Dataset = {}
    Dataset['X'] = Xnorm
    Dataset['Y'] = Ynorm
    Dataset['P'] = Pnorm
    Dataset['Z'] = Znorm
    Dataset['Rtrain'] = Rtrain
    Dataset['Rtest'] = Rtest
    Dataset['Otrain'] = Otrain
    Dataset['Otest'] = Otest
    
    DatasetSpec = {}
    DatasetSpec['Di'] = Di
    DatasetSpec['Du'] = Du
    DatasetSpec['Mi'] = Mi
    DatasetSpec['Mu'] = Mu
    DatasetSpec['I'] = I
    DatasetSpec['J'] = J
    

    return Dataset, DatasetSpec

