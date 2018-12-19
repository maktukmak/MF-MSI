import numpy as np


def softmax(x):
    tmp = np.append(x, np.zeros((1, x.shape[1])), axis = 0)
    e_x = np.log(np.sum(np.exp(tmp - np.amax(tmp,axis=0,keepdims=True)), axis = 0, keepdims=True)) + np.amax(tmp,axis=0,keepdims=True)
    
    out = tmp - e_x
    prob = np.exp(out)
    
    return prob


def logsumexp(x):
    tmp = np.append(x, np.zeros((1, x.shape[1])), axis = 0)
    lse = np.log(np.sum(np.exp(tmp - np.amax(tmp,axis=0,keepdims=True)), axis = 0, keepdims=True)) + np.amax(tmp,axis=0,keepdims=True)
    
    return lse

def logdet(x):
    
    return np.log(np.linalg.det(x))


def recall_eval(Rpred, Rtest, Ctest, at_K):
    Test_recall = 0
    cnt = 0
    recall_vec = np.zeros(Rpred.shape[0])
    for i in range(0, Rpred.shape[0]):
        usr_like = np.where(Rtest[i, :] == 1)[0]
        if len(usr_like) > 0:
            ind = np.where(Ctest[i, :] == 1)[0]
            pred_like = ind[np.flip(np.argsort(Rpred[i,ind]), axis = 0)][0:at_K]
            recall = len(np.intersect1d(pred_like, usr_like)) / min(at_K, len(usr_like))
            Test_recall = Test_recall + recall
            cnt = cnt + 1
            recall_vec[i] = recall
        else:
            recall_vec[i] = np.nan
    Test_recall = Test_recall / cnt
    return Test_recall, recall_vec

def eval_res(Rpred, Rtest, Rtrain, Ctest, Ctrain, at_K):
    
    Test_MSE = np.mean(np.square(Rpred - Rtest)[np.where(Ctest == 1)])
    Train_MSE = np.mean(np.square(Rpred - Rtrain)[np.where(Ctrain == 1)])
    Test_recall, Test_recall_vec = recall_eval(Rpred, Rtest, Ctest, at_K)
    Train_recall,_ = recall_eval(Rpred, Rtrain, Ctrain, at_K)
    
    return Test_MSE, Train_MSE, Test_recall, Train_recall, Test_recall_vec
    
