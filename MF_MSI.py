import numpy as np
from numpy.linalg import inv
from utils import softmax
from utils import logsumexp
from utils import logdet
import time
from utils import eval_res

class MF_MSI(object):
    
    def __init__(self, Du, Mu, Di, Mi, I, J, K = 10, MM = 0, ImpFeed = 0):
        self.K = K
        self.Du = Du
        self.Mu = Mu
        self.Di = Di
        self.Mi = Mi
        self.I = I
        self.J = J
        self.MM = MM
        self.ImpFeed = ImpFeed
        self.modelparams_u = self.model_params(Du, Mu, K, I)
        self.modelparams_v = self.model_params(Di, Mi, K, J)
        self.latentparams_u = self.latent_params(self.modelparams_u)
        self.latentparams_v = self.latent_params(self.modelparams_v)
    
    class model_params:
        def __init__(self, D, M, K, I):
            self.D = D
            self.M = M
            self.I = I
            self.K = K
            self.W = np.random.multivariate_normal(np.zeros(K), np.identity(K), D)
            self.H = np.random.multivariate_normal(np.zeros(K), np.identity(K), np.sum(M))
            self.Sigma_x = np.identity(D)
            self.Prec_x = np.identity(D)       
            self.U_mean_prior = np.random.normal(0, 1, K)
            self.c = 1
            self.nu0 = -(self.K+1)
            self.param_a = 1
            self.param_b = 1
            self.s0 = 0 * np.eye(self.K)
            self.Xon = 1
            self.Yon = 1
            self.Ron = 1
            self.LogLikOn = 0
        
    class latent_params:
        def __init__(self, modelparams):
            self.U_SumSecMoment = 0
            self.U_mean = np.tile(modelparams.U_mean_prior[:,None], [1,modelparams.I])
            self.Psi_u = modelparams.H @ self.U_mean
            self.Prec_u_prior = np.identity(modelparams.K)
            self.U_SecMoment = np.tile(self.Prec_u_prior + modelparams.U_mean_prior[:,None].T @ modelparams.U_mean_prior[:,None], [modelparams.I, 1, 1])
            self.LogLik = 0
            self.LogRating = 0
            self.Psd_X = np.zeros((np.sum(modelparams.M) + modelparams.D, modelparams.I))
            self.Sigma_u = np.zeros((modelparams.I, modelparams.K, modelparams.K))

            
    def e_step(self, modelparams, latentparams, X, Y, R, O, MeanVecCouple, SecMomCouple, MM, ImpFeed):
           
        M = modelparams.M
        I = modelparams.I
        D = modelparams.D
        K = modelparams.K
        c = modelparams.c
        
        H = modelparams.H
        W = modelparams.W
        Prec_x = modelparams.Prec_x
        Sigma_x = modelparams.Sigma_x
        U_mean_prior = modelparams.U_mean_prior
        Xon = modelparams.Xon
        Yon = modelparams.Yon
        Ron = modelparams.Ron
    
        Psd_Cov = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_Prec = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_X = np.zeros((np.sum(M) + D, I))
        Psd_X[0:D, :] = X
        
        F_u = []
        for i in range(0,M.shape[0]):
            F_u.append(1/2 * (np.identity(M[i]) - (1/(M[i]+1)) * np.ones((M[i],1)) * np.ones((M[i],1)).T))
             
        
        Prec_u = np.zeros((I, K, K))
        Sigma_u = np.zeros((I, K, K))
        tmp2 = latentparams.Prec_u_prior
        if Xon == 1:
            tmp2 = tmp2 + (W.T @ Prec_x @ W)
        if Yon == 1:
            tmp = 0
            for i in range(0,M.shape[0]):
                ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
                tmp = tmp +  H[ind,:].T @ F_u[i] @ H[ind]
            tmp2 = tmp2 + tmp 
        
        start = time.time()
        
        if ImpFeed == 0:
            for i in range(0, I):
                InfCouple = np.sum(np.reshape(c * O[i], (len(O[i]), 1, 1)) * SecMomCouple, axis = 0)
                Prec_u[i] = Ron * InfCouple + tmp2
                Sigma_u[i] = inv(Prec_u[i])
        else:
            InfCouple = c * np.sum(SecMomCouple, axis = 0)
            Prec = Ron * InfCouple + tmp2
            Prec_u = np.tile(Prec, [I, 1, 1])
            Sigma_u = np.tile(inv(Prec), [I, 1, 1])
    
        #print ("Sigma took", time.time() - start, "seconds.")
        
        if Yon == 1:
            iter_psi = 5
        else:
            iter_psi = 1
    
        start = time.time()
        
        for iterPsi in range(0,iter_psi):
            G_u = np.zeros((np.sum(M), I))
            for i in range(0,M.shape[0]):
                ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
                Psi_u_d = softmax(latentparams.Psi_u[ind, :])
                G_u[ind,:] = F_u[i] @ latentparams.Psi_u[ind, :] - Psi_u_d[0:-1, :]
    
            U_mean = np.zeros((K, I))
            U_SecMoment = np.zeros((I, K, K))
            U_SumSecMoment = np.zeros((K, K))
            InfSum = np.zeros((K, I))
            InfSum = InfSum + (latentparams.Prec_u_prior @ U_mean_prior)[None].T
            if Xon == 1:
                InfSum = InfSum + (W.T @ (X / np.diag(Sigma_x)[None].T))
            if Yon == 1:
                InfSum = InfSum + (H.T @ (Y + G_u))
                
    
            for i in range(0, I):
                U_mean[:, i] = Sigma_u[i] @ ( InfSum[:,i] + Ron * c * (MeanVecCouple @ R[i]))
                if MM == 0:
                    U_SecMoment[i] = Sigma_u[i] + U_mean[:, i][None].T @ U_mean[:, i][None]
                else:
                    U_SecMoment[i] = U_mean[:, i][None].T @ U_mean[:, i][None]
                U_SumSecMoment = U_SumSecMoment + U_SecMoment[i]
        
            Psi_u_old = latentparams.Psi_u.copy()
            Psi_u = H @ U_mean
            latentparams.Psi_u = Psi_u
            conv = np.sum((Psi_u_old - Psi_u)**2) / (Psi_u.shape[0] * Psi_u.shape[1])
            #print(conv)
            if conv < 1e-5:
                #print("Converged")
                break;
        #print ("Mean took", time.time() - start, "seconds.")
        
        
        start = time.time()
        
        LogMult = 0
        Psd_Cov[0:D,0:D] =  Sigma_x
        Psd_Prec[0:D,0:D] = Prec_x
        for i in range(0,M.shape[0]):
            ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
            Psi_u_d = softmax(Psi_u[ind, :])
            G_u[ind,:] = F_u[i] @ Psi_u[ind, :] - Psi_u_d[0:-1, :]
            ind_tilde = range(np.sum(M[0:i]) + D, np.sum(M[0:i+1]) + D)
            Y_tilde = inv(F_u[i]) @ (Y[ind, :] + G_u[ind,:]) 
            Psd_X[ind_tilde, :] = Y_tilde
            Psd_Cov[np.ix_((ind_tilde),(ind_tilde))] = inv(F_u[i])
            Psd_Prec[np.ix_((ind_tilde),(ind_tilde))] = F_u[i]
            
            if modelparams.LogLikOn == 1:
                LogMult_c = 0.5 * np.sum(Psi_u[ind, :] * (F_u[i] @ Psi_u[ind, :]), axis = 0) - np.sum(Psi_u_d[0:-1, :] * Psi_u[ind, :], axis = 0) + logsumexp(Psi_u[ind, :])
                LogMult = LogMult + 0.5 * np.log(2*np.pi) * M[i] + 0.5 * logdet(inv(F_u[i])) + 0.5 * np.sum(Y_tilde * (F_u[i] @ Y_tilde), axis = 0) - LogMult_c
                
        if modelparams.LogLikOn == 1:
            
            LogMult = Yon * np.sum(LogMult)
            
            if Xon == 1 or Yon == 1:
                if Xon == 1 and Yon == 1:
                    Psd_Beta = np.append(W,H,axis=0)
                elif Xon == 1:
                    Psd_Beta = W.copy()
                elif Yon == 1:
                    Psd_Beta = H.copy()
                Psd_Mean =  Psd_Beta @ U_mean
                LogLink =  np.sum(0.5 * (logdet(Psd_Prec) - (np.sum(M) + D) * np.log(2*np.pi)) - 0.5 * np.sum((Psd_X - Psd_Mean) * (Psd_Prec @ (Psd_X - Psd_Mean)), axis = 0))
                for i in range(0, I):
                    LogLink = LogLink - 0.5 * np.trace( Psd_Prec @ Psd_Beta @ Sigma_u[i] @ Psd_Beta.T )
            else:
                LogLink = 0
        
            Entropy = 0
            LogLatent = np.sum(0.5 * (logdet(latentparams.Prec_u_prior) - (K) * np.log(2*np.pi)) - 0.5 * np.sum((U_mean_prior[None].T - U_mean) * (latentparams.Prec_u_prior @ (U_mean_prior[None].T - U_mean)), axis = 0))
            for i in range(0, I):
                LogLatent = LogLatent - 0.5 * np.trace( latentparams.Prec_u_prior @ Sigma_u[i])
                Entropy =Entropy + 0.5 * (np.log(2*np.pi) * K + logdet(Sigma_u[i]))
            
            
            LogPrior = -(modelparams.param_a + 1) * np.sum(np.log(np.diag(Sigma_x))) - modelparams.param_b * np.sum(np.diag(Prec_x))
            LogPrior = Xon * LogPrior + 0.5 * (modelparams.nu0 + K + 1) * logdet(latentparams.Prec_u_prior) - 0.5 * np.trace(modelparams.s0 @ latentparams.Prec_u_prior)
            
            LogLik = (LogLink + LogMult + LogLatent + Entropy + LogPrior) / I
            
            LogRating = (0.5 * np.sum( O * (np.log(c) - c * ((U_mean.T @ MeanVecCouple - R)**2) - np.log(2*np.pi)))) / np.sum(O)
        else:
            LogLik = 0
            LogRating = 0
        
        #print ("Log took", time.time() - start, "seconds.")
        
        #latentparams_upd = latent_params(modelparams)
        
        latentparams.Psi_u = Psi_u
        latentparams.U_SumSecMoment = U_SumSecMoment
        latentparams.U_SecMoment = U_SecMoment
        latentparams.U_mean = U_mean
        latentparams.LogLik = LogLik
        latentparams.Psd_X = Psd_X
        latentparams.LogRating = LogRating
        latentparams.Sigma_u = Sigma_u
        
        #return latentparams
    
    def m_step(self, modelparams, latentparams, X, Y):
        
            YY = np.sum(X * X, axis = 1)
            I = modelparams.I
            D = modelparams.D
            
            U_mean_prior = np.mean(latentparams.U_mean,axis = 1)
            
            Beta = (latentparams.Psd_X @ latentparams.U_mean.T) @ inv(latentparams.U_SumSecMoment)
            W = Beta[0:D, :]
            Sigma_x = np.diag((2*modelparams.param_b + YY  - np.diag(W @ (latentparams.Psd_X @ latentparams.U_mean.T)[0:D, :].T)) / (I + 2*(modelparams.param_a+1)))
            Prec_x = np.diag(1/np.diag(Sigma_x))
            H = Beta[D:, :]
            
            modelparams.U_mean_prior = U_mean_prior
            modelparams.Prec_x = Prec_x
            modelparams.Sigma_x = Sigma_x
            modelparams.W = W
            modelparams.H = H
            
    def fit(self, X, Y, Z, P, Rtrain, Otrain, Rtest, Otest, iterno = 10):
        
        for iter in range(0,iterno):
        
            self.e_step(self.modelparams_u, self.latentparams_u, 
                   X, Y, Rtrain, Otrain, 
                   self.latentparams_v.U_mean, self.latentparams_v.U_SecMoment, 
                   self.MM, self.ImpFeed)
            
            self.e_step(self.modelparams_v, self.latentparams_v,
                   Z, P, Rtrain.T, Otrain.T, 
                   self.latentparams_u.U_mean, self.latentparams_u.U_SecMoment,
                   self.MM, self.ImpFeed)
            
            self.m_step(self.modelparams_u, self.latentparams_u, X, Y)
            self.m_step(self.modelparams_v, self.latentparams_v, Z, P)
            
            Rpred = self.latentparams_u.U_mean.T @ self.latentparams_v.U_mean
            
            #Test_MSE, Train_MSE, Test_recall, Train_recall,_ = eval_res(Rpred, Rtest, Rtrain, Otest, Otrain, 2)
            #print("LH = {:.2f}, MSE = {:.3f}, Recall = {:.3f}".format(0, Test_MSE, Test_recall))
        
        return Rpred