import numpy as np
from numpy.linalg import inv
from utils import softmax


class MF_MSI(object):
    
    def __init__(self, DatasetSpec, K = 10):
        
        self.K = K
        self.Du = DatasetSpec['Du']
        self.Mu = DatasetSpec['Mu']
        self.Di = DatasetSpec['Di']
        self.Mi = DatasetSpec['Mi']
        self.I = DatasetSpec['I']
        self.J = DatasetSpec['J']
        self.modelparams_u = self.model_params(self.Du, self.Mu, K, self.I)
        self.modelparams_v = self.model_params(self.Di, self.Mi, K, self.J)
        self.latentparams_u = self.latent_params(self.modelparams_u)
        self.latentparams_v = self.latent_params(self.modelparams_v)
    
    class model_params:
        def __init__(self, D, M, K, I):
            
            self.D = D
            self.M = M
            self.I = I
            self.K = K
            
            # Initialize global params
            self.W = np.random.multivariate_normal(np.zeros(K), np.identity(K), D)
            self.H = np.random.multivariate_normal(np.zeros(K), np.identity(K), np.sum(M))
            self.Sigma_x = np.identity(D)
            self.Prec_x = np.identity(D)
            
            # Fixed model priors
            self.U_mean_prior = np.random.normal(0, 1, K)
            self.Prec_u_prior = np.identity(K)
            self.c = 1
            self.nu0 = -(self.K+1)
            self.param_a = 1
            self.param_b = 1
            self.s0 = 0 * np.eye(self.K)
            
            # Enable/disable data sources
            self.Xon = 1
            self.Yon = 1
            self.Ron = 1
        
    class latent_params:
        def __init__(self, modelparams):
            
            # Initialize local params and sufficient statistics
            self.U_SumSecMoment = 0
            self.U_mean = np.tile(modelparams.U_mean_prior[:,None], [1,modelparams.I])
            self.Psi_u = modelparams.H @ self.U_mean
            self.U_SecMoment = np.tile(modelparams.Prec_u_prior + modelparams.U_mean_prior[:,None].T @ modelparams.U_mean_prior[:,None], [modelparams.I, 1, 1])
            self.Psd_X = np.zeros((np.sum(modelparams.M) + modelparams.D, modelparams.I))
            self.Sigma_u = np.zeros((modelparams.I, modelparams.K, modelparams.K))

            
    def e_step(self, modelparams, latentparams, X, Y, R, O, MeanVecCouple, SecMomCouple):
           
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
        
        # Infer posterior covariances
        F_u = []
        for i in range(0,M.shape[0]):
            F_u.append(1/2 * (np.identity(M[i]) - (1/(M[i]+1)) * np.ones((M[i],1)) * np.ones((M[i],1)).T))
             
        Prec_u = np.zeros((I, K, K))
        Sigma_u = np.zeros((I, K, K))
        InfPrec = modelparams.Prec_u_prior
        if Xon == 1:
            InfPrec = InfPrec + (W.T @ Prec_x @ W)
        if Yon == 1:
            InfCat = 0
            for i in range(0,M.shape[0]):
                ind = range(np.sum(M[0:i]), np.sum(M[0:i+1]))
                InfCat = InfCat +  H[ind,:].T @ F_u[i] @ H[ind]
            InfPrec = InfPrec + InfCat 

        for i in range(0, I):
            InfCouple = np.sum(np.reshape(c * O[i], (len(O[i]), 1, 1)) * SecMomCouple, axis = 0)
            Prec_u[i] = Ron * InfCouple + InfPrec
            Sigma_u[i] = inv(Prec_u[i])
        
        
        # Infer posterior means
        iter_psi = 1
        if Yon == 1:
            iter_psi = 5

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
            InfSum = InfSum + (modelparams.Prec_u_prior @ U_mean_prior)[None].T
            if Xon == 1:
                InfSum = InfSum + (W.T @ (X / np.diag(Sigma_x)[None].T))
            if Yon == 1:
                InfSum = InfSum + (H.T @ (Y + G_u))
                
            for i in range(0, I):
                U_mean[:, i] = Sigma_u[i] @ ( InfSum[:,i] + Ron * c * (MeanVecCouple @ R[i]))
                U_SecMoment[i] = Sigma_u[i] + U_mean[:, i][None].T @ U_mean[:, i][None]
                U_SumSecMoment = U_SumSecMoment + U_SecMoment[i]
        
            Psi_u = H @ U_mean
            latentparams.Psi_u = Psi_u
        
        # Fuse multimodal sources
        Psd_Cov = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_Prec = np.zeros((np.sum(M) + D, np.sum(M) + D))
        Psd_X = np.zeros((np.sum(M) + D, I))
        Psd_X[0:D, :] = X

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
            

        latentparams.Psi_u = Psi_u
        latentparams.U_SumSecMoment = U_SumSecMoment
        latentparams.U_SecMoment = U_SecMoment
        latentparams.U_mean = U_mean
        latentparams.Psd_X = Psd_X
        latentparams.Sigma_u = Sigma_u
    
    def m_step(self, modelparams, latentparams, X, Y):
        
            YY = np.sum(X * X, axis = 1)
            I = modelparams.I
            D = modelparams.D
            
            # Estimate global model parameters
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
            
    def fit(self, Dataset, iterno = 10):
        
        for iter in range(0,iterno):
        
            self.e_step(self.modelparams_u,
                        self.latentparams_u, 
                        Dataset['X'], 
                        Dataset['Y'],
                        Dataset['Rtrain'], 
                        Dataset['Otrain'], 
                        self.latentparams_v.U_mean,
                        self.latentparams_v.U_SecMoment)
            
            self.e_step(self.modelparams_v,
                        self.latentparams_v,
                        Dataset['Z'],
                        Dataset['P'],
                        Dataset['Rtrain'].T,
                        Dataset['Otrain'].T, 
                        self.latentparams_u.U_mean,
                        self.latentparams_u.U_SecMoment)
            
            
            self.m_step(self.modelparams_u,
                        self.latentparams_u,
                        Dataset['X'],
                        Dataset['Y'])
            
            self.m_step(self.modelparams_v,
                        self.latentparams_v,
                        Dataset['Z'],
                        Dataset['P'])
    
    def predict(self):        
    
        Rpred = self.latentparams_u.U_mean.T @ self.latentparams_v.U_mean

        return Rpred