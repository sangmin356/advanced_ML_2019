# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#%%
class Q5:
    
    # Class Variables: Learning Rate 
    eta_1 = np.repeat(np.sqrt((2*np.log(2))/1000), 1000) # Hedge
    eta_2 = np.repeat(np.sqrt((8*np.log(2))/1000), 1000) # Reparametrized Hedge
    eta_3 = np.sqrt(np.log(2)/np.arange(1,1001)) # Anytime Hedge
    eta_4 = 2*np.sqrt(np.log(2)/np.arange(1,1001)) # Anytime Reparametrized Hedge
    
    # Worst case for FTL
    mat_wc = [np.hstack(([[0.5],[0]], np.tile([[0,1],[1,0]],500)))]
    for i in range(10):
        mat_wc_ = mat_wc
        mat_wc = np.append(mat_wc, mat_wc_, axis = 0)
    
    # Class Variable: the number of running algorithm
    r = 10
    
    def __init__(self, mu, opt):
        self.mu = mu
        self.opt = opt # 0 for 5.1 and 1 for 5.3
        
    # Hedge Algorithm
    def Hedge(self, eta, matrix):
        L_array = np.zeros(shape = (2,1001))
        p_array = np.zeros(shape = (2,1000))
        At_array = np.array([])
        lAt_array = np.array([])
        regret_array = np.array([])
        
        for i in range(1000):
            v = np.exp(-eta[i]*L_array[:,i])
            v_sum = v.sum()
            p_array[:, i] = v / v_sum
            L_array[:, (i+1)] = L_array[:,i] + matrix[:,i]
            At = np.random.choice([0,1], 1, p = p_array[:,i])[0]
            At_array = np.append(At_array, At)
            lAt_array = np.append(lAt_array, matrix[At, i])
            if self.opt == 0:
                regret = (1 - 2*self.mu)*(At_array.sum())
            else:
                regret = lAt_array[0:(i+1)].sum() - np.amin(L_array[:, i]) 
            regret_array = np.append(regret_array, regret)
        
        return regret_array
    
    # Follow-The-Leader Algorithm
    def FTL(self, matrix):
        L_array = np.zeros(shape = (2,1000))
        regret_array = np.array([])
        L_array[:,0] = matrix[:,0]
        At = np.random.choice([0,1], 1, p = [0.5, 0.5])[0]
        At_array = np.array([At])
        lAt_array = np.array([matrix[At,0]])
        regret = (1 - 2*self.mu)*(At_array.sum())
        regret_array = np.append(regret_array, regret)
        
        for i in range(1,1000):
            At = np.where(L_array[:,(i-1)] == np.amin(L_array[:,(i-1)]))[0][0]
            At_array = np.append(At_array, At)
            lAt_array = np.append(lAt_array, matrix[At, i])
            L_array[:, i] = L_array[:,(i-1)] + matrix[:,i]
            if self.opt == 0:
                regret = (1 - 2*self.mu)*(At_array.sum())
            else:
                regret = lAt_array[0:(i+1)].sum() - np.amin(L_array[:, i])
            regret_array = np.append(regret_array, regret)
        
        return regret_array
    
    # Generate Expert Matrix
    def gen_matrix(self):
        s = np.random.binomial(1, self.mu, 1000)
        sp = np.repeat(1,1000) - s
        matrix = np.array([s,sp])
        
        return matrix   

    # Generate r=10 Expert Matrices for the experiment
    def exp_matrix(self):
        matrix = [Q5.gen_matrix(self)]
        tmp = Q5.r - 1
        for i in range(tmp):
            matrix_ = [Q5.gen_matrix(self)]
            matrix = np.append(matrix, matrix_, axis = 0)

        return matrix
    
    # Record the results
    def recorder(self, eta, matrix, alg):
        if alg == "F":
            result = [Q5.FTL(self, matrix[0])]
            tmp = Q5.r - 1
            for i in range(tmp):
                rlt = [Q5.FTL(self, matrix[(i+1)])]
                result = np.append(result, rlt, axis = 0)
                
            result_mean = np.average(result, axis=0)
            result_std = np.std(result, axis=0)
            
            return [result_mean, result_std]
        
        if alg == "H":
            result = [Q5.Hedge(self, eta, matrix[0])]
            tmp = Q5.r - 1
            for i in range(tmp):
                rlt = [Q5.Hedge(self, eta, matrix[(i+1)])]
                result = np.append(result, rlt, axis = 0)
                
            result_mean = np.average(result, axis=0)
            result_std = np.std(result, axis=0)
            
            return [result_mean, result_std]
    
    def result(self, matrix):
        data_1 = self.recorder(0, matrix, "F")
        data_2 = self.recorder(self.eta_1, matrix, "H")
        data_3 = self.recorder(self.eta_2, matrix, "H")
        data_4 = self.recorder(self.eta_3, matrix, "H")
        data_5 = self.recorder(self.eta_4, matrix, "H")
        
        return [data_1,data_2,data_3,data_4,data_5]
    
    def plot(self):
        if self.opt == 0:
            dt = self.result(self.exp_matrix())
        else:
            dt = self.result(self.mat_wc)
        m = self.mu
        tt = np.arange(1, 1001)
        fig, ax1 = plt.subplots(figsize=(15,10))
        ax1.fill_between(tt, dt[0][0] - dt[0][1]/np.sqrt(10),
                         dt[0][0] + dt[0][1]/np.sqrt(10),
                         alpha=.1)
        ax1.fill_between(tt, dt[1][0] - dt[1][1]/np.sqrt(10),
                         dt[1][0] + dt[1][1]/np.sqrt(10),
                         alpha=.1)
        ax1.fill_between(tt, dt[2][0] - dt[2][1]/np.sqrt(10),
                         dt[2][0] + dt[2][1]/np.sqrt(10),
                         alpha=.1)
        ax1.fill_between(tt, dt[3][0] - dt[3][1]/np.sqrt(10),
                         dt[3][0] + dt[3][1]/np.sqrt(10),
                         alpha=.1)
        ax1.fill_between(tt, dt[4][0] - dt[4][1]/np.sqrt(10),
                         dt[4][0] + dt[4][1]/np.sqrt(10),
                         alpha=.1)
        ax1.plot(tt, dt[0][0], label = 'FTL')
        ax1.plot(tt, dt[1][0], label = 'Hedge')
        ax1.plot(tt, dt[2][0], label = 'Reparametrized Hedge')
        ax1.plot(tt, dt[3][0], label = 'Hedge - Anytime version')
        ax1.plot(tt, dt[4][0], label = 'Reparametrized Hedge - Anytime version')
        ax1.set_xlabel('t', fontsize = 25)
        if self.opt == 0:
            ax1.set_ylabel('Emperical pseudo regret', fontsize = 25)
        else:
            ax1.set_ylabel('Regret', fontsize = 25)
        ax1.tick_params(axis="y", labelsize = 18)
        ax1.tick_params(axis="x", labelsize = 18)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width, box.height*0.8])
        ax1.legend(loc='center left', bbox_to_anchor=(0.25, -0.3), prop=dict(size=18))
        plt.title("Emperical comparison of FTL and Hedge \n", fontsize = 25)
        if self.opt == 0:
            plt.suptitle("$\mu = {}$".format(m), x = 0.19, y = 0.71, fontsize = 22)
        else:
            plt.suptitle("with adversarial sequence", x = 0.28, y = 0.71, fontsize = 22)
        plt.show()
        return
    
#%%
mu1 = 0.25
mu2 = 0.5 - (1/8)
mu3 = 0.5 - (1/16)

#%%
test1 = Q5(mu1, 0)
test1.plot()

#%%
test2 = Q5(mu2, 0)
test2.plot()

#%%
test3 = Q5(mu3, 0)
test3.plot()

#%% Worst case for FTL
test4 = Q5(mu3, 1)
test4.plot()
