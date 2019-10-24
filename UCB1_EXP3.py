# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#%%
class Q5:
    
    # Class Variable: the number of running algorithm
    r = 10
    T = 10000
    best_mu = 0.5
    
    def __init__(self, mu, k, b):
        self.subopt = mu
        self.k = k
        self.breakUCB1 = b
        
    # Generate matrix for experiment
    def gen_matrix(self):
        mat = [np.random.binomial(1, self.best_mu, self.T)]
        for i in range(self.k-1):
            mat_ = np.random.binomial(1, self.subopt, self.T)
            mat = np.append(mat, [mat_], axis = 0)
        
        return mat
    
    # Worst case for UCB1
    def gen_matrix_adv(self):
        delta = self.best_mu - self.subopt
        geom = np.around(np.geomspace(5, 6736, num=14, endpoint=False)).astype(int)
#        geom[-1] = geom[-1]+6
        Mat = [[]]
        for i in range(self.k-1):
            Mat = np.append(Mat, [[]], axis = 0)
        t = 0
        while True:
            j = geom[t]
            mat = [np.random.binomial(1, 1, j)]
            for i in range(self.k-1):
                mat_ = np.random.binomial(1, 1-delta, j)
                mat = np.append(mat, [mat_], axis = 0)
            Mat = np.append(Mat, mat, axis = 1).astype(int)
            j = geom[t+1]
            mat = [np.random.binomial(1, delta, j)]
            for i in range(self.k-1):
                mat_ = np.random.binomial(1, 0, j)
                mat = np.append(mat, [mat_], axis = 0)
            Mat = np.append(Mat, mat, axis = 1).astype(int)
            t = t+2
            if t == len(geom):
                break
            
        return Mat
    
    def upper_bound(self, t, N):
        return np.sqrt( (3*np.log(t)) / (2*N) )
    
    def improved_upper_bound(self, t, N):
        return np.sqrt( np.log(t) / N )
    
    # UCB1
    def UCB1(self, matrix, c):
        reward_sum = np.zeros(self.k)
        num_play = np.repeat(1, self.k)
        ucb = np.zeros(self.k)
        regret = np.zeros(self.T)
        delta = self.best_mu - self.subopt
                
        # Initialization
        for i in range(self.k):
            reward_sum[i] = matrix[i,i]
            regret[i] = delta
        regret[0] = 0
        
        # Play the game
        t = self.k + 1
        while True:
            for i in range(self.k):
                if c == "impr":
                    bound = self.improved_upper_bound(t, num_play[i])
                else:
                    bound = self.upper_bound(t, num_play[i])
                ucb[i] = (reward_sum[i]/num_play[i]) + bound
            action = np.argmax(ucb)
            reward = matrix[action, (t-1)]
            num_play[action] += 1
            reward_sum[action] += reward
            regret[t] = delta * (num_play[1:].sum())
            t = t + 1
            if t == self.T:
                break
        return regret

    # Learning rates
    def eta(self, t):
        return np.sqrt(np.log(self.k)/(t*self.k))

    # EXP3
    def EXP3(self, matrix):
        prob = np.zeros(self.k)
        L_tilda = np.zeros(self.k)
        num_play = np.zeros(self.k)
        regret = np.zeros(self.T)
        delta = self.best_mu - self.subopt
        
        t = 1
        while True:
            prob = np.exp(-self.eta(t)*L_tilda)
            prob = prob / prob.sum()
            action = np.random.choice(range(self.k), 1, p = prob)[0]
            num_play[action] += 1
            loss = 1 - matrix[action, (t-1)]
            L_tilda[action] += loss / prob[action]
            regret[(t-1)] = delta * (num_play[1:].sum())
            t = t + 1
            if t == (self.T + 1):
                break
            
        return regret

    # Generate r=10 Matrices for the experiment
    def exp_matrix(self):
        if self.breakUCB1 == 0:
            matrix = [self.gen_matrix()]
        else:
            matrix = [self.gen_matrix_adv()]
        tmp = Q5.r - 1
        for i in range(tmp):
            if self.breakUCB1 == 0:
                matrix_ = [self.gen_matrix()]
            else:
                matrix_ = [self.gen_matrix_adv()]
            matrix = np.append(matrix, matrix_, axis = 0)

        return matrix
    
    # Record the results
    def recorder(self, matrix, alg, c):
        if alg == "UCB1":
            result = [self.UCB1(matrix[0], c)]
            tmp = self.r - 1
            for i in range(tmp):
                rlt = [self.UCB1(matrix[(i+1)], c)]
                result = np.append(result, rlt, axis = 0)
                
            result_mean = np.average(result, axis=0)
            result_std = np.std(result, axis=0)
            
            return [result_mean, result_std]
        
        if alg == "EXP3":
            result = [self.EXP3(matrix[0])]
            tmp = self.r - 1
            for i in range(tmp):
                rlt = [self.EXP3(matrix[(i+1)])]
                result = np.append(result, rlt, axis = 0)
                
            result_mean = np.average(result, axis=0)
            result_std = np.std(result, axis=0)
            
            return [result_mean, result_std]
    
    def result(self, matrix):
        data_1 = self.recorder(matrix, "UCB1", "")
        data_2 = self.recorder(matrix, "UCB1", "impr")
        data_3 = self.recorder(matrix, "EXP3", "")
        
        return [data_1, data_2, data_3]
    
    def data(self):
        return self.result(self.exp_matrix())
    
    def plot(self):
        dt = self.result(self.exp_matrix())
        m = self.subopt
        k = self.k
        tt = np.arange(1, 10001)
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
        ax1.plot(tt, dt[0][0], label = 'UCB1')
        ax1.plot(tt, dt[1][0], label = 'Improved UCB1')
        ax1.plot(tt, dt[2][0], label = 'EXP3')
        ax1.set_xlabel('t', fontsize = 25)
        ax1.set_ylabel('Emperical pseudo-regret', fontsize = 25)
        ax1.tick_params(axis="y", labelsize = 18)
        ax1.tick_params(axis="x", labelsize = 18)
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width, box.height*0.8])
        ax1.legend(loc='center left', bbox_to_anchor=(0.25, -0.3), prop=dict(size=18))
        plt.title("Emperical comparison of UCB1 and EXP3 \n", fontsize = 25)
        if self.breakUCB1 == 0:
            plt.suptitle("$\mu$ = %s, K = %s" % (m, k), x = 0.22, y = 0.71, fontsize = 22)
        else:
            plt.suptitle("$\mu$ = %s, K = %s \n with adversarial sequence" % (m, k), x = 0.22, y = 0.71, fontsize = 22)
        plt.show()
        return
    
#%%
mu1 = 0.5 - (1/4)
mu2 = 0.5 - (1/8)
mu3 = 0.5 - (1/16)

#%%
test11 = Q5(mu1, 2, 0)
test11.plot()

#%%
test12 = Q5(mu1, 4, 0)
test12.plot()

#%%
test13 = Q5(mu1, 8, 0)
test13.plot()

#%%
test14 = Q5(mu1, 16, 0)
test14.plot()

#%%
test21 = Q5(mu2, 2, 0)
test21.plot()

#%%
test22 = Q5(mu2, 4, 0)
test22.plot()

#%%
test23 = Q5(mu2, 8, 0)
test23.plot()

#%%
test24 = Q5(mu2, 16, 0)
test24.plot()

#%%
test31 = Q5(mu3, 2, 0)
test31.plot()

#%%
test32 = Q5(mu3, 4, 0)
test32.plot()

#%%
test33 = Q5(mu3, 8, 0)
test33.plot()

#%%
test34 = Q5(mu3, 16, 0)
test34.plot()

#%% Break UCB


#%%
test41 = Q5(mu1, 2, 1)
test41.plot()

#%%
test42 = Q5(mu1, 4, 1)
test42.plot()

#%%
test43 = Q5(mu1, 8, 1)
test43.plot()

#%%
test44 = Q5(mu1, 16, 1)
test44.plot()

#%%
test51 = Q5(mu2, 2, 1)
test51.plot()

#%%
test52 = Q5(mu2, 4, 1)
test52.plot()

#%%
test53 = Q5(mu2, 8, 1)
test53.plot()

#%%
test54 = Q5(mu2, 16, 1)
test54.plot()
