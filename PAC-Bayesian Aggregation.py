# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, zero_one_loss
import numpy as np
import pandas as pd
import os

#%% Section 1.1 Import the dataset
os.chdir('/Users/cryin/Desktop/KU/2019-2020/ATML/Assignment 03')
df = pd.read_csv('ionosphere0.txt', sep = ',', header = None)
dff = df.values

#%% Section 1.2 Convert 'g' and 'b' to '+1' and '-1' respectively
for i in range(len(dff)):
    if dff[i,-1] == "g":
        dff[i,-1] = "1"
    else:
        dff[i,-1] = "-1"
        
#%% Section 1.3 Split given dataset into two groups: Training set and Test set
Train = dff[0:200,]
Test = dff[201:351,]





#%% Section 2.1 Implementation of Jaakkola's method for the grid of gamma 
## split the dataset into two arrays: 'g' and 'b'
xx_g = Train[Train[:,-1] == "1"][:,:-1]
xx_b = Train[Train[:,-1] == "-1"][:,:-1]

## calculate the eclidean distance between two arrays
dist = euclidean_distances(xx_g,xx_b)

## For every example of good class
## the distance to the closest example of bad class
min_g = np.min(dist, axis=1)

## For every example of bad class
## the distance to the closest example of good class
min_b = np.min(dist, axis=0)

## merge those two result into one array
min_merged = np.hstack((min_g,min_b))

## calulate its median value to get a sigma_Jaakkola
sig_J = np.median(min_merged)

## gamma_Jaakkola
gam_J = 1/(2*(sig_J**2))





#%% Section 3.1 Specify the grid of parameters
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'gamma': [gam_J*0.0001, gam_J*0.01,
                        gam_J, gam_J*100, gam_J*10000],
              'kernel': ['rbf']}

#%% Section 3.2 Specify the scoring method
score = make_scorer(zero_one_loss, greater_is_better=False)

#%% Section 3.3 RBF kernel SVM tuned by 5-fold cross-validation
def CV_SVM(Train_data, Test_data):
    dim_train = len(Train_data[0])
    dim_test = len(Test_data[0])
    start = time.time()
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters, scoring = score, cv = KFold(5))
    clf.fit(Train_data[:,0:(dim_train-2)], Train_data[:,-1])
    result = zero_one_loss(clf.predict(Test_data[:,0:(dim_test-2)]), Test_data[:,-1])
    end = time.time()
    runtime = end - start
    ## Test loss
    return [result, runtime]

#%% Section 3.4 Result
cv_svm = CV_SVM(Train, Test)





#%% Section 4.1 function for calculating validation errors
def validation_error_array(num_subs, data_train):
    result_loss = np.array([])
    result_h = np.array([])
    for i in range(num_subs):
        
        ## subsets of r=35 points uniformly at random from the training set
        subs = np.random.choice(200, 35, replace = False)
        aaa = data_train[subs,:]
        
        ## the validation set
        bbb = np.delete(data_train, subs, axis = 0)
        
        ## gamma is randomly selected from the same grid as used in the baseline
        gam2 = np.random.choice([gam_J*0.0001, gam_J*0.01,
                                 gam_J, gam_J*100, gam_J*10000],1)
    
        ## train RBF kernel SVM for given subset
        ## C = 1 since the selection of C is unnecessary
        svc2 = svm.SVC(C=1, kernel = 'rbf', gamma = gam2[0])
        svc2.fit(aaa[:,0:33], aaa[:,34])
        result_h = np.append(result_h, svc2)
        
        ## compute zero_one_loss
        loss = zero_one_loss(svc2.predict(bbb[:,0:33]), bbb[:,-1])
        result_loss = np.append(result_loss, loss)
    
    return [result_loss, result_h]

#%% Section 4.2 function for updating the distribution rho
def rho_update(pi, num_subs, lamb, val_errors):
    rho_nume_array = np.array([])
    for i in range(num_subs):
        rho_nume = pi * np.exp(-lamb * 165 * (val_errors[i] - np.amin(val_errors)))
        rho_nume_array = np.append(rho_nume_array, rho_nume)
    rho_deno = np.sum(rho_nume_array)
    return rho_nume_array/rho_deno

#%% Section 4.3 function for computation of KL divergence 
def kl_comp(pi, rho):
    return np.log(1/pi)*np.sum(rho)+np.sum(rho*np.log(rho))

#%% Section 4.4 function for updating lambda
def lambda_update(pi, num_subs, val_errors, epsilon = 1e-04):
    lam = 1
    while True:
        rho = rho_update(pi, num_subs, lam, val_errors)
        
        ## update KL divergence
        kl = kl_comp(pi, rho)
        
        ## update lambda
        lam_new = 2/(np.sqrt(((2*165*np.sum(rho*val_errors))/(kl + np.log(2*np.sqrt(165)/0.05))) + 1) + 1)
        
        ## stopping criteria
        stop_cri = abs(lam_new - lam)
        
        ## the new value function
        lam = lam_new
        
        if(stop_cri < 1e-04):
            break
        
    return lam





#%% Section 5.1 PAC-Bayes-kl bound
def PAC_Bayes_kl(pi, rho, n):
    numerator = kl_comp(pi, rho) + np.log((2 * np.sqrt(n)) / 0.05)
    return numerator / (n)

#%% Section 5.2 Randomized classifier
def randomized_classifier(num_subs, rho, val_errors, data_test):
    num = np.random.choice(num_subs, 1, replace = False, p = rho)[0]
    predict = val_errors[1][num].predict(data_test[:,0:33])
    loss = zero_one_loss(predict, data_test[:,-1])
    
    return loss

#%% Section 5.3 Upper inverse of kl
def inverse_kl(x, z, eps = 1e-8):
    if x < 0 or x > 1 or z < 0:
        print('error')
        return
    if z == 0:
        return x
    
    y = (1 + x) / 2    
    step = (1 - x) / 4    

    if (x > 0):
        p0 = x
    else:
        p0 = 1
        
    while step > eps:
        if ((x * np.log(p0/y) + (1-x) * np.log((1-x)/(1-y))) < z):
            y = y + step
        else:
            y = y - step
        step = step / 2
    
    return y

#%% Section 5.4 Select 20 values of m in [1, m] geomatrically
mm = np.concatenate((np.array([1,2,3,4,5]),
                     np.rint(np.geomspace(6, 200, num = 15, endpoint = True))),axis = 0 )


#%% Section 5.5 rho-weighted majority vote
def weighted_majority_vote(num_subs, val_errors, rho, data_test):
    vt_val_array = np.array([np.float32(val_errors[1][0].predict(data_test[:,0:33]))*rho[0]])
    for i in range(1, num_subs):
        vt_val = np.array([np.float32(val_errors[1][i].predict(data_test[:,0:33]))*rho[i]])
        vt_val_array = np.append(vt_val_array, vt_val, axis = 0)
    vote_sum = np.sum(vt_val_array, axis = 0)
    predict = (vote_sum+1e-15) / np.abs(vote_sum+1e-15)   
    predict = predict.astype('int')
    predict = predict.astype('str')   
    loss = zero_one_loss(predict, data_test[:,-1])
    
    return loss

#%% Section 5.6 Final experiment
def experiment(data_train, data_test):
    time_array = np.array([])
    loss_array = np.array([])
    bound_array = np.array([])
    for i in mm.astype(int):
        pi_ = 1/i
        start = time.time()
        val_errors_ = validation_error_array(i, data_train)
        lamb_ = lambda_update(pi_, i, val_errors_[0], epsilon = 1e-04)
        rho_ = rho_update(pi_, i, lamb_, val_errors_[0])
        end = time.time()
        loss = weighted_majority_vote(i, val_errors_, rho_, data_test)
        record = end - start
        z_ = PAC_Bayes_kl(pi_, rho_, len(data_test))
        x_ = randomized_classifier(i, rho_, val_errors_, data_test)
        bound = inverse_kl(x_, z_, eps = 1e-8)
        loss_array = np.append(loss_array, loss)
        time_array = np.append(time_array, record)
        bound_array = np.append(bound_array, bound)
    return [loss_array, time_array, bound_array] 

#%% Section 5.7 Repitation
def rep(data_train, data_test, rep):
    temp_0 = experiment(data_train, data_test)
    array_1 = [temp_0[0]] # Our Method
    array_2 = [temp_0[1]] # t_m
    array_3 = [temp_0[2]] # Bound
    print(1)
    for i in range(rep - 1):
        temp = experiment(data_train, data_test)
        array_1 = np.append(array_1, [temp[0]], axis = 0)
        array_2 = np.append(array_2, [temp[1]], axis = 0)
        array_3 = np.append(array_3, [temp[2]], axis = 0)
        print(i+2)
    mean_1 = np.average(array_1, axis = 0)
    mean_2 = np.average(array_2, axis = 0)
    mean_3 = np.average(array_3, axis = 0)
    std_1 = np.std(array_1, axis = 0)
    std_3 = np.std(array_3, axis = 0)

    
    return [mean_1,mean_2,mean_3,std_1, std_3]

#%% Section 5.8 Result
result = rep(Train, Test, 100)

#%% Section 6 Plot the graph
fig, ax1 = plt.subplots(figsize=(15,10))

ax2 = ax1.twinx()
ax1.fill_between(mm, result[0]-1.96*result[3]/np.sqrt(150),
                 result[0]+1.96*result[3]/np.sqrt(150),
                 alpha=.1, color = 'black')
ax1.fill_between(mm, result[2]-1.96*result[4]/np.sqrt(150),
                 result[2]+1.96*result[4]/np.sqrt(150),
                 alpha=.1, color = 'b')
ax1.plot(mm, result[0], color = 'black', label = 'Our Method')
ax1.axhline(y=cv_svm[0], color = 'r', linestyle='-', label = 'CV SVM')
ax1.plot(mm, result[2], color = 'b', label = 'Bound')
ax2.plot(mm, result[1], color = 'black', linestyle = '--', label = '$t_m$')
ax2.axhline(y=cv_svm[1], color = 'r', linestyle='--', label = '$t_{CV}$')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.set_xlabel('m', fontsize = 25)
ax1.set_ylabel('Test Loss', fontsize = 25)
ax2.set_ylabel('Runtime (s)', fontsize = 25)
ax1.tick_params(axis="x", which="major", bottom=True, top=True,
                labelsize = 23, length=9, direction='in')
ax1.tick_params(axis="x", which="minor", bottom=True, top=True,
                labelsize = 23, length=6, direction='in')
ax1.tick_params(axis="y", labelsize = 20)
ax2.tick_params(axis="x", bottom=True, top=True, labelsize = 20)
ax2.tick_params(axis="y", labelsize = 20)

fig.legend(bbox_to_anchor=(0.67, 0.79), loc='upper left', borderaxespad=0.,
           prop=dict(size=18))
plt.show()