# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 12:28:16 2019

@author: cryin
"""
from __future__ import print_function
import numpy as np

#%%
WORLD_SIZE_X = 3
WORLD_SIZE_Y = 4
INITIAL = [0, 0]
DISCOUNT = 1

#%%
# left, up, right, down
actions = ['L', 'U', 'R', 'D']

actionProb = []
for i in range(0, WORLD_SIZE_Y):
    actionProb.append([])
    for j in range(0, WORLD_SIZE_X):
        actionProb[i].append(dict({'L':0.25, 'U':0.25, 'R':0.25, 'D':0.25}))
        
nextState = []
actionReward = []
for i in range(0, WORLD_SIZE_Y):
    nextState.append([])
    actionReward.append([])
    for j in range(0, WORLD_SIZE_X):
        next = dict()
        reward = dict()
        
        if [i, j] == [0, 0]:
            next['U'] = [i, j]
            reward['U'] = -1.0
            
            next['D'] = [i + 1, j]
            reward['D'] = -1.0
            
            next['L'] = [i, j]
            reward['L'] = -1.0
            
            next['R'] = [i, j + 1]
            reward['R'] = -1.0
            
        if [i, j] == [0, 1]:
            next['U'] = [i, j]
            reward['U'] = -1.0
            
            next['D'] = [i + 1, j]
            reward['D'] = -6.0
            
            next['L'] = [i, j - 1]
            reward['L'] = -1.0
            
            next['R'] = [i, j + 1]
            reward['R'] = -1.0
        
        if [i, j] == [0, 2]:
            next['U'] = [i, j]
            reward['U'] = -1.0
            
            next['D'] = [i + 1, j]
            reward['D'] = -1.0
            
            next['L'] = [i, j - 1]
            reward['L'] = -1.0
            
            next['R'] = [i, j]
            reward['R'] = -1.0

        if [i, j] == [1, 0]:
            next['U'] = [i - 1, j]
            reward['U'] = -1.0
            
            next['D'] = [i + 1, j]
            reward['D'] = -6.0
            
            next['L'] = [i, j]
            reward['L'] = -1.0
            
            next['R'] = [i, j]
            reward['R'] = -1.0
            
        if [i, j] == [1, 1]:
            next['U'] = [i - 1, j]
            reward['U'] = -1.0
            
            next['D'] = [i + 1, j]
            reward['D'] = -1.0
            
            next['L'] = [i, j]
            reward['L'] = -1.0
            
            next['R'] = [i, j]
            reward['R'] = -1.0
            
        if [i, j] == [1, 2]:
            next['U'] = [i - 1, j]
            reward['U'] = -1.0
            
            next['D'] = [i + 1, j]
            reward['D'] = -11.0
            
            next['L'] = [i, j]
            reward['L'] = -1.0
            
            next['R'] = [i, j]
            reward['R'] = -1.0
            
        if [i, j] == [2, 0]:
            next['U'] = [i - 1, j]
            reward['U'] = -1.0
            
            next['D'] = [i + 1, j]
            reward['D'] = -1.0
            
            next['L'] = [i, j]
            reward['L'] = -1.0
            
            next['R'] = [i, j + 1]
            reward['R'] = -1.0
            
        if [i, j] == [2, 1]:
            next['U'] = [i - 1, j]
            reward['U'] = -6.0
            
            next['D'] = [i + 1, j]
            reward['D'] = -1.0
            
            next['L'] = [i, j - 1]
            reward['L'] = -6.0
            
            next['R'] = [i, j + 1]
            reward['R'] = -11.0
            
        if [i, j] == [2, 2]:
            next['U'] = [i - 1, j]
            reward['U'] = -1.0
            
            next['D'] = [i, j]
            reward['D'] = -1.0
            
            next['L'] = [i, j - 1]
            reward['L'] = -1.0
            
            next['R'] = [i, j]
            reward['R'] = -1.0
            
        if [i, j] == [3, 0]:
            next['U'] = [i - 1, j]
            reward['U'] = -6.0
            
            next['D'] = [i, j]
            reward['D'] = -1.0
            
            next['L'] = [i, j]
            reward['L'] = -1.0
            
            next['R'] = [i, j]
            reward['R'] = -1.0
            
        if [i, j] == [3, 1]:
            next['U'] = [i - 1, j]
            reward['U'] = -1.0
            
            next['D'] = [i, j]
            reward['D'] = -1.0
            
            next['L'] = [i, j]
            reward['L'] = -1.0
            
            next['R'] = [i, j + 1]
            reward['R'] = -1.0
            
        if [i, j] == [3, 2]:
            next['U'] = [i, j]
            reward['U'] = 0.0
            
            next['D'] = [i, j]
            reward['D'] = 0.0
            
            next['L'] = [i, j]
            reward['L'] = 0.0
            
            next['R'] = [i, j]
            reward['R'] = 0.0
            
        nextState[i].append(next)
        actionReward[i].append(reward)

#%% Random Policy
world = np.zeros((WORLD_SIZE_Y, WORLD_SIZE_X))
Value = 0
while True:
    newWorld = np.zeros((WORLD_SIZE_Y, WORLD_SIZE_X))
    Delta = 0
    for i in range(0, WORLD_SIZE_Y):
        for j in range(0, WORLD_SIZE_X):
            for action in actions:
                newPosition = nextState[i][j][action]
                # bellman equation
                newWorld[i, j] += actionProb[i][j][action] * (actionReward[i][j][action] + DISCOUNT * world[newPosition[0], newPosition[1]])
                newValue = np.sum(newWorld)
                newDelta = np.max([Delta, np.abs(Value - newValue)])
    if newDelta < 1e-4:
        print('Random Policy')
        print(newWorld)
        break
    Value = newValue
    world = newWorld
                
#%% Optimal Policy
world = np.zeros((WORLD_SIZE_Y, WORLD_SIZE_X))
Value = 0
while True:
    # keep iteration until convergence
    newWorld = np.zeros((WORLD_SIZE_Y, WORLD_SIZE_X))
    Delta = 0
    for i in range(0, WORLD_SIZE_Y):
        for j in range(0, WORLD_SIZE_X):
            values = []
            for action in actions:
                newPosition = nextState[i][j][action]
                # value iteration
                values.append(actionReward[i][j][action] + DISCOUNT * world[newPosition[0], newPosition[1]])
            newWorld[i][j] = np.max(values)
            newValue = np.sum(newWorld)
            newDelta = np.max([Delta, np.abs(Value - newValue)])
    if newDelta < 1e-4:
        print('Optimal Policy')
        print(newWorld)
        break
    Value = newValue
    world = newWorld

    
    
    
    
    