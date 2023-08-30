#!/usr/bin/python
import argparse
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import glob
from calc_metrics import calc_metrics
from gplearn.fitness import _fitness_map, _Fitness
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/I_29_16/*_train*.txt")
print(train_files)

def scaled_rmse(y_true,y,w):
  try:
    clf = Ridge(alpha=1.0)
    #clf.fit(y_true.reshape(-1,1),y.reshape(-1,1))
    clf.fit(y.reshape(-1,1),y_true.reshape(-1,1))
    b = (clf.coef_).reshape((-1,))
    a = clf.intercept_
  except Exception as e:
    print(e)
    return 1000000
  return np.sqrt(np.abs(np.average(((y_true-(b*y+a)) ** 2), weights=w)))

scaled_rmse_fitness = _Fitness(function=scaled_rmse, greater_is_better=False)

for file in train_files:
    data = pd.read_csv(file, sep=" ", header=None)
    train_ind = int(data.shape[0] * 0.8)
    data = data.to_numpy()
    X_train = data[:train_ind, :-1]
    X_test = data[train_ind:, :-1]
    y_train = data[:train_ind, -1]
    y_test = data[train_ind:, -1]
    parameters = {'population_size':np.arange(100,1000,100),'p_crossover':[0.7,0.8,0.9,1], 'p_subtree_mutation':[0,0.01,0.02,0.05],'p_hoist_mutation':[0,0.01,0.02,0.05],'p_point_mutation':[0,0.01,0.02,0.05]}
    est_gp = SymbolicRegressor()
    clf = GridSearchCV(est_gp, parameters,n_jobs=-1)
    
    clf.fit(X_train,y_train)
    print(file)
    print(clf.cv_results_)
    
