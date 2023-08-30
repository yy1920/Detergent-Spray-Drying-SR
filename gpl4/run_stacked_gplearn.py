#!/usr/bin/python
import argparse
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import glob
from calc_metrics import calc_metrics
from gplearn.fitness import _fitness_map, _Fitness
from sklearn.linear_model import Ridge

#train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*/*_train*.txt"
train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/pg_data_dimless/*_train*.txt")
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
meta = "ridge"
for file in train_files:
    data = pd.read_csv(file, sep=" ", header=None).to_numpy()
    target_string = file.split("_train")
    data_test = pd.read_csv(target_string[0]+"_test"+target_string[1],sep=" ", header=None).to_numpy()
    X_train = data[:, :-1]
    X_test = data_test[:, :-1]
    y_train = data[:, -1]
    y_test = data_test[:, -1]
    est_gp = SymbolicRegressor(population_size=1000,
                           function_set = ['add', 'sub', 'mul', 'div','sqrt','inv','sin','cos'],
                               n_jobs=-1, random_state=0, metric=scaled_rmse_fitness)
    est_gp.fit(X_train, y_train)
    print(file)
    y_train_pred = est_gp.stacked_predict2(X_train, meta=meta)
    y_test_pred = est_gp.stacked_predict2(X_test,meta=meta)
    print(str(est_gp._program))
    print("Train metrics")
    print(calc_metrics(y_train, y_train_pred))
    print("Test metrics")
    print(calc_metrics(y_test, y_test_pred))

    try:
        for i in range(1,4):
            new_string  = target_string[0] + "_extrap_"+str(i) + target_string[1]
            print(new_string)
            extrap_data = pd.read_csv(new_string, header=None, sep=" ").to_numpy()
            X_extrap = extrap_data[:, :-1]
            y_extrap = extrap_data[:, -1]
            y_extrap_pred = est_gp.stacked_predict2(X_extrap,meta=meta)
            print("Extrap "+str(i)+": ",calc_metrics(y_extrap, y_extrap_pred))
    except Exception as e:
        pass
