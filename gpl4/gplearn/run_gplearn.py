#!/usr/bin/python
import argparse
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import glob
from calc_metrics import calc_metrics
train_files = glob.glob("//nobackup/dcjk57/Test_Data/*/*_train*.txt")
print(train_files)

for file in train_files:
    data = pd.read_csv(file, sep=" ", header=None)
    train_ind = int(data.shape[0] * 0.8)
    data = data.to_numpy()
    X_train = data[:train_ind, :-1]
    X_test = data[train_ind:, :-1]
    y_train = data[:train_ind, -1]
    y_test = data[train_ind:, -1]

    est_gp = SymbolicRegressor(population_size=250,
                           function_set = ['add', 'sub', 'mul', 'div','sqrt','inv','sin','cos'],
                           n_jobs=-1, random_state=0)
    PA = est_gp.fit(X_train, y_train)
    print(file)
    print(est_gp._program)
    
    y_test_pred = est_gp.predict(X_test)
    print(calc_metrics(y_test, y_test_pred))
    target_string = file.split("_train")
    try:
        for i in range(1,4):
            new_string  = target_string[0] + "_extrap_"+str(i) + target_string[1]
            print(new_string)
            extrap_data = pd.read_csv(new_string, header=None, sep=" ").to_numpy()
            X_extrap = extrap_data[:, :-1]
            y_extrap = extrap_data[:, -1]
            y_extrap_pred = est_gp.predict(X_extrap)
            print("Extrap "+str(i)+": ",calc_metrics(y_extrap, y_extrap_pred))
    except Exception as e:
        pass

    '''
    target_string = target_string[0] + "_extrap" + target_string[1]
    extrap_data = pd.read_csv(target_string, header=None, sep=" ").to_numpy()
   
    X_extrap = extrap_data[:, :-1]
    y_extrap = extrap_data[:, -1]
    y_extrap_pred = est_gp.predict(X_extrap)
    print(calc_metrics(y_extrap, y_extrap_pred))
    '''