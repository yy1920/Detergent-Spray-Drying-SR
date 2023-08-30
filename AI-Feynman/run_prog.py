#!/usr/bin/python
import argparse
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import glob
from calc_metrics import calc_metrics
from aifeynman.S_run_aifeynman import run_aifeynman
#from aifeynman.sklearn_wrapper import AIFeynmanRegressor
from sklearn.metrics import mean_squared_error
train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*/*_train*.txt")
print(train_files)


def rmse(scorer,X,y_true):
    y_pred = scorer.predict(X)
    
    return np.sqrt(np.abs(mean_squared_error(y_true,y_pred)))

for file in train_files:
    target_string = file.split("_train")
    data = pd.read_csv(file, sep=" ", header=None).to_numpy()
#    data_test = pd.read_csv(target_string[0]+"_test"+target_string[1], sep=" ", header=None).to_numpy()
    
    X_train = data[:, :-1]
 #   X_test = data_test[:, :-1]
    y_train = data[:, -1]
 #   y_test = data_test[:, -1]
    dir = file.split("/")
    file_name = dir[-1]
    directory_name = "/".join(dir[:-1])
    est_gp = run_aifeynman(directory_name+"/", file_name, 30, "19ops.txt", polyfit_deg=4, NN_epochs=400)
'''
#  est_gp.fit(X_train, y_train)
  #  print(file)
  #  print(est_gp._program)
    print("Train metrics")
    y_train_pred = est_gp.predict(X_train)
    print(calc_metrics(y_train, y_train_pred))
    print("Test metrics")
    y_test_pred = est_gp.predict(X_test)
    print(calc_metrics(y_test, y_test_pred))
    
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
