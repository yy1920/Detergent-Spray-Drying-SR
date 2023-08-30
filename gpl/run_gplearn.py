#!/usr/bin/python
import argparse
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import glob
from calc_metrics import calc_metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*/*_train*.txt")
print(train_files)
#pop_sizes = [50,100,200,300,400,500,600,700,800,900,1000]
pop_sizes = [1100,1200,1300,1400,1500]


def rmse(scorer,X,y_true):
    y_pred = scorer.predict(X)
    
    return np.sqrt(np.abs(mean_squared_error(y_true,y_pred)))

for file in train_files:
    target_string = file.split("_train")
    data = pd.read_csv(file, sep=" ", header=None).to_numpy()
    data_test = pd.read_csv(target_string[0]+"_test"+target_string[1], sep=" ", header=None).to_numpy()
    
    X_train = data[:, :-1]
    X_test = data_test[:, :-1]
    y_train = data[:, -1]
    y_test = data_test[:, -1]
    
    '''
    # code for gridsearching population_size parameter
    model = SymbolicRegressor()
    parameters = {'population_size':pop_sizes}
    clf = GridSearchCV(model, parameters,scoring=rmse)
    clf.fit(X_train, y_train)
    print(clf.cv_results_)
    '''
    est_gp = SymbolicRegressor(population_size=1000,
                               function_set = ['add', 'sub', 'mul', 'div','sqrt','inv','sin','cos'],
                               n_jobs=-1, metric="rmse", random_state=0)
    est_gp.fit(X_train, y_train)
    print(file)
    print(est_gp._program)
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
    
