#!/usr/bin/python
import argparse
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import glob
from calc_metrics import calc_metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from gplearn.fitness import _Fitness
from sklearn.linear_model import Ridge

train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*/*_train*.txt")
print(train_files)
#pop_sizes = [50,100,200,300,400,500,600,700,800,900,1000]
pop_sizes = [1100,1200,1300,1400,1500]

def scaled_rmse(y_true,y,w):
  #
  #
  try:
    #clf = Ridge(alpha=1.0)
    #clf.fit(y.reshape(-1,1), y_true.reshape(-1,1))
    #b = (clf.coef_).reshape((-1,))
    #a = clf.intercept_
    #b = np.cov(y_true,y)/np.var(y)
    if np.isnan(np.sum(y)):
      print("*******There is nan in y*******")
      return 10000
    denominator = (y-np.average(y,weights=w))**2
    denominator = np.where(denominator==0,1e-5,denominator)
    b = np.sum(((y-np.average(y,weights=w))*(y_true-np.average(y_true,weights=w)))/denominator)
    #print(b, type(b))
    
    a = np.average(y_true,weights=w) - b*np.average(y,weights=w)
    #print(a,type(a))
  except Exception as e:
    print(e)
    return 1000000
  #if not np.isfinite(b):
    #return 1000000
  #
  #print("a,b ",a,b)                                                                                                                                                                                                  
  #print("fitness ",((y_true-(b*y+a)) ** 2))                                                                                                                                                                          
  return np.sqrt(np.abs(np.average(((y_true-(b*y+a)) ** 2), weights=w)))


scaled_rmse_fitness = _Fitness(function=scaled_rmse, greater_is_better=False)


train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*/*_train*.txt")
                        

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
                               n_jobs=-1, metric="rmse", random_state=1)
    est_gp.fit(X_train, y_train)
    print(file)
    y_train_pred = est_gp.predict(X_train)
    #clf = Ridge(alpha=1.0)
    #clf.fit(y_train_pred.reshape(-1,1), y_train.reshape(-1,1))
    #b = (clf.coef_).reshape((-1,))[0]
    #a = clf.intercept_
    #denominator = (y_train_pred-np.average(y_train_pred))**2
    #denominator = np.where(denominator==0,1e-5,denominator)
    #b = np.sum(((y_train_pred-np.average(y_train_pred))*(y_train-np.average(y_train)))/denominator)
    #a = np.average(y_train) - b*np.average(y_train_pred)
    clf = Ridge(alpha=1.0)
    clf.fit(y_train_pred.reshape(-1,1), y_train.reshape(-1,1))
    b = (clf.coef_).reshape((-1,))[0]
    a = clf.intercept_
    print("("+str(est_gp._program)+") * "+str(b)+" + "+str(a))
    print("Train metrics")
    
    print(calc_metrics(y_train, b*y_train_pred+a))
    print("Test metrics")
    y_test_pred = est_gp.predict(X_test)
    print(calc_metrics(y_test, y_test_pred*b+a))
    
    try:
        for i in range(1,4):
            new_string  = target_string[0] + "_extrap_"+str(i) + target_string[1]
            print(new_string)
            extrap_data = pd.read_csv(new_string, header=None, sep=" ").to_numpy()
            X_extrap = extrap_data[:, :-1]
            y_extrap = extrap_data[:, -1]
            y_extrap_pred = est_gp.predict(X_extrap)
            print("Extrap "+str(i)+": ",calc_metrics(y_extrap, y_extrap_pred*b+a))
    except Exception as e:
        pass
    
