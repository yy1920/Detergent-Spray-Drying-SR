# Symbolic-Regression Boosting (SyRBo)
# copyright 2021 moshe sipper  
# www.moshesipper.com 
import argparse
import glob
USAGE = '  python syrbo.py resdir dsname n_replicates stages'
import pandas as pd
from string import ascii_lowercase
from random import choices
from sys import argv, stdin
from os import makedirs
from os.path import exists
from pandas import read_csv
import numpy as np

from time import process_time
import sys
sys.path.append('gplearn/')
from gplearnx.genetic import SymbolicRegressor
from gplearnx.functions import make_function, mul2,add2
from gplearnx.fitness import _Fitness
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Ridge
import numpy as np


def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(np.abs(mean_squared_error(y_true, y_pred)))
    target_range = np.max(y_true) - np.min(y_true)
    if target_range == 0:
        target_range = np.min(y_true)
    rmspe = (rmse / (target_range*0.5)) * 100
    #rmspe = np.sqrt(np.average(np.square((y_true-y_pred)/y_true)))*100
    
    r2 = r2_score(y_true, y_pred)
    return rmse, target_range,rmspe, r2

def _if3(x1, x2, x3): 
    return np.where(np.greater_equal(x1, np.zeros(x1.shape)), x2, x3)

def _if4(x1, x2, x3, x4): 
    return np.where(np.greater_equal(x1, x2), x3, x4)

def exponent(x):
  a = np.exp(x)
  a[~np.isfinite(a)] = 0
  return a

def scaled_rmse(y_true,y,w):
  #b = np.cov(y_true,y)/np.var(y_true)                                                                                                                                                                        
  #b = np.sum(((y-np.average(y,weights=w))*(y_true-np.average(y_true,weights=w)))/(y-np.average(y,weights=w))**2)                                                                                             
  try:
    clf = Ridge(alpha=1.0)
    clf.fit(y.reshape(-1,1),y_true.reshape(-1,1))
    b = (clf.coef_).reshape((-1,))
    a = clf.intercept_
  except Exception as e:
    print(e)
    return 1000000
  #if not np.isfinite(b):                                                                                                                                                                                     
    #return 1000000                                                                                                                                                                                           
  #a = np.average(y,weights=w) - b*np.average(y_true,weights=w)                                                                                                                                               
  #print("a,b ",a,b)                                                                                                                                                                                          
  #print("fitness ",((y_true-(b*y+a)) ** 2))                                                                                                                                                                  
  return np.sqrt(np.abs(np.average(((y_true-(b*y+a)) ** 2), weights=w)))

def scaled_rmspe(y_true,y,w):
                                                                                                                                                                     
  try:
    clf = Ridge(alpha=1.0)
    clf.fit(y_true.reshape(-1,1),y.reshape(-1,1))
    b = (clf.coef_).reshape((-1,))
    a = clf.intercept_
  except Exception as e:
    print(e)
    return 1000000

  return np.sqrt(np.abs(np.average((((y_true-(b*y+a))/y_true) ** 2), weights=w)))


scaled_rmse_fitness = _Fitness(function=scaled_rmse, greater_is_better=False)
scaled_rmspe_fitness = _Fitness(function=scaled_rmspe, greater_is_better=False)

if3 = make_function(function=_if3, name='if3', arity=3)
if4 = make_function(function=_if4, name='if4', arity=4)
exponential = make_function(function=exponent, name='exp', arity=1)
def square(x):
    a = np.square(x)
    a[~np.isfinite(a)] = 0
    return a
squared = make_function(function=square, name='sqr',arity=1)


class SyRBo:
    def __init__(self, stages=-1, population_size=-1, generations=-1): 
        self.stages = stages
        self.population_size = population_size
        self.generations = generations
        self.boosters = []
   
    def fit(self, X, y):
        for stage in range(self.stages):
            gp = SymbolicRegressor(population_size=self.population_size, generations=self.generations,parsimony_coefficient=0.001,\
                                   function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'inv','sin','cos'),n_jobs=-1, metric=scaled_rmse_fitness)
                                   #function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min', exponential),n_jobs=-1, metric=scaled_rmse_fitness)
            gp.fit(X, y)
            y_pred = gp.predict(X)
            # Add the linear scaling parameters to the program equation
            
            clf = Ridge(alpha=1.0)
            clf.fit(y_pred.reshape(-1,1),y.reshape(-1,1))
            b = (clf.coef_).reshape((-1,))[0]
            a = clf.intercept_
            gp._program.program.insert(0,mul2)
            gp._program.program.insert(0,add2)
            gp._program.program.append(b)
            gp._program.program.append(a)
            
            self.boosters.append(gp)
            p = np.nan_to_num(gp.predict(X))
            y -= p
        '''    
        pred = np.zeros(X.shape[0])
        for i in range(self.stages): 
            pred += np.nan_to_num(self.boosters[i].predict(X))
        y_pred = np.nan_to_num(pred)
    
        clf = Ridge(alpha=1.0)
        clf.fit(y.reshape(-1,1),y_pred.reshape(-1,1))
        b = (clf.coef_).reshape((-1,))[0]
        a = clf.intercept_
        gp = self.boosters[-1]
        gp._program.program.insert(0,mul2)
        gp._program.program.insert(0,add2)
        gp._program.program.append(b)
        gp._program.program.append(a)
        ''' 
    def predict(self, X): 
        pred = np.zeros(X.shape[0])
        for i in range(self.stages): 
            pred += np.nan_to_num(self.boosters[i].predict(X))
        return np.nan_to_num(pred)
    
    def score(self, X, y, sample_weight=None):
        return mean_absolute_error(y, self.predict(X))
    
    def get_params(self, deep=True):
        return { 'stages': self.stages, 'population_size': self.population_size, 'generations': self.gens } 

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def get_equation(self):
        result_equation = str(self.boosters[0]._program)
        for i in range(1,len(self.boosters)):
            result_equation = "add("+result_equation+","+str(self.boosters[i]._program)+")"
        return result_equation
            
# end class 

Algorithms = [SyRBo, SymbolicRegressor]
AlgParams ={
SyRBo: { 'stages': -1, 'population_size': 200, 'generations': 200 }, 
SymbolicRegressor: {   'population_size': 200, 'generations': 200 }
}

def rand_str(n): return ''.join(choices(ascii_lowercase, k=n))

def fprint(fname, s):
    if stdin.isatty(): print(s) # running interactively 
    with open(Path(fname),'a') as f: f.write(s)

def get_args():        
    if len(argv) == 5: 
        resdir, dsname, n_replicates, stages =\
            argv[1]+'/', argv[2], int(argv[3]), int(argv[4])
    else: # wrong number of args
        exit('-'*80                       + '\n' +\
             'Incorrect usage:'           + '\n' +\
             '  python ' + ' '.join(argv) + '\n' +\
             'Please use:'                + '\n' +\
             USAGE                        + '\n' +\
             '-'*80)
                    
    if not exists(resdir): makedirs(resdir)
    fname = resdir + dsname + '_' + rand_str(6) + '.txt'
    return fname, resdir, dsname, n_replicates, stages

def get_dataset(dsname):
    if dsname ==  'regtest':
        X, y = make_regression(n_samples=10, n_features=2, n_informative=1)
    elif dsname == 'boston':
        X, y = load_boston(return_X_y=True)
    elif dsname == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
    elif dsname in regression_dataset_names: # PMLB datasets
        X, y = fetch_data(dsname, return_X_y=True, local_cache_dir='pmlb') #../datasets/pmlbreg/
    else:
        try: # dataset from openml?
            X, y = fetch_openml(dsname, return_X_y=True, as_frame=False, cache=False)
        except:
            try: # a csv file in datasets folder?
                data = read_csv('../datasets/' + dsname + '.csv', sep=',')
                array = data.values
                X, y = array[:,0:-1], array[:,-1] # target is last col
                # X, y = array[:,1:], array[:,0] # target is 1st col
            except Exception as e: 
                print('looks like there is no such dataset')
                exit(e)
                
    X = normalize(X, norm='l2')
    # scaler = RobustScaler()
    # X = scaler.fit_transform(X)
             
    n_samples, n_features = X.shape
    return X, y, n_samples, n_features

def print_params(fname, dsname, n_replicates, n_samples, n_features, stages):
    fprint(fname,\
        'dsname: ' + dsname + '\n' +\
        'n_samples: ' + str(n_samples) + '\n' +\
        'n_features: ' + str(n_features) + '\n' +\
        'n_replicates: ' + str(n_replicates) + '\n' +
        'stages: ' + str(stages) + '\n')
# main

##############
train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*/*_train*.txt")
#train_files = glob.glob("//nobackup/dcjk57/Test_Data/pg_data/*_train*.txt")
for file in train_files:
    data_train = pd.read_csv(file, sep=" ", header=None).to_numpy()
    target_string = file.split("_train")
    data_test = pd.read_csv(target_string[0]+"_test"+target_string[1], sep=" ", header=None).to_numpy()
    #train_ind = int(data.shape[0] * 0.8)
    X_train = data_train[:, :-1]
    X_test = data_test[:, :-1]
    y_train = data_train[:, -1]
    y_test = data_test[:, -1]
    est_gp = SyRBo(population_size=1000,generations=20,stages=4)    
    est_gp.fit(X_train, y_train)
    y_train_pred = est_gp.predict(X_train)
    print(file)
    print(est_gp.get_equation())
    print("Train metrics")
    print(calc_metrics(y_train,y_train_pred))
    y_test_pred = est_gp.predict(X_test)
    print("Test: ")
    print(calc_metrics(y_test, y_test_pred))

    folder_name = target_string[0]
    try:
        for i in range(1,4):
            new_string  = target_string[0] + "_extrap_"+str(i) + target_string[1]
            print(new_string)
            extrap_data = pd.read_csv(new_string, header=None, sep=" ").to_numpy()
            X_extrap = extrap_data[:, :-1]
            y_extrap = extrap_data[:, -1]
            y_extrap_pred = est_gp.predict(X_extrap)
            print("Extrap "+str(i)+": ",calc_metrics(y_extrap, y_extrap_pred))
    except:
        pass
