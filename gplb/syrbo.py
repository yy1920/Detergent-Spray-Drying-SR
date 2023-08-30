# Symbolic-Regression Boosting (SyRBo)
# copyright 2021 moshe sipper  
# www.moshesipper.com 

USAGE = '  python syrbo.py resdir dsname n_replicates stages'

from string import ascii_lowercase
from random import choices
from sys import argv, stdin
from os import makedirs
from os.path import exists
from pandas import read_csv
import pandas as pd
import glob
from calc_metrics import calc_metrics
from pathlib import Path
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error
from time import process_time
# from copy import deepcopy


from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

def rmse(scorer,X,y_true):
    y_pred = scorer.predict(X)

    return np.sqrt(np.abs(mean_squared_error(y_true,y_pred)))

def _if3(x1, x2, x3): 
    return np.where(np.greater_equal(x1, np.zeros(x1.shape)), x2, x3)

def _if4(x1, x2, x3, x4): 
    return np.where(np.greater_equal(x1, x2), x3, x4)

if3 = make_function(function=_if3, name='if3', arity=3)
if4 = make_function(function=_if4, name='if4', arity=4)

class SyRBo:
    def __init__(self, stages=-1, population_size=-1, generations=-1): 
        self.stages = stages
        self.population_size = population_size
        self.generations = generations
        self.boosters = []
   
    def fit(self, X, y):
        for stage in range(self.stages):
            gp = SymbolicRegressor(population_size=self.population_size, generations=self.generations,\
                                   function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'inv','sin','cos'), random_state=0)
            gp.fit(X, y)
            self.boosters.append(gp)
            p = np.nan_to_num(gp.predict(X))
            y -= p
        
    def predict(self, X): 
        pred = np.zeros(X.shape[0])
        for i in range(self.stages): 
            pred += np.nan_to_num(self.boosters[i].predict(X))
        return np.nan_to_num(pred)
    
    def score(self, X, y, sample_weight=None):
        return mean_absolute_error(y, self.predict(X))
    
    def get_params(self, deep=True):
        return { 'stages': self.stages, 'population_size': self.population_size, 'generations': self.generations} 

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
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
    
    X_train = data_train[:, :-1]
    X_test = data_test[:, :-1]
    y_train = data_train[:, -1]
    y_test = data_test[:, -1]
    gens=[2,3,4,5,6]
    #model = SyRBo(population_size=1000,generations=20)                                                                   
    #parameters = {'stages':gens}                                                                                                
    #clf = GridSearchCV(model, parameters,scoring=rmse)                                                                                        
    #clf.fit(X_train, y_train)                                                                                                                 
    #print(clf.cv_results_)   
    est_gp = SyRBo(population_size=1100,generations=20,stages=4)
    est_gp.fit(X_train, y_train)
    print(file)
    print(" ")
    print("Train metrics")
    y_train_pred = est_gp.predict(X_train)
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
    
