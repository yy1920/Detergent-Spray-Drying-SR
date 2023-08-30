#!/usr/bin/python
import argparse
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import glob
from calc_metrics import calc_metrics
from gplearn.fitness import _fitness_map, _Fitness
from sklearn.linear_model import Ridge

#train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*/*_train*.txt")
train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*_dimless/*_train*.txt")
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

def I_13_12_set_0(y,data):
    result = y*data[:,0]*np.square(data[:,1])/data[:,3]
    return result
def I_13_12_set_1(y,data):
    return y*data[:,0]*np.square(data[:,1])/data[:,4]
def I_13_12_set_2(y,data):
    return y*data[:,0]*np.square(data[:,2])/data[:,3]
def I_13_12_set_3(y,data):
    return y*data[:,0]*np.square(data[:,2])/data[:,4]

def I_29_16_set_0(y,data):
    return y*data[:,0]
def I_29_16_set_1(y,data):
    return y*data[:,1]

def I_32_17_set_0(y,data):
    return y*data[:,2]**2*data[:,0]*data[:,3]**3/data[:,1]
def I_32_17_set_1(y,data):
    return y*data[:,2]**2*data[:,0]*data[:,1]**2/data[:,4]**3
def I_32_17_set_2(y,data):
    return y*data[:,2]**2*data[:,0]*data[:,1]**2/data[:,5]**3
def I_32_17_set_3(y,data):
    return y*data[:,2]**2*data[:,0]*data[:,3]**2/data[:,4]
def I_32_17_set_4(y,data):
    return y*data[:,2]**2*data[:,0]*data[:,3]**2/data[:,5]

func_dict = {"I_13_120":I_13_12_set_0,
             "I_13_121":I_13_12_set_1,
             "I_13_122":I_13_12_set_2,
             "I_13_123":I_13_12_set_3,
             "I_29_160":I_29_16_set_0,
             "I_29_161":I_29_16_set_1,
             "I_32_170":I_32_17_set_0,
             "I_32_171":I_32_17_set_1,
             "I_32_172":I_32_17_set_2,
             "I_32_173":I_32_17_set_3,
             "I_32_174":I_32_17_set_4}

for file in train_files:
    data = pd.read_csv(file, sep=" ", header=None).to_numpy()
    target_string = file.split("_train")
    dataset_name = target_string[0].split("/")[-1]
    set_number = target_string[1].split("_")[-1].split(".")[0]
    if dataset_name == "pg":
      continue
    data_train = pd.read_csv("//nobackup/dcjk57/Test_Data_Final/"+dataset_name+"/"+dataset_name+"_train.txt", sep=" ", header=None).to_numpy()
    data_test_X = pd.read_csv(target_string[0]+"_test"+target_string[1], sep=" ", header=None).to_numpy()
    data_extrap_X = pd.read_csv(target_string[0]+"_extrap_1"+target_string[1], sep=" ", header=None).to_numpy()
    data_extrap_X2 = pd.read_csv(target_string[0]+"_extrap_2"+target_string[1], sep=" ", header=None).to_numpy()
    data_extrap_X3 = pd.read_csv(target_string[0]+"_extrap_3"+target_string[1], sep=" ", header=None).to_numpy()
    data_test = pd.read_csv("//nobackup/dcjk57/Test_Data_Final/"+dataset_name+"/"+dataset_name+"_test.txt", sep=" ", header=None).to_numpy()
    data_extrap = pd.read_csv("//nobackup/dcjk57/Test_Data_Final/"+dataset_name+"/"+dataset_name+"_extrap_1.txt", sep=" ", header=None).to_numpy()
    data_extrap2 = pd.read_csv("//nobackup/dcjk57/Test_Data_Final/"+dataset_name+"/"+dataset_name+"_extrap_2.txt", sep=" ", header=None).to_numpy()
    data_extrap3 = pd.read_csv("//nobackup/dcjk57/Test_Data_Final/"+dataset_name+"/"+dataset_name+"_extrap_3.txt", sep=" ", header=None).to_numpy()
    X_train = data[:, :-1]
    X_test = data_test_X[:, :-1]
    y_train = data[:, -1]
    y_test = data_test[:, -1]
    X_extrap = data_extrap_X[:,:-1]
    y_extrap = data_extrap[:,-1]
    X_extrap2 = data_extrap_X2[:,:-1]
    y_extrap2 = data_extrap2[:,-1]
    X_extrap3 = data_extrap_X3[:,:-1]
    y_extrap3 = data_extrap3[:,-1]
    
    y_train_y = data_train[:,-1]
    est_gp = SymbolicRegressor(population_size=1000,
                           function_set = ['add', 'sub', 'mul', 'div','sqrt','inv','sin','cos'],
                               n_jobs=-1, random_state=0, metric=scaled_rmse_fitness)
    est_gp.fit(X_train, y_train)
    print(file)
    y_train_pred = est_gp.predict(X_train)
    y_test_pred = est_gp.predict(X_test)

    print(str(est_gp._program))#+"*"+str(b)+"+"+str(a))
    #points = (est_gp.PA.get_pareto_points())
    #for point in points:
    #    print(point[0:2],point[2])
    try:
      transform = func_dict[dataset_name+set_number]
      print("Train metrics")
      print(calc_metrics(y_train_y, transform(y_train_pred,data_train)))#*b+a))
      y_test_pred = est_gp.predict(X_test)
      print("Test metrics")
      print(calc_metrics(y_test, transform(y_test_pred,data_test)))#*b+a))
      y_test_pred = est_gp.predict(X_extrap)
      print("Extrapolation  metrics")
      print(calc_metrics(y_extrap, transform(y_test_pred,data_extrap)))#*b+a))
      y_test_pred = est_gp.predict(X_extrap2)
      print(calc_metrics(y_extrap2, transform(y_test_pred,data_extrap2)))#*b+a))
      y_test_pred = est_gp.predict(X_extrap3)
      print(calc_metrics(y_extrap3, transform(y_test_pred,data_extrap3)))#*b+a))
    except Exception as e:
      print(e)

