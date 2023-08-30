#!/usr/bin/python
import argparse
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np
import glob
from calc_metrics import calc_metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol, lambdify, N
train_files = glob.glob("//nobackup/dcjk57/Test_Data_Final/*/*_train*.txt")
print(train_files)


def AIF_predict(expr,data):
    try:
        N_vars = data.shape[1] #WGL: no label passed
        variables = ["x%s" %i for i in np.arange(N_vars)]
        eq = parse_expr(expr)
        f = lambdify(variables, N(eq))
        output = f(*[x for x in data.T])
        return output
    except Exception as e:
        raise e
    return None


def rmse(scorer,X,y_true):
    y_pred = scorer.predict(X)
    
    return np.sqrt(np.abs(mean_squared_error(y_true,y_pred)))
chem_eng = "1.019905392223+(-sin(((exp(x3))**(-1)+1)))"
I_29_16 = "-1.78653240712045*x0**4 + 2.74265653938113*x0**3*x1 + 0.693478261551717*x0**3*x2 + 1.51893966797091*x0**3*x3 + 1.05809008885949*x0**3 - 8.90914108369895*x0**2*x1**2 + 2.313184902286*x0**2*x1*x2 - 0.610302506906684*x0**2*x1*x3 + 4.05496763865446*x0**2*x1 - 3.22927796065915*x0**2*x2**2 + 1.29979906947473*x0**2*x2*x3 + 0.15046637172591*x0**2*x2 - 1.61035966953248*x0**2*x3**2 - 1.19721783718238*x0**2*x3 + 0.916547527042824*x0**2 + 5.39603485787952*x0*x1**3 - 1.6173427947285*x0*x1**2*x2 + 0.383548360293983*x0*x1**2*x3 + 1.76441931827289*x0*x1**2 + 3.86511779058008*x0*x1*x2**2 - 6.63031604860491*x0*x1*x2*x3 - 0.812507810095345*x0*x1*x2 + 5.3167415030509*x0*x1*x3**2 - 1.66566774388912*x0*x1*x3 - 6.86009686529627*x0*x1 + 1.27933668152043*x0*x2**3 - 0.000487641774845338*x0*x2**2*x3 - 0.172199420407091*x0*x2**2 + 3.50679075029294*x0*x2*x3**2 - 3.11836370281625*x0*x2*x3 + 0.985158726474076*x0*x2 - 0.192325630918204*x0*x3**3 - 2.02646110837012*x0*x3**2 + 2.84384347457019*x0*x3 + 0.196014891336504*x0 - 2.65579982616603*x1**4 + 1.01123115252492*x1**3*x2 - 1.02738822067868*x1**3*x3 + 1.91103495717656*x1**3 - 1.48133741198047*x1**2*x2**2 + 3.60307516183734*x1**2*x2*x3 - 1.22789238226051*x1**2*x2 - 0.253656919771413*x1**2*x3**2 - 0.426339158867328*x1**2*x3 + 1.53639910649094*x1**2 + 1.20205181332284*x1*x2**3 + 2.90365915471729*x1*x2**2*x3 - 2.94865040383283*x1*x2**2 + 1.69359945768018*x1*x2*x3**2 - 6.31086756719515*x1*x2*x3 + 3.35114643527818*x1*x2 + 2.27577015639516*x1*x3**3 - 6.22113612429261*x1*x3**2 + 5.16620337634322*x1*x3 - 0.702704845621252*x1 - 3.84125205092138*x2**4 - 1.64291153517017*x2**3*x3 + 7.73906047122299*x2**3 - 3.55616229900037*x2**2*x3**2 + 4.69630959166568*x2**2*x3 - 5.21572955088661*x2**2 - 1.02102682070146*x2*x3**3 + 3.19921260306238*x2*x3**2 - 2.136739287076*x2*x3 + 0.905009209983947*x2 - 0.045554310405127*x3**4 - 0.171269324160102*x3**3 + 1.54297942292242*x3**2 - 1.47996378070159*x3 + 0.254932167900225"
pg_expr = "-0.621212000000*(sqrt(sqrt((x1*x0)))-1)"
I_32_17 = "tan(0.000001116206*(x2*(x3+asin((x0+x1)))))"
I_13_12 = "0.745558006473+(x2*(atan((x3-x4))-(x3-x4)))"
exprs = [chem_eng,I_29_16,pg_expr,I_32_17,I_13_12,None,None]
for i in range(len(train_files)):
    if exprs[i] == None:
        continue
    
    file = train_files[i]
    target_string = file.split("_train")
    data = pd.read_csv(file, sep=" ", header=None).to_numpy()
    data_test = pd.read_csv(target_string[0]+"_test"+target_string[1], sep=" ", header=None).to_numpy()
    
    X_train = data[:, :-1]
    X_test = data_test[:, :-1]
    y_train = data[:, -1]
    y_test = data_test[:, -1]
    
    
    print(file)
    print(exprs[i])
    print("Train metrics")
    y_train_pred = np.nan_to_num(AIF_predict(exprs[i],X_train))
    print(calc_metrics(y_train, y_train_pred))
    print("Test metrics")
    y_test_pred = np.nan_to_num(AIF_predict(exprs[i],X_test))
    print(calc_metrics(y_test, y_test_pred))
    
    try:
        for j in range(1,4):
            new_string  = target_string[0] + "_extrap_"+str(j) + target_string[1]
            print(new_string)
            extrap_data = pd.read_csv(new_string, header=None, sep=" ").to_numpy()
            X_extrap = extrap_data[:, :-1]
            y_extrap = extrap_data[:, -1]
            y_extrap_pred = np.nan_to_num(AIF_predict(exprs[i],X_extrap))
            print("Extrap "+str(j)+": ",np.nan_to_num(calc_metrics(y_extrap, y_extrap_pred)))
    except Exception as e:
        print(e)
        pass
    
