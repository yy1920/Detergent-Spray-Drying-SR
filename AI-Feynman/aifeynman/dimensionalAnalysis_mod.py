import numpy as np
import pandas as pd
from scipy.sparse.linalg import lsqr
from scipy.linalg import *
from sympy import Matrix
from sympy import symbols, Add, Mul, S
from .getPowers import getPowers

def dimensional_analysis(input,output,units):
    M = units[input[0]]
    for i in range(1,len(input)):
        M = np.c_[M, units[input[i]]]
    if len(input)==1:
        M = np.array(M)
        M = np.reshape(M,(len(M),1))
    print("in dim_ana func")
    params = getPowers(M,units[output])
    print("called get Powers")
    M = Matrix(M)
    B = M.nullspace()
    return (params, B)

# load the data from a file
def load_data(pathdir, filename):
    n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
    variables = np.loadtxt(pathdir+filename, usecols=(0,))
    for i in range(1,n_variables):
        v = np.loadtxt(pathdir+filename, usecols=(i,))
        variables = np.column_stack((variables,v))
    f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))
    return(variables.T,f_dependent)

def dimensionalAnalysis(pathdir, filename, eq_symbols):
    #file = pd.read_excel("units.xlsx")
    file = pd.read_csv("units.csv") #reading the units file
    print("in dimensional Analysis") 
    print(file.head()) #It is successfully printing the first 5 rows of file
    units = {}
    for i in range(len(file["Variable"])):
        val = [file["m"][i],file["s"][i],file["kg"][i],file["T"][i],file["V"][i]]#,file["Units"][i]] # changed ["cd"] to ["Variable"]
        val = np.array(val)
        units[file["Variable"][i]] = val
    print("FInished for loop in DA")
    dependent_var = eq_symbols[-1]

    file_sym = open(filename + "_dim_red_variables.txt" ,"w")
    file_sym.write(filename)
    file_sym.write(", ")
    print("opened this dim red var file")
    # load the data corresponding to the first line (from mystery_world)
    varibs = load_data(pathdir,filename)[0]
    deps = load_data(pathdir,filename)[1]
    print("Before this 2nd loop")
    # get the data in symbolic form and associate the corresponding values to it
    input = []
    print("eq_symbols "+str((eq_symbols)))
    for i in range(len(eq_symbols)-1):
        input = input + [eq_symbols[i]]
        vars()[eq_symbols[i]] = varibs[i]
    output = dependent_var
    print("finished 2nd loop")
    # Check if all the independent variables are dimensionless
    ok = 0
    print(units[str(input[1])])
    for j in range(len(input)):
        #if(units[input[j]].any()):
        if input[j] in units: #changed this line; checks if the input variable units are in the units.csv file data
            ok=1
            break
    print("finished third for loop")
    if ok==0:
        print("independent varaibles are dimensionless")
        dimless_data = load_data(pathdir, filename)[0].T
        dimless_dep = load_data(pathdir, filename)[1]
        if dimless_data.ndim==1:
            dimless_data = np.reshape(dimless_data,(1,len(dimless_data)))
            dimless_data = dimless_data.T
        np.savetxt(pathdir + filename + "_dim_red", dimless_data)
        file_sym.write(", ")
        for j in range(len(input)):
            file_sym.write(str(input[j]))
            file_sym.write(", ")
        file_sym.write("\n")
    else:
        print("we have solved something")
        # get the symbolic form of the solved part
        solved_powers = dimensional_analysis(input,output,units)[0]
        print("after solvedpowers")
        input_sym = symbols(input)
        sol = symbols("sol")
        sol = 1
        print("Reached 4th for loop")
        for i in range(len(input_sym)):
            sol = sol*input_sym[i]**np.round(solved_powers[i],2)
        print("Finished 4th for loop")
        file_sym.write(str(sol))
        file_sym.write(", ")
        print("written to file_sym")
        # get the symbolic form of the unsolved part
        unsolved_powers = dimensional_analysis(input,output,units)[1]
        print("Called dim_analysis internal function")
        uns = symbols("uns")
        unsolved = []
        for i in range(len(unsolved_powers)):
            uns = 1
            for j in range(len(unsolved_powers[i])):
                uns = uns*input_sym[j]**unsolved_powers[i][j]
            file_sym.write(str(uns))
            file_sym.write(", ")
            unsolved = unsolved + [uns]
        file_sym.write("\n")

        # get the discovered part of the function
        func = 1
        for j in range(len(input)):
            func = func * vars()[input[j]]**dimensional_analysis(input,output,units)[0][j]
        func = np.array(func)

        # get the new variables needed
        new_vars = []
        for i in range(len(dimensional_analysis(input,output,units)[1])):
            nv = 1
            for j in range(len(input)):
                nv = nv*vars()[input[j]]**dimensional_analysis(input,output,units)[1][i][j]
            new_vars = new_vars + [nv]

        new_vars = np.array(new_vars)
        new_dependent = deps/func

        if new_vars.size==0:
            print("No new vars")
            np.savetxt(pathdir + filename + "_dim_red", new_dependent)

        # save this to file
        all_variables = np.vstack((new_vars, new_dependent)).T
        np.savetxt(pathdir + filename + "_dim_red", all_variables)
    
    file_sym.close()
    print("Finished dim analysis")

