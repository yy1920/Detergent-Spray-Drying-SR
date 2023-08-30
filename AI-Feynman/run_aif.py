#import aifeynman
from aifeynman.S_run_aifeynman import run_aifeynman
import time
import argparse

start = time.time()
#aifeynman.get_demos("example_data") # Download examples from server
#aifeynman.run_aifeynman("","fma.txt", 30, "14ops.txt", polyfit_deg=4, NN_epochs=70, vars_name=["m","m","L","a","F"])
#run_aifeynman("","fma.txt", 60, "14ops.txt", polyfit_deg=4, NN_epochs=300, vars_name=["m","m","L","a","F"])
#run_aifeynman("example_data/","example2.txt", 30, "14ops.txt", polyfit_deg=3, NN_epochs=400)#, vars_name=["m","m","m","m","m"])


parser = argparse.ArgumentParser()
parser.add_argument("-f","--filename", help="Name of file with data to be passed to symbolic regressor", type=str)
parser.add_argument("-d","--directory", help="Location of file with data to be passed to symbolic regressor", type=str)

args = parser.parse_args()
FILE_LOC = args.directory
FILE_NAME = args.filename

try:
    run_aifeynman(FILE_LOC+"/", FILE_NAME, 30, "14ops.txt", polyfit_deg=4, NN_epochs=400)
except:
    print("Could not access data")
    exit()

end = time.time()
print("Execution Time = " + str(end-start))
