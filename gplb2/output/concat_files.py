import os
import glob
if __name__ == "__main__":
    #i = 0
    all_results = open("all_results.txt", "w")
    #for filename in os.listdir(os.getcwd()):
    for filename in glob.glob("*_train*.out"):
        #if i == 3:
         #   break
        all_results.write(filename)
        all_results.write("\n")
        try:
            with open(os.path.join(os.getcwd(), filename), 'r') as f:
                all_results.write(f.read())
                f.close()
        except:
            print("File error")
        all_results.write("\n")
        all_results.write("?$_!")
        all_results.write("\n")
        #i+=1
