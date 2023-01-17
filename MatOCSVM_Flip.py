import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from time import process_time
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'

def ocsvm(filename, optSettings):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        if os.path.getsize(folderpath+filename+".mat") > 200000: # 200KB
            return
        try:
            df = loadmat(folderpath+filename+".mat")
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename+".mat")

        gt=df["y"]
        gt = gt.reshape((len(gt)))
        X=df['X']
        if np.isnan(X).any():
            # print("Didn\'t run -> NaN - ", filename)
            return
        
    elif os.path.exists(folderpath+filename+".csv") == 1:
        if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
            print("Didn\'t run -> Too large - ", filename)    
            return
        X = pd.read_csv(folderpath+filename+".csv")
        target=X["target"].to_numpy()
        X=X.drop("target", axis=1)
        gt = target
        if X.isna().any().any() == 1:
            print("Didn\'t run -> NaN value - ", filename)  
            return
    else:
        print("File doesn't exist")
        return
    
    runs = 50
    
    
    # parameters = []

    # ContaminationFraction = [0.05, 0.1, 0.15, 0.2, 0.25];
    # KernelScale = [1, "auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    # Lambda = ["auto", 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5];
    # NumExpansionDimensions = ["auto", 2^12, 2^15, 2^17, 2^19];
    # StandardizeData = [0, 1];
    # BetaTolerance = [1e-2, 1e-3, 1e-4, 1e-5];
    # GradientTolerance = [1e-3, 1e-4, 1e-5, 1e-6];
    # IterationLimit = [100, 200, 500, 1000, 2000];
    
    # parameters.append(["ContaminationFraction", 0.1, ContaminationFraction])
    # parameters.append(["KernelScale", 1, KernelScale])
    # parameters.append(["Lambda", 'auto', Lambda])
    # parameters.append(["NumExpansionDimensions", 'auto', NumExpansionDimensions])
    # parameters.append(["StandardizeData", 0, StandardizeData])
    # parameters.append(["BetaTolerance", 1e-4, BetaTolerance])
    # parameters.append(["GradientTolerance", 1e-4, GradientTolerance])
    # parameters.append(["IterationLimit", 1000, IterationLimit])
    
    parameters_default = [0.1, 1, "auto", "auto", 0, 1e-4, 1e-4, 1000]
    parameters_fast = optSettings
    parameters_fast[3] = 2^8
    ReRun_CSV(filename, parameters_default)
    ReRun_CSV(filename, optSettings)
    ReRun_CSV(filename, parameters_fast)
    
    time = eng.MatOCSVM_Rerun(runs)
    print(time)
    '''
    Default
    '''
    runOCSVM(filename, X, parameters_default, runs, 'Default', time[0][0])

    '''
    Optimal
    '''
    runOCSVM(filename, X, optSettings, runs, "Optimal", time[0][1])
    '''
    Fast
    '''
    runOCSVM(filename, X, parameters_fast, runs, 'Fast', time[0][2])
    

def ReRun_CSV(filename, params):
    frr=open("GD_ReRun/MatOCSVM.csv", "a")
    frr.write(filename+","+str(params[0])+","+str(params[1])+","+str(params[2])+","+str(params[3])+","+str(params[4])+","+str(params[5])+","+str(params[6])+","+str(params[7])+'\n')
    frr.close()

  
def runOCSVM(filename, X, params, runs, mode, t):
    labelFile = filename + "_" + str(params[0]) + "_" + str(params[1]) + "_" + str(params[2]) + "_" + str(params[3]) + "_" + str(params[4]) + "_" + str(params[5]) + "_" + str(params[6]) + "_" + str(params[7])
    
    if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv"):
        labels =  pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv", header=None).to_numpy()
    else:
        return
    flipped, runNumber50p, avgFlippedPerRun, avgFlippedPerRunPercentage = drawGraphs(filename, labels, runs, mode)
    
    f=open("Stats/MatOCSVM.csv", "a")
    f.write(filename+','+mode+','+str(t)+','+str(flipped)+','+str(runNumber50p)+','+str(avgFlippedPerRun)+','+str(avgFlippedPerRunPercentage)+'\n')
    f.close()
    
def drawGraphs(filename, labels, runs, mode):
    norms = 0
    outliers = 0
    flipped = 0
    avgs = []
    
    for i in range(len(labels[0])):
        s = 0
        for j in range(runs):
            s+=labels[j][i]
        avg = s/runs
        avgs.append(avg)
        if avg == 0:
            norms += 1
        elif avg == 1:
            outliers += 1
        else:
            flipped += 1
    # print("*****")
    # print(norms, outliers, flipped)
    if flipped == 0:
        return flipped, -1, 0, 0
    
    f = plt.figure()
    avgs = np.array(avgs)
    sns.displot(avgs, kde=True, stat='count')
    plt.savefig("FlipFig/"+filename+"_MatOCSVM_"+mode+"_NormDist.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()

    
    variables_iter = []
    for h in range(1,runs):
        norms = 0
        outliers = 0
        variables = 0
        avg = 0
        for i in range(len(labels[0])):
            s = 0
            for j in range(h):
                s+=labels[j][i]
            avg = s/h
            # avgs.append(avg)
            if avg == 0:
                norms += 1
            elif avg == 1:
                outliers += 1
            else:
                variables += 1
        variables_iter.append(variables)
    
    
    probability = [x / flipped for x in variables_iter]
    for i in range(runs-1):
        if probability[i] < 0.5:
            continue
        else:
            runNumber50p = i
            # print(mode, "50% - ", runNumber50p)
            break

    g = plt.figure
    plt.plot(variables_iter)
    plt.axhline(y=0.5*flipped, color='r', linestyle='-')
    plt.savefig("FlipFig/"+filename+"_MatOCSVM_"+mode+"_Count.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


    '''
    Flipped in a single run
    '''
    flippedIn2Runs = []
    for i in range(runs):
        for j in range(i+1,runs):
            
            norms = 0
            outliers = 0
            variables = 0
            for n in range(len(labels[0])):
                s = labels[i][n] + labels[j][n]                
                avg = s/2
                if avg == 0:
                    norms += 1
                elif avg == 1:
                    outliers += 1
                else:
                    variables += 1
            flippedIn2Runs.append(variables)
            # print(variables, end=' , ')
        
    avgFlippedPerRun = sum(flippedIn2Runs)/len(flippedIn2Runs)
    
    # print(mode, "- flipped in a single run ", avgFlippedPerRun, " - ", avgFlippedPerRun/len(labels[0]))
    return flipped, runNumber50p, avgFlippedPerRun, avgFlippedPerRun/len(labels[0])

    
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    if os.path.exists("Stats/MatOCSVM.csv"):
        df = pd.read_csv("Stats/MatOCSVM.csv")
        done_files = df["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
    
    optimalSettingsUni = pd.read_csv("OptimalSettings/MatOCSVM_Uni.csv")
    
    if os.path.exists("Stats/MatOCSVM.csv")==0:
        f=open("Stats/MatOCSVM.csv", "w")
        f.write('Filename,Mode,AvgTimeElapsed,Flipped,RunNumber50p,AvgFlippedPerRun,AvgFlippedPerRunPercentage\n')
        f.close()
    
    for fname in master_files:
        frr=open("GD_ReRun/MatOCSVM.csv", "w")
        frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
        frr.close()
        optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == fname].to_numpy()[0][1:]
        ocsvm(fname, optSettings)


    # optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == 'breastw'].to_numpy()[0][1:]
        
    # ocsvm('breastw', optSettings)
    # # isolationforest("arsenic-female-lung")
    