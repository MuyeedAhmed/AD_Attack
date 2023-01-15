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

def isolationforest(filename, optSettings):
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
    
    runs = 10
    
    
    '''
    Default
    '''
    parameters_default = [0.1, 100, 'auto']
    runIF(filename, X, parameters_default, runs, 'Default')

    '''
    Optimal
    '''
    runIF(filename, X, optSettings, runs, "Optimal")
    '''
    Fast
    '''
    parameters_fast = [optSettings[0], 50, 64]
    runIF(filename, X, parameters_fast, runs, 'Fast')
    
    
    
def runIF(filename, X, params, runs, mode):
    labelFile = filename + "_" + str(params[0]) + "_" + str(params[1]) + "_" + str(params[2])
    print(params)
    frr=open("GD_ReRun/MatIF.csv", "a")
    frr.write(filename+","+str(params[0])+","+str(params[1])+","+str(params[2])+'\n')
    frr.close()
    # try:
    t1_start = process_time()
    
    eng.MatIF_Rerun(nargout=0)
    t1_stop = process_time()
    avgTimeElapsed = (t1_stop-t1_start)/runs
    
    frr=open("GD_ReRun/MatIF.csv", "w")
    frr.write('Filename,ContaminationFraction,NumLearners,NumObservationsPerLearner\n')
    frr.close()
    if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv") == 0:      
        print("\nFaild to run Matlab Engine from Python.\n")
        exit(0)
    # except:
    #     print("\nFaild to run Matlab Engine from Python.\n")
    #     exit(0)    
    labels =  pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv", header=None).to_numpy()
    
    flipped, runNumber50p, avgFlippedPerRun, avgFlippedPerRunPercentage = drawGraphs(filename, labels, runs, mode)
    
    f=open("Stats/MatIF.csv", "a")
    f.write(filename+','+mode+','+str(avgTimeElapsed)+','+str(flipped)+','+str(runNumber50p)+','+str(avgFlippedPerRun)+','+str(avgFlippedPerRunPercentage)+'\n')
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
    plt.savefig("FlipFig/"+filename+"_MatIF_"+mode+"_NormDist.pdf", bbox_inches="tight", pad_inches=0)
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
    plt.savefig("FlipFig/"+filename+"_MatIF_"+mode+"_Count.pdf", bbox_inches="tight", pad_inches=0)
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
    if os.path.exists("Stats/MatIF.csv"):
        df = pd.read_csv("Stats/MatIF.csv")
        done_files = df["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
    
    optimalSettingsUni = pd.read_csv("OptimalSettings/MatIF_Uni.csv")
    
    if os.path.exists("Stats/MatIF.csv")==0:
        f=open("Stats/MatIF.csv", "w")
        f.write('Filename,Mode,AvgTimeElapsed,Flipped,RunNumber50p,AvgFlippedPerRun,AvgFlippedPerRunPercentage\n')
        f.close()
    
    for fname in master_files:
        optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == fname].to_numpy()[0][1:]
        isolationforest(fname, optSettings)


    # optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == 'breastw'].to_numpy()[0][1:]
        
    # isolationforest('breastw', optSettings)
    # isolationforest('breastw')
    # isolationforest("arsenic-female-lung")
    