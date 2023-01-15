import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn import metrics
from copy import copy, deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
# import pingouin as pg
import random
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
# import scikit_posthocs as sp
import scipy.stats as stats
from scipy.stats import gmean
import math
import scipy.stats as ss
import bisect 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

from pylab import *
from scipy.optimize import curve_fit
from time import process_time

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
    
    runs = 50
    
    
    '''
    Default
    '''
    parameters_default = [100, 'auto', 'auto', 1.0, False, None, False]
    runIF(filename, X, parameters_default, runs, 'Default')

    '''
    Optimal
    '''
    runIF(filename, X, optSettings, runs, "Optimal")
    '''
    Fast
    '''
    parameters_fast = [50, 64, 'auto', 1.0, False, None, False]
    runIF(filename, X, parameters_fast, runs, 'Fast')
    
    
    
def runIF(filename, X, params, runs, mode):
    
    labels = []
    timeElapsed = []
    for i in range(runs):
        #time
        t1_start = process_time() 
        clustering = IsolationForest(n_estimators=params[0], max_samples=params[1], 
                                      max_features=params[3], bootstrap=params[4], 
                                      n_jobs=params[5], warm_start=params[6]).fit(X)
        t1_stop = process_time()
        timeElapsed.append(t1_stop-t1_start)

        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
    avgTimeElapsed = sum(timeElapsed)/len(timeElapsed)
    
    flipped, runNumber50p, avgFlippedPerRun, avgFlippedPerRunPercentage = drawGraphs(filename, labels, runs, mode)
    
    f=open("Stats/SkIF.csv", "a")
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

    f = plt.figure()
    avgs = np.array(avgs)
    sns.displot(avgs, kde=True, stat='count')
    plt.savefig("FlipFig/"+filename+"_"+mode+"_NormDist.pdf", bbox_inches="tight", pad_inches=0)
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
    plot(variables_iter)
    plt.savefig("FlipFig/"+filename+"_"+mode+"_Count.pdf", bbox_inches="tight", pad_inches=0)
    plt.axhline(y=0.5*flipped, color='r', linestyle='-')
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
    if os.path.exists("Stats/SkIF.csv"):
        df = pd.read_csv("Stats/SkIF.csv")
        done_files = df["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
    
    optimalSettingsUni = pd.read_csv("OptimalSettings/SkIF_Uni.csv")
    
    if os.path.exists("Stats/SkIF.csv")==0:
        f=open("Stats/SkIF.csv", "w")
        f.write('Filename,Mode,AvgTimeElapsed,Flipped,RunNumber50p,AvgFlippedPerRun,AvgFlippedPerRunPercentage\n')
        f.close()
    
    for fname in master_files:
        optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == fname].to_numpy()[0][1:]
        try:
            optSettings[1] = float(optSettings[1])
        except:
            pass
        optSettings[5] = None
        isolationforest(fname, optSettings)
        
    # isolationforest('ar1')
    # isolationforest('breastw')
    # isolationforest("arsenic-female-lung")
    