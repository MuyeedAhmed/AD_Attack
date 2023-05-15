#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 08:45:19 2023

@author: muyeedahmed
"""

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
import time
import seaborn as sns
import matplotlib.pyplot as plt

datasetFolderDir = 'Dataset/'

import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from time import process_time
from numpy import unravel_index
datasetFolderDir = 'Dataset/'


def isolationforest(filename):
    # print(filename)
    # folderpath = datasetFolderDir
    
    # if os.path.exists(folderpath+filename+".mat") == 1:
    #     if os.path.getsize(folderpath+filename+".mat") > 200000: # 200KB
    #         # print("Didn\'t run -> Too large - ", filename)    
    #         return
    #     try:
    #         df = loadmat(folderpath+filename+".mat")
    #     except NotImplementedError:
    #         df = mat73.loadmat(folderpath+filename+".mat")

    #     gt=df["y"]
    #     gt = gt.reshape((len(gt)))
    #     X=df['X']
    #     if np.isnan(X).any():
    #         # print("Didn\'t run -> NaN - ", filename)
    #         return
        
    # elif os.path.exists(folderpath+filename+".csv") == 1:
    #     # if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
    #     #     print("Didn\'t run -> Too large - ", filename)    
    #     #     return
    #     X = pd.read_csv(folderpath+filename+".csv")
    #     target=X["target"].to_numpy()
    #     print(X)
    #     X=X.drop("target", axis=1)
    #     gt = target
    #     if X.isna().any().any() == 1:
    #         print("Didn\'t run -> NaN value - ", filename)  
    #         return
    # else:
    #     print("File not found")
    #     return
    n_e = [50,100,150,200,250,300,350,400,450,500]
    
    # ari_all = []
    # t_all = []
    # for n in n_e:
    #     labels = []
    #     ari = []
    #     t = []
    #     for i in range(30):
    #         start = time.process_time()
    #         clustering = IsolationForest(n_estimators=n).fit(X)
    #         t.append(time.process_time() - start)
    #         l = clustering.predict(X)
    #         l = [0 if x == 1 else 1 for x in l]
    #         labels.append(l)


    
    #     for i in range(len(labels)):
    #         for j in range(i+1, len(labels)):
    #           ari.append(adjusted_rand_score(labels[i], labels[j]))
    #     ari_all.append(np.mean(ari))
    #     t_all.append(np.mean(t))
    # print(ari_all)
    # print(t_all)
    
    
    #matlab
    f = [11.302857142857142, 6.252244897959184, 5.298775510204082, 5.260408163265306, 4.204081632653061, 3.7624489795918366, 3.1208163265306124 , 3.097142857142857, 2.7689795918367346, 2.7983673469387753]
    f_p = []
    
    t_all = [0.04969595999999939, 0.09974557999999917, 0.15023724000000072, 0.1970731800000007, 0.24362298000000124, 0.29080861999999913, 0.3405281199999996, 0.3881066999999985, 0.4373698400000006, 0.49845322000000236]
    
    fig,ax = plt.subplots(figsize=(10,4))
    # ax.grid(False)

    # make a plot
    ax.plot(n_e, f,
            color="red", 
            marker="o")
    # set x-axis label
    ax.set_xlabel("n_estimators", fontsize = 15)
    # ax.set_xlabel("NumLearners", fontsize = 12)
    # set y-axis label
    # ax.set_ylabel("ARI",
    ax.set_ylabel("Flipped Points",
                  color="red",
                  fontsize=15)
    ax2=ax.twinx()
    ax2.grid(False)

    # make a plot with different y-axis using second axis object
    ax2.plot(n_e, t_all,color="blue",marker="o")
    ax2.set_ylabel("Time (seconds)",color="blue",fontsize=15)
    plt.show()
    
    fig.savefig('Fig/n_e_graph.pdf', bbox_inches='tight')

    # plt.plot(n_e, ari_all)
    # plt.plot(n_e, t_all)
    


def isolationforest2(filename):
    print(filename)
    folderpath = datasetFolderDir
    
    
    if os.path.exists(folderpath+filename+".mat") == 1:
       
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
    
    runIF(filename, X, gt, [50, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [100, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [150, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [200, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [250, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [300, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [350, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [400, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [450, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')
    runIF(filename, X, gt, [500, 'auto', 'auto', 1.0, False, None, False], runs, 'Default')

    
    
    
    
def runIF(filename, X, gt, params, runs, mode):
    
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
    
    flipped, avgFlippedPerRun, avgFlippedPerRunPercentage = drawGraphs(filename, gt, labels, runs, mode)
    
    print(params[0], avgFlippedPerRun, avgFlippedPerRunPercentage, avgTimeElapsed)
    
def drawGraphs(filename, gt, labels, runs, mode):
    '''Flip Summary'''
    norms = 0
    outliers = 0
    flipped = 0
    avgs = []
    flippable = [False]*len(labels[0])
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
            flippable[i] = True
    # print("*****")
    # print(norms, outliers, flipped)
    if flipped == 0:
        return flipped, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    inlier = len(gt) - sum(gt)
    outlier = sum(gt)
    ti_fo = 0
    to_fi = 0
    for i in range(len(gt)):
        if flippable[i] == True:
            if gt[i] == 0:
                ti_fo += 1
            else:
                to_fi += 1
    ti_fo_per_all = ti_fo/inlier
    to_fi_per_all = to_fi/outlier
    
    
    '''
    Flipped in a single run (mean)
    '''
    ti_fo_avg = []
    to_fi_avg = []
    flippedIn2Runs = []
    for i in range(runs):
        for j in range(i+1,runs):
            ti_fo = 0
            to_fi = 0
            norms = 0
            outliers = 0
            variables = 0
            for n in range(len(gt)):
                
                if labels[i][n] != labels[j][n]:
                    variables += 1
                    if gt[n] == 0:
                       ti_fo += 1
                    else:
                       to_fi += 1
            ti_fo_avg.append(ti_fo)
            to_fi_avg.append(to_fi)
            flippedIn2Runs.append(variables)
    
    
    ti_fo_per_avg = np.mean(ti_fo_avg)/inlier
    to_fi_per_avg = np.mean(to_fi_avg)/outlier
    avgFlippedPerRun = sum(flippedIn2Runs)/len(flippedIn2Runs)

    
    return flipped, avgFlippedPerRun, avgFlippedPerRun/len(labels[0])
if __name__ == '__main__':

    isolationforest('spambase')
    
    