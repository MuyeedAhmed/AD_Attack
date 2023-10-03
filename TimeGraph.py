import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.covariance import EllipticEnvelope
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from time import process_time
from numpy import unravel_index

import warnings
warnings.filterwarnings(action='ignore')

datasetFolderDir = 'Dataset/'

def ReadFile(filename):
    print(filename)
    folderpath = datasetFolderDir
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        # if os.path.getsize(folderpath+filename+".mat") > 200000: # 200KB
        #     return
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
        # if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
        #     print("Didn\'t run -> Too large - ", filename)    
        #     return
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
    return X, gt
    
    
    
    
def runEE(filename, X, gt, runs):
    times = []
    flips = []
    print("Rows: ", len(X))
    sfs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for sf in sfs:    
        labels = []
        timeElapsed = []
        for i in range(runs):
            #time
            t1_start = process_time()
            
            clustering = EllipticEnvelope(support_fraction=sf, contamination=0.1).fit(X)

            t1_stop = process_time()
            timeElapsed.append(t1_stop-t1_start)
    
            l = clustering.predict(X)
            l = [0 if x == 1 else 1 for x in l]
            labels.append(l)
        avgTimeElapsed = sum(timeElapsed)/len(timeElapsed)
        
        flipped = flip_count(filename, gt, labels, runs)
        times.append(avgTimeElapsed)
        flips.append(flipped)
    print(times)
    print(flips)
    draw(filename, sfs,flips,times)

def draw(filename, sfs, f, t):
    fig,ax = plt.subplots()
    # ax.grid(False)

    # make a plot
    ax.plot(sfs, f,
            color="red", 
            marker="o")
    # set x-axis label
    ax.set_xlabel("support_fraction", fontsize = 12)
    # ax.set_xlabel("NumLearners", fontsize = 12)
    # set y-axis label
    # ax.set_ylabel("ARI",
    ax.set_ylabel("Flipped Points",
                  color="red",
                  fontsize=12)
    ax2=ax.twinx()
    ax2.grid(False)

    # make a plot with different y-axis using second axis object
    ax2.plot(sfs, t,color="blue",marker="o")
    ax2.set_ylabel("Time (seconds)",color="blue",fontsize=12)
    plt.show()
    
    fig.savefig('Fig/Time/SkEE_'+filename+'.pdf', bbox_inches='tight')
    
def flip_count(filename, gt, labels, runs):
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
    
    return flipped
    
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
        
    runs = 10
    for fname in master_files:
        if os.path.exists("Fig/Time/SkEE_"+fname+".pdf"):
            print(fname, " already done!")
            continue
        X, gt = ReadFile(fname)
        runEE(fname, X, gt, runs)