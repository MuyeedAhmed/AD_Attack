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
    aris = []
    m_aris = []
    print("Rows: ", len(X))
    sfs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # sfs = [0.1]
    for sf in sfs:
        print(sf, end=",")
        labels = []
        timeElapsed = []
        ari = []
        m_ari =[]
        for i in range(runs):
            #time
            t1_start = process_time()
            
            clustering = EllipticEnvelope(support_fraction=sf, contamination=0.1).fit(X)

            t1_stop = process_time()
            timeElapsed.append(t1_stop-t1_start)
    
            l = clustering.predict(X)
            l = [0 if x == 1 else 1 for x in l]
            ari.append(adjusted_rand_score(gt, l))
            labels.append(l)
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):            
                m_ari.append(adjusted_rand_score(labels[i], labels[j]))
        
        avgTimeElapsed = sum(timeElapsed)/len(timeElapsed)
        aris.append(np.mean(ari))
        m_aris.append(np.mean(m_ari))
        flipped = flip_count(filename, gt, labels, runs)
        times.append(avgTimeElapsed)
        flips.append(flipped)
    print()
    print(times)
    print(flips)
    print(aris)
    print(m_aris)
    draw(filename, sfs,flips,times, aris, m_aris)

def draw(filename, sfs, f, t, a, ma):
    f = f/(np.max(f))
    a = [(x-np.min(a))/(np.max(a) - np.min(a)) for x in a]
    ma = [(x-np.min(ma))/(np.max(ma) - np.min(ma)) for x in ma]
        
    fig,ax = plt.subplots()
    
    plt1 = ax.plot(sfs, f, color="red", marker="o", label="Flips")
    plt2 = ax.plot(sfs, a, color="orange", marker='o', label="ARI")
    plt3 = ax.plot(sfs, ma, color="green", marker='o', label="Mutual ARI")
    
    ax.set_xlabel("support_fraction", fontsize = 12)
    ax.set_ylabel("Vulnerability", color="red", fontsize=12)
    
    
    
    ax2=ax.twinx()
    ax2.grid(False)

    # make a plot with different y-axis using second axis object
    plt3 = ax2.plot(sfs, t,color="blue",marker="o")
    ax2.set_ylabel("Time (seconds)",color="blue",fontsize=12)
    ax.legend()
    plt.show()
    
    # fig.savefig('Fig/Time/SkEE_'+filename+'.pdf', bbox_inches='tight')

sfs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
tt = [0.7555669999999985, 0.7542099, 0.7787064000000001, 0.8093423999999985, 0.7836930000000024, 0.7779766999999964, 0.7921503000000001, 0.8128416999999999, 0.8052817000000004]
ff = [33, 4, 5, 0, 6, 4, 6, 3, 5]
aa = [0.032293785658586406, 0.029256153818515134, 0.02382069494719217, 0.032864422379966014, 0.034553276514958195, 0.039268234475448326, 0.03892815804553522, 0.0362075466062304, 0.026673878554090223]
mma = [0.904130834504301, 0.9902547576701362, 0.9471428413980081, 1.0, 0.9692982558045985, 0.9918002945853023, 0.9781667423646448, 0.9906289080974885, 0.9855528999836279]

draw("pima", sfs, ff, tt, aa, mma)    

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
        # if os.path.exists("Fig/Time/SkEE_"+fname+".pdf"):
        #     # print(fname, " already done!")
        #     continue
        if fname != 'KnuggetChase3':
            continue
        X, gt = ReadFile(fname)
        runEE(fname, X, gt, runs)
        
        
        