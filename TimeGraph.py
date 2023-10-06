import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from sklearn.metrics.cluster import adjusted_rand_score

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from time import process_time
from numpy import unravel_index

import scipy.stats as stats


import warnings
warnings.filterwarnings(action='ignore')

datasetFolderDir = 'Dataset/'

implementation = "SkEE"
parameter_st = "support_fraction"

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
            # print("Didn\'t run -> nan - ", filename)
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
            print("Didn\'t run -> nan value - ", filename)  
            return
    else:
        print("File doesn't exist")
        return
    return X, gt
    
    

def runIF(filename, X, gt, runs):
    times = []
    flips = []
    aris = []
    # params = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for param in params:
    for param in params:
        print(param, end=",")
        labels = []
        timeElapsed = []
        ari = []
        m_ari =[]
        for i in range(runs):
            t1_start = process_time()
            
            # clustering = IsolationForest(n_estimators=param).fit(X)
            clustering = IsolationForest(max_samples=param).fit(X)

            t1_stop = process_time()
            timeElapsed.append(t1_stop-t1_start)
    
            l = clustering.predict(X)
            l = [0 if x == 1 else 1 for x in l]
            ari.append(adjusted_rand_score(gt, l))
            labels.append(l)
        
        avgTimeElapsed = sum(timeElapsed)/len(timeElapsed)
        aris.append(np.mean(ari))
        flipped = flip_count(filename, gt, labels, runs)
        times.append(avgTimeElapsed)
        flips.append(flipped)
    print()

    draw(filename,params,flips,times, aris)
    
    t = times/(np.max(times))
    f = flips/(np.max(flips))
    a = [(x-np.min(aris))/(np.max(aris) - np.min(aris)) for x in aris]
    
    time_slope, _, _, _, _ = stats.linregress(params, t)
    flip_slope, _, _, _, _ = stats.linregress(params, f)
    ari_slope, _, _, _, _ = stats.linregress(params, a)
    
    return time_slope, flip_slope, ari_slope
    
    
def runEE(filename, X, gt, runs):
    times = []
    flips = []
    aris = []
    # m_aris = []
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
        
        avgTimeElapsed = sum(timeElapsed)/len(timeElapsed)
        aris.append(np.mean(ari))
        flipped = flip_count(filename, gt, labels, runs)
        times.append(avgTimeElapsed)
        flips.append(flipped)
    print()
    # print(times)
    # print(flips)
    # print(aris)

    draw(filename, sfs,flips,times, aris)
    t = times/(np.max(times))
    f = flips/(np.max(flips))
    a = [(x-np.min(aris))/(np.max(aris) - np.min(aris)) for x in aris]
    
    # time_alpha, time_beta = alpha_beta(t)
    # flip_alpha, flip_beta = alpha_beta(f)
    # ari_alpha, ari_beta = alpha_beta(a)
    
    time_slope, _, _, _, _ = stats.linregress(sfs, t)
    flip_slope, _, _, _, _ = stats.linregress(sfs, f)
    ari_slope, _, _, _, _ = stats.linregress(sfs, a)
    
    return time_slope, flip_slope, ari_slope
    # return time_beta, flip_beta, ari_beta

# def alpha_beta(data):
#     μ = np.mean(data)
#     σ2 = np.var(data, ddof=1)

#     α = μ * ((μ * (1 - μ) / σ2) - 1)
#     β = (1 - μ) * ((μ * (1 - μ) / σ2) - 1)
#     return α, β

    
def draw(filename, sfs, f, t, a):
    # f = f/(np.max(f))
    # a = [(x-np.min(a))/(np.max(a) - np.min(a)) for x in a]
    # ma = [(x-np.min(ma))/(np.max(ma) - np.min(ma)) for x in ma]
        
    fig,ax = plt.subplots()
    
    plt1 = ax.plot(sfs, f, color="red", marker="o", label="Flips")
    # plt2 = ax.plot(sfs, a, color="orange", marker='o', label="ARI")
    # plt3 = ax.plot(sfs, ma, color="green", marker='o', label="Mutual ARI")
    
    ax.set_xlabel(parameter_st, fontsize = 12)
    ax.set_ylabel("Flips", color="red", fontsize=12)
    
    ax2=ax.twinx()
    ax2.grid(False)

    plt3 = ax2.plot(sfs, t,color="blue",marker="o")
    ax2.set_ylabel("Time (seconds)",color="blue",fontsize=12)
    ax.legend()
    plt.show()
    
    fig.savefig('Fig/Time/'+implementation+'_'+parameter_st+'_'+filename+'.pdf', bbox_inches='tight')

# sfs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# tt = [0.7555669999999985, 0.7542099, 0.7787064000000001, 0.8093423999999985, 0.7836930000000024, 0.7779766999999964, 0.7921503000000001, 0.8128416999999999, 0.8052817000000004]
# ff = [33, 4, 5, 0, 6, 4, 6, 3, 5]
# aa = [0.032293785658586406, 0.029256153818515134, 0.02382069494719217, 0.032864422379966014, 0.034553276514958195, 0.039268234475448326, 0.03892815804553522, 0.0362075466062304, 0.026673878554090223]
# mma = [0.904130834504301, 0.9902547576701362, 0.9471428413980081, 1.0, 0.9692982558045985, 0.9918002945853023, 0.9781667423646448, 0.9906289080974885, 0.9855528999836279]

# draw("pima", sfs, ff, tt, aa, mma)    

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


def drawBeanPlot(bt, bf, ba):
    data_combined = np.concatenate([bt, bf, ba])
    labels = ['Time'] * len(bt) + ['Flips'] * len(bf) + ['ARI'] * len(ba)
    df = pd.DataFrame({'Group': labels, 'Value': data_combined})
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='Group', y='Value')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=15)
    plt.savefig("Fig/Slope_"+implementation+"_"+parameter_st+".pdf", bbox_inches='tight')

    plt.show()

def save_slope(filename, t_b, f_b, a_b):
    if os.path.exists("Stats/Slope_"+implementation+"_"+parameter_st+".csv") == 0:
        f=open("Stats/Slope_"+implementation+"_"+parameter_st+".csv", "w")
        f.write('Filename,Time_slope,Flip_slope,ARI_Slope\n')
        f.close()

    f=open("Stats/Slope_"+implementation+"_"+parameter_st+".csv", "a")
    f.write(filename+','+str(t_b)+','+str(f_b)+','+str(a_b)+'\n')
    f.close()
    

if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    
    slope_ts = []
    slope_fs = []
    slope_as = []
    
    runs = 50
    
    for fname in master_files:
        # if os.path.exists("Fig/Time/SkEE_"+fname+".pdf"):
        #     # print(fname, " already done!")
        #     continue
        X, gt = ReadFile(fname)
        t_b, f_b, a_b = runEE(fname, X, gt, runs)
        # t_b, f_b, a_b = runIF(fname, X, gt, runs)
    #     print("Slopes: ", t_b, f_b, a_b)
    #     save_slope(fname, t_b, f_b, a_b)
    #     slope_ts.append(t_b)
    #     slope_fs.append(f_b)
    #     slope_as.append(a_b)
    
    
    # df = pd.read_csv("Stats/Slope_"+implementation+"_"+parameter_st+".csv")
    # slope_ts = df["Time_slope"].to_numpy()
    # slope_fs = df["Flip_slope"].to_numpy()
    # slope_as = df["ARI_Slope"].to_numpy()
    
    # drawBeanPlot(slope_ts, slope_fs, slope_as)
    
    
    