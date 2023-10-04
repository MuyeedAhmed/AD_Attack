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

implementation = "SkIF"
parameter_st = "n_estimators"

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
    n_es = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    for n_e in n_es:
        print(n_e, end=",")
        labels = []
        timeElapsed = []
        ari = []
        m_ari =[]
        for i in range(runs):
            t1_start = process_time()
            
            clustering = IsolationForest(n_estimators=n_e).fit(X)

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

    draw(filename, n_es,flips,times, aris)
    
    t = times/(np.max(times))
    f = flips/(np.max(flips))
    a = [(x-np.min(aris))/(np.max(aris) - np.min(aris)) for x in aris]
    
    time_slope, _, _, _, _ = stats.linregress(n_es, t)
    flip_slope, _, _, _, _ = stats.linregress(n_es, f)
    ari_slope, _, _, _, _ = stats.linregress(n_es, a)
    
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
    
    ax.set_xlabel("support_fraction", fontsize = 12)
    ax.set_ylabel("Vulnerability", color="red", fontsize=12)
    
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
    
    runs = 10
    
    for fname in master_files:
        # if os.path.exists("Fig/Time/SkEE_"+fname+".pdf"):
        #     # print(fname, " already done!")
        #     continue
        X, gt = ReadFile(fname)
        # t_b, f_b, a_b = runEE(fname, X, gt, runs)
        t_b, f_b, a_b = runIF(fname, X, gt, runs)
        print("Slopes: ", t_b, f_b, a_b)
        save_slope(fname, t_b, f_b, a_b)
        slope_ts.append(t_b)
        slope_fs.append(f_b)
        slope_as.append(a_b)
    # print("b_t = ", beta_ts)
    # print("b_f = ", beta_fs)
    # print("b_a = ", beta_as)
    # beta_ts =  [2.3394775979612383, 0.5810729176116967, 0.8220599627348946, 0.36671007482570717, 1.2337192105688684, 2.708059155198904, 2.4977108368930483, 0.8198934538534841, 0.8197092654768238, 0.6024426795521665, 1.1898728901032494, 0.9024436560817698, 1.9495998557838694, 3.267789345414634, 1.5863860846838136, 1.445090030896423, 2.0140211371358605, 4.386643414536396, 0.4429737358377145, 4.498071412794145, 2.752393683931087, 1.2418186074498538, 5.231802077130445, 2.3337090093318986, 1.025468806999524, 0.43549653324668036, 1.3198204854360507, 0.9331578270377503, 0.9176208626817576, 1.7082965673510067, 0.37412434141926393, 2.3954046731520786, 1.8900299099891378, 3.277310464932674, 2.97204784716666, 2.317981468750144, 1.9240876339937916, 2.089084594256954, 1.639123473715283, 1.0958583035369143, 1.665697198962195, 5.294807544078271, 5.079741173144854, 2.390777762874524, 2.8321638950609134, 0.7522481605775333, 2.3834559171769834, 2.4247535820397927, 2.0334714823340594, 1.5026255752326598, 2.678390810610784, 2.318033979316061, 0.9616456096639302, 2.227303009838377, 1.8390450326314165]
    # beta_fs =  [0.632811055602595, 0.5027437597858563, 1.373033983901582, 0.3068056145659396, 0.2018284242394247, 0.13233359828767663, 0.7305266795304689, 0.2507232637503445, 0.6067746578695484, 0.9941002949852511, 0.12369315462301433, 0.8668260267735418, 1.1158816016828184, 0.6248606179129569, 0.24921724517829397, 0.755777298760859, 1.4108655019426313, 1.4365493326346777, 1.3417349275716157, 0.48430059953720883, 0.7129173954941178, 1.499707651341638, 1.0168601925520275, -0.0665015960983665, 1.0396618191481777, 0.7939078198215282, 1.238260278627251, 1.0363893551421233, 0.34842708362335745, 0.31317532624722183, 0.11574347984240319, -0.034003968123053036, 0.5344297704403832, 1.0608897105445454, 0.6676629978976147, 1.243208859235073, 0.9486780723483945, 1.926507682837043, 0.5185232613145191, 1.1622532498796345, 0.9883774157094425, 0.6412614009822386, 1.3014338362848479, 1.4212127644020975, np.nan, 0.9019999999999999, -0.09876543209876547, 0.32579035827107294, -0.09876543209876547, 0.2008568946252856, np.nan, 0.9650981790548051, 0.3643133272762904, 1.1856056864945748, 1.0867373698837546]
    # beta_as =  [0.4840786478821617, 0.3069455035932473, 1.0385304797352426, 0.5848138054492341, -0.0026579702564136277, 0.1780133602285725, 0.2460836950976305, 0.293259050875669, 0.1522723085909422, 0.5659230582206488, 0.34216114390518365, 1.0760903254763359, 0.16798116544985608, 0.1964641161315091, np.nan, -0.09876543209876547, 1.4295191412569581, 0.7482243878003476, 0.13301492368736761, 0.7025904275182135, 1.108942928021392, 0.6354061932837098, 0.731993179861633, np.nan, 0.5119566086023036, 0.38847483791305937, 0.8743144185884312, 0.378456540370215, 0.8507031376203663, 0.053099180466042076, 0.19923262368917022, -0.07407407407407404, 0.9062448567937937, 0.6491681279517363, 0.2862575651123984, 0.30375307932325973, 1.1657594028428275, 0.9128229612198151, 0.37156275037610614, 0.5715533454026411, 0.20929035578412464, 0.23613212034105271, 0.880597842813497, 0.6746577184499133, np.nan, 0.6496046348246571, -0.012345679012345652, 0.23435518357628, -0.09876543209876547, 0.06442442514793251, np.nan, 1.24547268246242, 0.48610531987811884, 0.4465658006229036, 0.611508970326768]

    drawBeanPlot(slope_ts, slope_fs, slope_as)
    