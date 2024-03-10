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

def isolationforest(filename, optSettings):
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
    
    runs = 30
    
    
    '''
    Default
    '''
    parameters_default = [100, 'auto', 'auto', 1.0, False, None, False]
    runIF(filename, X, gt, parameters_default, runs, 'Default')

    # '''
    # Optimal
    # '''
    # runIF(filename, X, gt, optSettings, runs, "Optimal")
    # '''
    # Fast
    # '''
    # parameters_fast = [50, 64, 'auto', 1.0, False, None, False]
    # runIF(filename, X, gt, parameters_fast, runs, 'Fast')
    
    
    
def runIF(filename, X, gt, params, runs, mode):
    print(X.shape)
    labels = []
    timeElapsed = []
    for i in range(runs):
        # print(i)
        t1_start = process_time() 
        clustering = IsolationForest(n_estimators=params[0], max_samples=params[1], 
                                      max_features=params[3], bootstrap=params[4], 
                                      n_jobs=params[5], warm_start=params[6],random_state=90+i).fit(X)
        t1_stop = process_time()
        timeElapsed.append(t1_stop-t1_start)
        
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        # print(l)
        labels.append(l)
    avgTimeElapsed = sum(timeElapsed)/len(timeElapsed)
    
    import Graphs
    
    g = Graphs.draw(filename, "Sk", "IF", gt, labels, runs, mode)
    g.getFlipSummary()
    ti_fo_all, to_fi_all, ti_fo_avg, to_fi_avg = g.avgFlippedInSingleRun()
    avgFlippedPerRunPercentage = g.avgFlippedPerRun/len(labels[0])
    g.labelSize = 18
    g.tickSize = 18
    print("ti_fo_all, to_fi_all, ti_fo_avg, to_fi_avg")
    print(ti_fo_all, to_fi_all, ti_fo_avg, to_fi_avg)
    # g.avgFlippedInSingleRun()
    # g.flippedInSingleRunSorted()
    # g.flippedInSingleRunUnsorted()
    # g.flippedInSingleRunUnsortedBarchart(True)
    
    _, typ = g.get50thPercentile()
    asc = g.flippedInSingleRunSortedASC()
    dsc = g.flippedInSingleRunSortedDSC()
    
    g.MergeEffectsOfRunOrder(typ,dsc,asc, True)
    
    # f=open("Stats/SkIF.csv", "a")
    # f.write(filename+','+mode+','+str(avgTimeElapsed)+','+str(g.flipped)+','+str(g.flipped/len(gt))+','+str(g.get50thPercentile())+','+str(g.avgFlippedPerRun)+','+str(avgFlippedPerRunPercentage))
    # f.write(","+str(g.ti_fo_per_all)+","+str(g.to_fi_per_all)+","+str(g.ti_fo_per_avg)+","+str(g.to_fi_per_avg)+","+str(ti_fo_all)+","+str(to_fi_all)+","+str(ti_fo_avg)+","+str(to_fi_avg)+'\n')
    # f.close()
   
    
if __name__ == '__main__':
    # folderpath = datasetFolderDir
    # master_files1 = glob.glob(folderpath+"*.mat")
    # master_files2 = glob.glob(folderpath+"*.csv")
    # master_files = master_files1 + master_files2
    # for i in range(len(master_files)):
    #     master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    # if os.path.exists("Stats/SkIF.csv"):
    #     df = pd.read_csv("Stats/SkIF.csv")
    #     done_files = df["Filename"].to_numpy()
    #     master_files = [item for item in master_files if item not in done_files]
    # master_files.sort()

    optimalSettingsUni = pd.read_csv("OptimalSettings/SkIF_Uni.csv")
    
    # if os.path.exists("Stats/SkIF.csv")==0:
    #     f=open("Stats/SkIF.csv", "w")
    #     f.write('Filename,Mode,AvgTimeElapsed,Flipped,FlippedPercentage,RunNumber50p,AvgFlippedPerRun,AvgFlippedPerRunPercentage,ti_fo_per_all,to_fi_per_all,ti_fo_per_avg,to_fi_per_avg,ti_fo_all,to_fi_all,ti_fo_avg,to_fi_avg\n')
    #     f.close()
    
    # for fname in master_files:
    #     try:
    #         optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == fname].to_numpy()[0][1:]
    #     except:
    #         print(fname, "dont exist")
    #     try:
    #         optSettings[1] = float(optSettings[1])
    #     except:
    #         pass
    #     optSettings[5] = None
    #     isolationforest(fname, optSettings)
              
    # optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == 'breastw'].to_numpy()[0][1:]
  
    # isolationforest('breastw', optSettings)
    
    isolationforest('spambase', [500, 'auto', 'auto', False, False, False])
    
    
def drawFlipsAllDataset():
    df = pd.read_csv("Stats/SkIF_Trim.csv")
    df['FlippedPercentage'] = df['FlippedPercentage'].apply(lambda x: x*100)
    df['AvgFlippedPerRunPercentage'] = df['AvgFlippedPerRunPercentage'].apply(lambda x: x*100)
    df['ti_fo_per_all'] = df['ti_fo_per_all'].apply(lambda x: x*100)
    df['to_fi_per_all'] = df['to_fi_per_all'].apply(lambda x: x*100)
    df['ti_fo_per_avg'] = df['ti_fo_per_avg'].apply(lambda x: x*100)
    df['to_fi_per_avg'] = df['to_fi_per_avg'].apply(lambda x: x*100)
    
    df_def = df[df["Mode"] == 'Default'].sort_values(by=["to_fi_per_all"])
    
    # df.rename(columns={"ti_fo_per_all": "a", "to_fi_per_all": "c", "ti_fo_per_avg":"", "to_fi_per_avg":""})
    
    
    # g = plt.figure(figsize=(20, 5), dpi=80)
    # ax = df_def.plot.bar(x='Filename', y='FlippedPercentage', figsize=(20, 5), legend=None, xticks=[], xlabel="Dataset", ylabel="Flipped Points (%)")
    # ax.patches[27].set_facecolor('red')
    # plt.savefig("FlipFig/SkIF_AllDataset_Default_Combined.pdf", bbox_inches="tight", pad_inches=0)
    
    g = plt.figure(figsize=(9, 5), dpi=80)
    ax = df_def.plot.bar(x='Filename', y=['ti_fo_per_all','to_fi_per_all'], figsize=(10, 5), width=1, legend=None)
    # ax.patches[27].set_facecolor('red')
    plt.xlabel("Dataset",fontsize=20)
    plt.ylabel("Flipped Points (%)", fontsize=20)
    # plt.xticks(rotation = 45, horizontalalignment='right', fontsize=16)
    plt.xticks([])
    plt.yticks(fontsize=15)
    ax.legend(["True Inlier -> False Outlier", "True Outlier -> False Inlier"], fontsize=20)
    plt.savefig("Fig/SkIF_AllDataset_Default_Combined_Broken_no_name.pdf", bbox_inches="tight", pad_inches=0)
    
    
    
    df_def = df[df["Mode"] == 'Default'].sort_values(by=["to_fi_per_avg"])
    
    # g = plt.figure(figsize=(20, 5), dpi=80)
    # ax = df_def.plot.bar(x='Filename', y='AvgFlippedPerRunPercentage', figsize=(20, 5), legend=None, xticks=[], xlabel="Dataset", ylabel="Flipped Points (%)")
    # ax.patches[27].set_facecolor('red')
    # plt.savefig("FlipFig/SkIF_AllDataset_Default_Avg.pdf", bbox_inches="tight", pad_inches=0)
    
    
    g = plt.figure(figsize=(9, 5), dpi=80)
    ax = df_def.plot.bar(x='Filename', y=['ti_fo_per_avg','to_fi_per_avg'], figsize=(10, 5), width=1, legend=None)
    plt.xlabel("Dataset",fontsize=20)
    plt.ylabel("Flipped Points (%)", fontsize=20)
    # plt.xticks(rotation = 45, horizontalalignment='right', fontsize=16)
    plt.xticks([])
    plt.yticks(fontsize=15)

    ax.legend(["True Inlier -> False Outlier", "True Outlier -> False Inlier"], fontsize=20)
    plt.savefig("Fig/SkIF_AllDataset_Default_Avg_Broken_no_name.pdf", bbox_inches="tight", pad_inches=0)
    
