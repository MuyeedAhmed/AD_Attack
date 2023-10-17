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

# import matlab.engine
# eng = matlab.engine.start_matlab()

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
    
    
    
def runAlgo(imp, filename, X, gt, runs):
    flips = []
    labels = []
    for i in range(runs):
        if imp == "SkEE":
            clustering = EllipticEnvelope(contamination=0.1).fit(X)
    
            l = clustering.predict(X)
            l = [0 if x == 1 else 1 for x in l]
            labels.append(l)
        elif imp == "SkIF":
            clustering = IsolationForest().fit(X)
            l = clustering.predict(X)
            l = [0 if x == 1 else 1 for x in l]
            labels.append(l)
        elif imp == "MatEE":
            params = ["fmcd", 0.5, 500, 1, 2, "tauscale", "rfch", 10, "classical"]
            frr=open("GD_ReRun/MatEE.csv", "w")
            frr.write('Filename,Method,OutlierFraction,NumTrials,BiasCorrection,NumOGKIterations,UnivariateEstimator,ReweightingMethod,NumConcentrationSteps,StartMethod\n')
            frr.write(filename+","+str(params[0])+","+str(params[1])+","+str(params[2])+","+str(params[3])+","+str(params[4])+","+str(params[5])+","+str(params[6])+","+str(params[7])+","+str(params[8])+'\n')
            frr.close()
            time = eng.MatEE_Rerun(runs)
            labelFile = filename + "_" + str(params[0]) + "_" + str(params[1]) + "_" + str(params[2]) + "_" + str(params[3]) + "_" + str(params[4]) + "_" + str(params[5]) + "_" + str(params[6]) + "_" + str(params[7]) + "_" + str(params[8])
            if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile+".csv"):
                labels =  pd.read_csv("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile+".csv", header=None).to_numpy()
            else:
                print(filename, ": Not found!")
                return
        elif imp == "MatIF":
            params = [0.1, 100, 'auto']
            frr=open("GD_ReRun/MatIF.csv", "w")
            frr.write('Filename,ContaminationFraction,NumLearners,NumObservationsPerLearner\n')
            frr.write(filename+","+str(params[0])+","+str(params[1])+","+str(params[2])+'\n')
            frr.close()
            time = eng.MatIF_Rerun(runs)
            labelFile = filename + "_" + str(params[0]) + "_" + str(params[1]) + "_" + str(params[2])
            if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv"):
                labels =  pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv", header=None).to_numpy()
            else:
                print(filename, ": Not found!")
                return
        elif imp == "MatOCSVM":
            params = [0.1, 1, "auto", "auto", 0, 1e-4, 1e-4, 1000]
            
            frr=open("GD_ReRun/MatOCSVM.csv", "w")
            frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
            frr.write(filename+","+str(params[0])+","+str(params[1])+","+str(params[2])+","+str(params[3])+","+str(params[4])+","+str(params[5])+","+str(params[6])+","+str(params[7])+'\n')
            frr.close() 
            time = eng.MatOCSVM_Rerun(runs)

            labelFile = filename + "_" + str(params[0]) + "_" + str(params[1]) + "_" + str(params[2]) + "_" + str(params[3]) + "_" + str(params[4]) + "_" + str(params[5]) + "_" + str(params[6]) + "_" + str(params[7])
            if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv"):
                labels =  pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile+".csv", header=None).to_numpy()
            else:
                return
            
    flipped, p50 = flip_count(filename, gt, labels, runs)
    print(flipped, p50)
    return p50

def flip_count(filename, gt, labels, runs):
    flipped_list = [0]*len(gt)
    flipped_count = []
    print(len(labels))
    for i in range(runs-1):
        f = np.logical_xor(labels[i],labels[i+1])
        flipped_list = np.logical_or(flipped_list, f)
        flipped_count.append(sum(flipped_list))
    flipped = flipped_count[-1]
    if flipped == 0:
        return 0, -1
    runNumber50p=1
    for i in flipped_count:
        if i < flipped/2:
            runNumber50p+=1
    print(flipped_count, flipped, runNumber50p)
        
    return flipped, runNumber50p


def drawBeanPlot(df):
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, palette="Set3")
    plt.xlabel('')
    plt.ylabel('Restart', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xticks(rotation=30)

    plt.savefig("Fig/Flips_50.pdf", bbox_inches='tight')

    plt.show()

    

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
    df = pd.read_csv("Stats/Flips_50.csv")

    # for fname in master_files:
    #     # if os.path.exists("Fig/Time/SkEE_"+fname+".pdf"):
    #     #     # print(fname, " already done!")
    #     #     continue
    #     # if fname != "flare":
    #     #     continue
    #     X, gt = ReadFile(fname)
        
    #     p50 = runAlgo("MatEE", fname, X, gt, runs)
    #     df.loc[df['Filename'] == fname, 'Matlab/RobCov'] = p50
    #     p50 = runAlgo("MatIF", fname, X, gt, runs)
    #     df.loc[df['Filename'] == fname, 'Matlab/IF'] = p50
    #     p50 = runAlgo("MatOCSVM", fname, X, gt, runs)
    #     df.loc[df['Filename'] == fname, 'Matlab/OCSVM'] = p50

    # df.to_csv("Stats/Flips_50.csv")        
    df.replace(-1, np.nan, inplace=True)
    drawBeanPlot(df)
    # eng.quit()

    