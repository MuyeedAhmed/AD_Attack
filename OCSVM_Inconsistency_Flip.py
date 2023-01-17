import warnings
warnings.filterwarnings('ignore')
import sys
import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


import subprocess
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'


withGT = True

def ocsvm(filename):
    folderpath = datasetFolderDir
    global withGT
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        try:
            df = loadmat(folderpath+filename+".mat")
        except NotImplementedError:
            df = mat73.loadmat(folderpath+filename+".mat")

        gt=df["y"]
        gt = gt.reshape((len(gt)))
        X=df['X']
        if np.isnan(X).any():
            print("File contains NaN")
            return
    elif os.path.exists(folderpath+filename+".csv") == 1:
        X = pd.read_csv(folderpath+filename+".csv")
        if 'target' in X.columns:
            target=X["target"].to_numpy()
            X=X.drop("target", axis=1)
            gt = target
        else:
            gt = []
            withGT = False
        if X.isna().any().any() == 1:
            print("File contains NaN")
            return
    else:
        print("File doesn't exist")
        return
    
    if_cont = IF_ContFactor(X)
    
    clustering = OneClassSVM().fit(X)
    l = clustering.predict(X)
    labels_sk = [0 if x == 1 else 1 for x in l]
    
    param_r_default = ['radial', 3, "scale", 0, 0.001, 0.5, "TRUE", 200, 0.1]
    param_mat_default = [0.1, 1, 'auto', 'auto', 0, 0.0001, 0.0001, 1000]
    
    param_r_mod = ['radial', 3, "auto", 0, 0.001, if_cont, "TRUE", 200, 0.1]
    param_mat_mod = [if_cont, 1, 'auto', 'auto', 0, 0.0001, 0.0001, 1000]
    
    
    param_sk_mod = ['rbf', 3, 'auto', 0.0, 0.001, if_cont, True, 200, -1]
    
    frr=open("GD_ReRun/ROCSVM.csv", "a")
    frr.write(filename+","+str(param_r_default[0])+","+str(param_r_default[1])+","+str(param_r_default[2])+","+str(param_r_default[3])+","+str(param_r_default[4])+","+str(param_r_default[5])+","+str(param_r_default[6])+","+str(param_r_default[7])+","+str(param_r_default[8])+'\n')
    frr.write(filename+","+str(param_r_mod[0])+","+str(param_r_mod[1])+","+str(param_r_mod[2])+","+str(param_r_mod[3])+","+str(param_r_mod[4])+","+str(param_r_mod[5])+","+str(param_r_mod[6])+","+str(param_r_mod[7])+","+str(param_r_mod[8])+'\n')
    frr.close()
    frr=open("GD_ReRun/MatOCSVM.csv", "a")
    frr.write(filename+","+str(param_mat_default[0])+","+str(param_mat_default[1])+","+str(param_mat_default[2])+","+str(param_mat_default[3])+","+str(param_mat_default[4])+","+str(param_mat_default[5])+","+str(param_mat_default[6])+","+str(param_mat_default[7])+'\n')
    frr.write(filename+","+str(param_mat_mod[0])+","+str(param_mat_mod[1])+","+str(param_mat_mod[2])+","+str(param_mat_mod[3])+","+str(param_mat_mod[4])+","+str(param_mat_mod[5])+","+str(param_mat_mod[6])+","+str(param_mat_mod[7])+'\n')
    frr.close()
    
    labelFile_r_default = filename + "_" + str(param_r_default[0]) + "_" + str(param_r_default[1]) + "_" + str(param_r_default[2]) + "_" + str(param_r_default[3]) + "_" + str(param_r_default[4]) + "_" + str(param_r_default[5]) + "_" + str(param_r_default[6]) + "_" + str(param_r_default[7]) + "_" + str(param_r_default[8])
    labelFile_mat_default = filename + "_" + str(param_mat_default[0]) + "_" + str(param_mat_default[1]) + "_" + str(param_mat_default[2]) + "_" + str(param_mat_default[3]) + "_" + str(param_mat_default[4]) + "_" + str(param_mat_default[5]) + "_" + str(param_mat_default[6]) + "_" + str(param_mat_default[7])
    
    
    ## Default
    if os.path.exists("Labels/OCSVM_R/"+labelFile_r_default+".csv") == 0:
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "ROCSVM_Rerun.r"]))
            if os.path.exists("Labels/OCSVM_R/"+labelFile_r_default+".csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat_default+".csv") == 0:        
        try:
            eng.MatOCSVM_Rerun(nargout=0)
            if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat_default+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
    
    labels_r = pd.read_csv("Labels/OCSVM_R/"+labelFile_r_default+".csv", header=None).to_numpy()
    labels_r = np.reshape(labels_r, (1, len(labels_r)))
    labels_mat = pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat_default+".csv", header=None).to_numpy()
    
    labels_r = labels_r[0]
    labels_mat = labels_mat[0]


    mvr, mvr_p = drawGraphs(labels_r, labels_mat)
    rvs, rvs_p = drawGraphs(labels_sk, labels_r)
    mvs, mvs_p = drawGraphs(labels_sk, labels_mat)
    
    f=open("Stats/OCSVM_Inconsistency.csv", "a")
    f.write(filename+',Default,'+str(mvr)+','+str(mvr_p)+','+str(rvs)+','+str(rvs_p)+','+str(mvs)+','+str(mvs_p) +'\n')
    f.close()
    
    ## Mod
    clustering = OneClassSVM(kernel=param_sk_mod[0], degree=param_sk_mod[1], gamma=param_sk_mod[2], coef0=param_sk_mod[3], tol=param_sk_mod[4], nu=param_sk_mod[5], 
                              shrinking=param_sk_mod[6], cache_size=param_sk_mod[7], max_iter=param_sk_mod[8]).fit(X)
    l = clustering.predict(X)
    labels_sk = [0 if x == 1 else 1 for x in l]
    
    labelFile_r_mod = filename + "_" + str(param_r_mod[0]) + "_" + str(param_r_mod[1]) + "_" + str(param_r_mod[2]) + "_" + str(param_r_mod[3]) + "_" + str(param_r_mod[4]) + "_" + str(param_r_mod[5]) + "_" + str(param_r_mod[6]) + "_" + str(param_r_mod[7]) + "_" + str(param_r_mod[8])
    labelFile_mat_mod = filename + "_" + str(param_mat_mod[0]) + "_" + str(param_mat_mod[1]) + "_" + str(param_mat_mod[2]) + "_" + str(param_mat_mod[3]) + "_" + str(param_mat_mod[4]) + "_" + str(param_mat_mod[5]) + "_" + str(param_mat_mod[6]) + "_" + str(param_mat_mod[7])
    
    
    if os.path.exists("Labels/OCSVM_R/"+labelFile_r_mod+".csv") == 0:
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "ROCSVM_Rerun.r"]))
            if os.path.exists("Labels/OCSVM_R/"+labelFile_r_mod+".csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat_mod+".csv") == 0:        
        try:
            eng.MatOCSVM_Rerun(nargout=0)
            if os.path.exists("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat_mod+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
            
    
    labels_r = pd.read_csv("Labels/OCSVM_R/"+labelFile_r_mod+".csv", header=None).to_numpy()
    labels_r = np.reshape(labels_r, (1, len(labels_r)))
    labels_mat = pd.read_csv("Labels/OCSVM_Matlab/Labels_Mat_OCSVM_"+labelFile_mat_mod+".csv", header=None).to_numpy()
    
    labels_r = labels_r[0]
    labels_mat = labels_mat[0]
    
    mvr, mvr_p = drawGraphs(labels_r, labels_mat)
    rvs, rvs_p = drawGraphs(labels_sk, labels_r)
    mvs, mvs_p = drawGraphs(labels_sk, labels_mat)
    f=open("Stats/OCSVM_Inconsistency.csv", "a")
    f.write(filename+',Mod,'+str(mvr)+','+str(mvr_p)+','+str(rvs)+','+str(rvs_p)+','+str(mvs)+','+str(mvs_p) +'\n')
    f.close()
    
    
def drawGraphs(labels1, labels2):
    norms = 0
    outliers = 0
    flipped = 0

    
    for i in range(len(labels1)):

        avg = (labels1[i]+labels2[i])/2
        if avg == 0:
            norms += 1
        elif avg == 1:
            outliers += 1
        else:
            flipped += 1

    return flipped, flipped/len(labels1)

def IF_ContFactor(X):
    labels = []
    num_label = 0
    for i in range(5):
        clustering = IsolationForest().fit(X)
    
        l = clustering.predict(X)
        num_label = len(l)
        l = [0 if x == 1 else 1 for x in l]
        labels.append(l)
    _, counts_if = np.unique(labels, return_counts=True)
    if_per = min(counts_if)/(num_label*5)

    if if_per == 1:
        if_per = 0
        
        
    return round(if_per, 3)  

if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    if os.path.exists("Stats/OCSVM_Inconsistency.csv"):
        df = pd.read_csv("Stats/OCSVM_Inconsistency.csv")
        done_files = df["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
    
    frr=open("GD_ReRun/ROCSVM.csv", "w")
    frr.write('Filename,kernel,degree,gamma,coef0,tolerance,nu,shrinking,cachesize,epsilon\n')
    frr.close()
    
    frr=open("GD_ReRun/MatOCSVM.csv", "w")
    frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
    frr.close()
    
    
    if os.path.exists("Stats/OCSVM_Inconsistency.csv")==0:
        f=open("Stats/OCSVM_Inconsistency.csv", "w")
        f.write('Filename,Mode,MatvR,MatvR_Percentage,RvSk,RvSk_Percentage,MatvSk,MatvSk_Percentage\n')
        f.close()
    
    for fname in master_files:
        frr=open("GD_ReRun/ROCSVM.csv", "w")
        frr.write('Filename,kernel,degree,gamma,coef0,tolerance,nu,shrinking,cachesize,epsilon\n')
        frr.close()
        
        frr=open("GD_ReRun/MatOCSVM.csv", "w")
        frr.write('Filename,ContaminationFraction,KernelScale,Lambda,NumExpansionDimensions,StandardizeData,BetaTolerance,BetaTolerance,GradientTolerance,IterationLimit\n')
        frr.close()
        
        ocsvm(fname)
        
    
    