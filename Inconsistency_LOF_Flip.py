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


import subprocess
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'


withGT = True

def lof(filename):
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
    
    lof_cont, labels_sk = LOF_ContFactor(X)
    
    f = open("GD_ReRun/LOF.csv", "w")
    f.write(filename+","+str(lof_cont))
    f.close()
    
    
    ## Default
    if os.path.exists("Labels/LOF_R/"+filename+"_Default.csv") == 0:
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "RLOF_Rerun.r"]))
            if os.path.exists("Labels/LOF_R/"+filename+"_Default.csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/LOF_Matlab/"+filename+"_Default.csv") == 0:        
        try:
            eng.MatLOF_Rerun(nargout=0)
            if os.path.exists("Labels/LOF_Matlab/"+filename+"_Default.csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
     
    labels_r = pd.read_csv("Labels/LOF_R/"+filename+"_Default.csv", header=None).to_numpy()
    labels_r = np.reshape(labels_r, (1, len(labels_r)))
    labels_mat = pd.read_csv("Labels/LOF_Matlab/"+filename+"_Default.csv", header=None).to_numpy()
    
    labels_r = labels_r[0]
    labels_mat = labels_mat[0]


    mvr, mvr_p = drawGraphs(labels_r, labels_mat)
    rvs, rvs_p = drawGraphs(labels_sk, labels_r)
    mvs, mvs_p = drawGraphs(labels_sk, labels_mat)
    
    f=open("Stats/LOF_Inconsistency.csv", "a")
    f.write(filename+',Default,'+str(mvr)+','+str(mvr_p)+','+str(rvs)+','+str(rvs_p)+','+str(mvs)+','+str(mvs_p) +'\n')
    f.close()
    
    if os.path.exists("Labels/LOF_R/"+filename+"_Mod.csv") == 0:
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "RLOF_Rerun.r"]))
            if os.path.exists("Labels/LOF_R/"+filename+"_Mod.csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/LOF_Matlab/"+filename+"_Mod.csv") == 0:        
        try:
            eng.MatLOF_Rerun(nargout=0)
            if os.path.exists("Labels/LOF_Matlab/"+filename+"_Mod.csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
            
    
    labels_r = pd.read_csv("Labels/LOF_R/"+filename+"_Mod.csv", header=None).to_numpy()
    labels_r = np.reshape(labels_r, (1, len(labels_r)))
    labels_mat = pd.read_csv("Labels/LOF_Matlab/"+filename+"_Mod.csv", header=None).to_numpy()
    
    labels_r = labels_r[0]
    labels_mat = labels_mat[0]
    
    mvr, mvr_p = drawGraphs(labels_r, labels_mat)
    rvs, rvs_p = drawGraphs(labels_sk, labels_r)
    mvs, mvs_p = drawGraphs(labels_sk, labels_mat)
    f=open("Stats/LOF_Inconsistency.csv", "a")
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


    

def LOF_ContFactor(X):
    labels_lof = LocalOutlierFactor().fit_predict(X)
    labels_lof = [0 if x == 1 else 1 for x in labels_lof]
    num_label = len(labels_lof)
    _, counts_lof = np.unique(labels_lof, return_counts=True)
    lof_per = min(counts_lof)/(num_label)
    
    if lof_per == 1:
        lof_per = 0
    
    return lof_per, labels_lof



if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    if os.path.exists("Stats/LOF_Inconsistency.csv"):
        df = pd.read_csv("Stats/LOF_Inconsistency.csv")
        done_files = df["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
    
    
    
    if os.path.exists("Stats/LOF_Inconsistency.csv")==0:
        f=open("Stats/LOF_Inconsistency.csv", "w")
        f.write('Filename,Mode,MatvR,MatvR_Percentage,RvSk,RvSk_Percentage,MatvSk,MatvSk_Percentage\n')
        f.close()
    
    for fname in master_files:
        lof(fname)
    
    