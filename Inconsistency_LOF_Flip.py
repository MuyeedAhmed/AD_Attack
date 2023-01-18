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


    mvr, mvr_p, r_tp_m_fn, r_tp_m_fn_per, r_tn_m_fp, r_tn_m_fp_per, m_tp_r_fn, m_tp_r_fn_per, m_tn_r_fp, m_tn_r_fp_per = drawGraphs(gt, labels_r, labels_mat)
    rvs, rvs_p, s_tp_r_fn, s_tp_r_fn_per, s_tn_r_fp, s_tn_r_fp_per, r_tp_s_fn, r_tp_s_fn_per, r_tn_s_fp, r_tn_s_fp_per = drawGraphs(gt, labels_sk, labels_r)
    mvs, mvs_p, s_tp_m_fn, s_tp_m_fn_per, s_tn_m_fp, s_tn_m_fp_per, m_tp_s_fn, m_tp_s_fn_per, m_tn_s_fp, m_tn_s_fp_per = drawGraphs(gt, labels_sk, labels_mat)
    
    f=open("Stats/LOF_Inconsistency.csv", "a")
    f.write(filename+',Default,'+str(mvr)+','+str(mvr_p)+','+str(rvs)+','+str(rvs_p)+','+str(mvs)+','+str(mvs_p)+",")
    f.write(r_tp_m_fn+","+r_tp_m_fn_per+","+r_tn_m_fp+","+r_tn_m_fp_per+","+m_tp_r_fn+","+m_tp_r_fn_per+","+m_tn_r_fp+","+m_tn_r_fp_per+",")
    f.write(s_tp_r_fn+","+ s_tp_r_fn_per+","+ s_tn_r_fp+","+ s_tn_r_fp_per+","+ r_tp_s_fn+","+ r_tp_s_fn_per+","+r_tn_s_fp+","+r_tn_s_fp_per+",")
    f.write(s_tp_m_fn+","+ s_tp_m_fn_per+","+ s_tn_m_fp+","+ s_tn_m_fp_per+","+ m_tp_s_fn+","+ m_tp_s_fn_per+","+ m_tn_s_fp+","+ m_tn_s_fp_per)
    f.write("\n")
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
    
    mvr, mvr_p, r_tp_m_fn, r_tp_m_fn_per, r_tn_m_fp, r_tn_m_fp_per, m_tp_r_fn, m_tp_r_fn_per, m_tn_r_fp, m_tn_r_fp_per = drawGraphs(gt, labels_r, labels_mat)
    rvs, rvs_p, s_tp_r_fn, s_tp_r_fn_per, s_tn_r_fp, s_tn_r_fp_per, r_tp_s_fn, r_tp_s_fn_per, r_tn_s_fp, r_tn_s_fp_per = drawGraphs(gt, labels_sk, labels_r)
    mvs, mvs_p, s_tp_m_fn, s_tp_m_fn_per, s_tn_m_fp, s_tn_m_fp_per, m_tp_s_fn, m_tp_s_fn_per, m_tn_s_fp, m_tn_s_fp_per = drawGraphs(gt, labels_sk, labels_mat)
    
    f=open("Stats/LOF_Inconsistency.csv", "a")
    f.write(filename+',Mod,'+str(mvr)+','+str(mvr_p)+','+str(rvs)+','+str(rvs_p)+','+str(mvs)+','+str(mvs_p) +",")
    f.write(r_tp_m_fn+","+r_tp_m_fn_per+","+r_tn_m_fp+","+r_tn_m_fp_per+","+m_tp_r_fn+","+m_tp_r_fn_per+","+m_tn_r_fp+","+m_tn_r_fp_per+",")
    f.write(s_tp_r_fn+","+ s_tp_r_fn_per+","+ s_tn_r_fp+","+ s_tn_r_fp_per+","+ r_tp_s_fn+","+ r_tp_s_fn_per+","+r_tn_s_fp+","+r_tn_s_fp_per+",")
    f.write(s_tp_m_fn+","+ s_tp_m_fn_per+","+ s_tn_m_fp+","+ s_tn_m_fp_per+","+ m_tp_s_fn+","+ m_tp_s_fn_per+","+ m_tn_s_fp+","+ m_tn_s_fp_per)
    f.write("\n")
    f.close()
    
    
    
def drawGraphs(gt, labels1, labels2):
    flipped = 0
    for i in range(len(gt)):
        avg = (labels1[i]+labels2[i])/2
        if avg != 0 and avg!= 1:
            flipped += 1

    l1_tp = 0
    l1_tp_l2_fn = 0
    l1_tn = 0
    l1_tn_l2_fp = 0
    
    l2_tp = 0
    l2_tp_l1_fn = 0
    l2_tn = 0
    l2_tn_l1_fp = 0
    
    for i in range(len(gt)):
        if labels1[i] == gt[i]:
            if labels1[i] == 0:
                l1_tp += 1
                if labels2[i] == 1:
                    l1_tp_l2_fn += 1
            else:
                l1_tn += 1
                if labels2[i] == 0:
                    l1_tn_l2_fp += 1
        
        if labels2[i] == gt[i]:
            if labels2[i] == 0:
                l2_tp += 1
                if labels1[i] == 1:
                    l2_tp_l1_fn += 1
            else:
                l2_tn += 1
                if labels1[i] == 0:
                    l2_tn_l1_fp += 1
    # print(l1_tp, l1_tp_l2_fn, l1_tn, l1_tn_l2_fp)
    
    if l1_tp != 0:
        l1_tp_l2_fn_per = (l1_tp_l2_fn/l1_tp)
    else:
        l1_tp_l2_fn_per = 0
    
    if l1_tn != 0:
        l1_tn_l2_fp_per = (l1_tn_l2_fp/l1_tn)
    else:
        l1_tn_l2_fp_per = 0
    if l2_tp != 0:
        l2_tp_l1_fn_per = (l2_tp_l1_fn/l2_tp)
    else:
        l2_tp_l1_fn_per = 0
    if l2_tn != 0:
        l2_tn_l1_fp_per = (l2_tn_l1_fp/l2_tn)
    else:
        l2_tn_l1_fp_per = 0
    
    return flipped, flipped/len(labels1), str(l1_tp_l2_fn), str(l1_tp_l2_fn_per), str(l1_tn_l2_fp), str(l1_tn_l2_fp_per), str(l2_tp_l1_fn), str(l2_tp_l1_fn_per), str(l2_tn_l1_fp), str(l2_tn_l1_fp_per)


    

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
    # g = [0,0,0,0,0,1,1,1]
    # l1 = [0,0,0,1,0,1,0,1]
    # l2 = [0,0,0,0,1,0,0,1]
    # drawGraphs(g,l1,l2)
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
        f.write('Filename,Mode,MatvR,MatvR_Percentage,RvSk,RvSk_Percentage,MatvSk,MatvSk_Percentage,')
        f.write("r_tp_m_fn"+","+"r_tp_m_fn_per"+","+"r_tn_m_fp"+","+"r_tn_m_fp_per"+","+"m_tp_r_fn"+","+"m_tp_r_fn_per"+","+"m_tn_r_fp"+","+"m_tn_r_fp_per"+",")
        f.write("s_tp_r_fn"+","+ "s_tp_r_fn_per"+","+ "s_tn_r_fp"+","+ "s_tn_r_fp_per"+","+ "r_tp_s_fn"+","+ "r_tp_s_fn_per"+","+"r_tn_s_fp"+","+"r_tn_s_fp_per"+",")
        f.write("s_tp_m_fn"+","+ "s_tp_m_fn_per"+","+ "s_tn_m_fp"+","+ "s_tn_m_fp_per"+","+ "m_tp_s_fn"+","+ "m_tp_s_fn_per"+","+ "m_tn_s_fp"+","+ "m_tn_s_fp_per")
        f.write("\n")    
        f.close()
    
    for fname in master_files:
        lof(fname)
    
    