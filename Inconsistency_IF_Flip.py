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

from sklearn.metrics import f1_score
from scipy.stats import gmean

import subprocess
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'


withGT = True

def isolationforest(filename):
    print(filename)
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
    
    runs = 50
    
    if_cont = IF_ContFactor(X)
    
    if if_cont == 0:
        if_cont = 0.1
    
    
    param_r_default = [500, "TRUE", "auto", "def"]
    param_mat_default = [0.1, 100, "auto"]
    
    param_r_mod = [500, "TRUE", "auto", "def"]
    param_mat_mod = [if_cont, 500, "auto"]
    
    
    param_sk_mod = [500, 'auto', 'auto', 1.0, False, None, False]
    
    frr=open("GD_ReRun/RIF.csv", "a")
    frr.write(filename+","+str(param_r_default[0])+","+str(param_r_default[1])+","+str(param_r_default[2])+","+str(param_r_default[3])+'\n')
    frr.write(filename+","+str(param_r_mod[0])+","+str(param_r_mod[1])+","+str(param_r_mod[2])+","+str(param_r_mod[3])+'\n')
    frr.close()
    frr=open("GD_ReRun/MatIF.csv", "a")
    frr.write(filename+","+str(param_mat_default[0])+","+str(param_mat_default[1])+","+str(param_mat_default[2])+'\n')
    frr.write(filename+","+str(param_mat_mod[0])+","+str(param_mat_mod[1])+","+str(param_mat_mod[2])+'\n')
    frr.close()
    
    
    ## Default
    labels_sk = []
    for i in range(runs):
        clustering = IsolationForest().fit(X)
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels_sk.append(l)
        
    labelFile_r_default = filename + "_" + str(param_r_default[0]) + "_" + str(param_r_default[1]) + "_" + str(param_r_default[2]) + "_" + str(param_r_default[3])
    labelFile_mat_default = filename + "_" + str(param_mat_default[0]) + "_" + str(param_mat_default[1]) + "_" + str(param_mat_default[2])
    
    if os.path.exists("Labels/IF_R/"+labelFile_r_default+".csv") == 0:
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "RIF_Rerun.r"])) # Set runs manually in RIF_Rerun.r
            if os.path.exists("Labels/IF_R/"+labelFile_r_default+".csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat_default+".csv") == 0:        
        try:
            eng.MatIF_Rerun(runs)
            if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat_default+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
    
    labels_r = pd.read_csv("Labels/IF_R/"+labelFile_r_default+".csv").to_numpy()
    # labels_r = np.int64((labels_r[0][1:])*1)
    labels_mat = pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat_default+".csv", header=None).to_numpy()
    
    
    mvs, mvs_p, s_tp_m_fn, s_tp_m_fn_per, s_tn_m_fp, s_tn_m_fp_per, m_tp_s_fn, m_tp_s_fn_per, m_tn_s_fp, m_tn_s_fp_per = [], [], [], [], [], [], [], [], [], []
    mvr, mvr_p, r_tp_m_fn, r_tp_m_fn_per, r_tn_m_fp, r_tn_m_fp_per, m_tp_r_fn, m_tp_r_fn_per, m_tn_r_fp, m_tn_r_fp_per = [], [], [], [], [], [], [], [], [], []
    rvs, rvs_p, s_tp_r_fn, s_tp_r_fn_per, s_tn_r_fp, s_tn_r_fp_per, r_tp_s_fn, r_tp_s_fn_per, r_tn_s_fp, r_tn_s_fp_per = [], [], [], [], [], [], [], [], [], []

    for m in range(runs):
        for n in range(runs):            
            a,b,c,d,e,f,g,h,i,j = drawGraphs(gt, labels_sk[m], labels_mat[n])
            mvs.append(a)
            mvs_p.append(b)
            s_tp_m_fn.append(c)
            s_tp_m_fn_per.append(d)
            s_tn_m_fp.append(e)
            s_tn_m_fp_per.append(f)
            m_tp_s_fn.append(g)
            m_tp_s_fn_per.append(h)
            m_tn_s_fp.append(i)
            m_tn_s_fp_per.append(j)
            
    mvs_max_index = mvs.index(max(mvs))
    mvs_min_index = mvs.index(min(mvs))
    mvs_max, mvs_p_max, s_tp_m_fn_max, s_tp_m_fn_per_max, s_tn_m_fp_max, s_tn_m_fp_per_max, m_tp_s_fn_max, m_tp_s_fn_per_max, m_tn_s_fp_max, m_tn_s_fp_per_max = mvs[mvs_max_index], mvs_p[mvs_max_index], s_tp_m_fn[mvs_max_index], s_tp_m_fn_per[mvs_max_index], s_tn_m_fp[mvs_max_index], s_tn_m_fp_per[mvs_max_index], m_tp_s_fn[mvs_max_index], m_tp_s_fn_per[mvs_max_index], m_tn_s_fp[mvs_max_index], m_tn_s_fp_per[mvs_max_index]
    mvs_min, mvs_p_min, s_tp_m_fn_min, s_tp_m_fn_per_min, s_tn_m_fp_min, s_tn_m_fp_per_min, m_tp_s_fn_min, m_tp_s_fn_per_min, m_tn_s_fp_min, m_tn_s_fp_per_min = mvs[mvs_min_index], mvs_p[mvs_min_index], s_tp_m_fn[mvs_min_index], s_tp_m_fn_per[mvs_min_index], s_tn_m_fp[mvs_min_index], s_tn_m_fp_per[mvs_min_index], m_tp_s_fn[mvs_min_index], m_tp_s_fn_per[mvs_min_index], m_tn_s_fp[mvs_min_index], m_tn_s_fp_per[mvs_min_index]
    s_tp_m_fn_per, s_tn_m_fp_per, m_tp_s_fn_per, m_tn_s_fp_per = [(i * 100)+1 for i in s_tp_m_fn_per], [(i * 100)+1 for i in s_tn_m_fp_per], [(i * 100)+1 for i in m_tp_s_fn_per], [(i * 100)+1 for i in m_tn_s_fp_per]
    s_tp_m_fn_avg, s_tn_m_fp_avg, m_tp_s_fn_avg, m_tn_s_fp_avg = gmean(s_tp_m_fn_per)-1, gmean(s_tn_m_fp_per)-1, gmean(m_tp_s_fn_per)-1, gmean(m_tn_s_fp_per)-1
            

    for m in range(runs):
        for n in range(runs):
            a,b,c,d,e,f,g,h,i,j = drawGraphs(gt, np.int64((labels_r[m][1:])*1), labels_mat[n])
            mvr.append(a)
            mvr_p.append(b)
            r_tp_m_fn.append(c)
            r_tp_m_fn_per.append(d)
            r_tn_m_fp.append(e)
            r_tn_m_fp_per.append(f)
            m_tp_r_fn.append(g)
            m_tp_r_fn_per.append(h)
            m_tn_r_fp.append(i)
            m_tn_r_fp_per.append(j)


    mvr_max_index = mvr.index(max(mvr))
    mvr_min_index = mvr.index(min(mvr))
    mvr_max, mvr_p_max, r_tp_m_fn_max, r_tp_m_fn_per_max, r_tn_m_fp_max, r_tn_m_fp_per_max, m_tp_r_fn_max, m_tp_r_fn_per_max, m_tn_r_fp_max, m_tn_r_fp_per_max = mvr[mvr_max_index], mvr_p[mvr_max_index], r_tp_m_fn[mvr_max_index], r_tp_m_fn_per[mvr_max_index], r_tn_m_fp[mvr_max_index], r_tn_m_fp_per[mvr_max_index], m_tp_r_fn[mvr_max_index], m_tp_r_fn_per[mvr_max_index], m_tn_r_fp[mvr_max_index], m_tn_r_fp_per[mvr_max_index]
    mvr_min, mvr_p_min, r_tp_m_fn_min, r_tp_m_fn_per_min, r_tn_m_fp_min, r_tn_m_fp_per_min, m_tp_r_fn_min, m_tp_r_fn_per_min, m_tn_r_fp_min, m_tn_r_fp_per_min = mvr[mvr_min_index], mvr_p[mvr_min_index], r_tp_m_fn[mvr_min_index], r_tp_m_fn_per[mvr_min_index], r_tn_m_fp[mvr_min_index], r_tn_m_fp_per[mvr_min_index], m_tp_r_fn[mvr_min_index], m_tp_r_fn_per[mvr_min_index], m_tn_r_fp[mvr_min_index], m_tn_r_fp_per[mvr_min_index]
    r_tp_m_fn_per, r_tn_m_fp_per, m_tp_r_fn_per, m_tn_r_fp_per = [(i * 100)+1 for i in r_tp_m_fn_per], [(i * 100)+1 for i in r_tn_m_fp_per], [(i * 100)+1 for i in m_tp_r_fn_per], [(i * 100)+1 for i in m_tn_r_fp_per]
    r_tp_m_fn_avg, r_tn_m_fp_avg, m_tp_r_fn_avg, m_tn_r_fp_avg = gmean(r_tp_m_fn_per)-1, gmean(r_tn_m_fp_per)-1, gmean(m_tp_r_fn_per)-1, gmean(m_tn_r_fp_per)-1
    
    for m in range(runs):
        for n in range(runs):
            a,b,c,d,e,f,g,h,i,j = drawGraphs(gt, labels_sk[m], np.int64((labels_r[n][1:])*1))
            rvs.append(a)
            rvs_p.append(b)
            s_tp_r_fn.append(c)
            s_tp_r_fn_per.append(d)
            s_tn_r_fp.append(e)
            s_tn_r_fp_per.append(f)
            r_tp_s_fn.append(g)
            r_tp_s_fn_per.append(h)
            r_tn_s_fp.append(i)
            r_tn_s_fp_per.append(j)


    rvs_max_index = rvs.index(max(rvs))
    rvs_min_index = rvs.index(min(rvs))
    rvs_max, rvs_p_max, s_tp_r_fn_max, s_tp_r_fn_per_max, s_tn_r_fp_max, s_tn_r_fp_per_max, r_tp_s_fn_max, r_tp_s_fn_per_max, r_tn_s_fp_max, r_tn_s_fp_per_max = rvs[rvs_max_index], rvs_p[rvs_max_index], s_tp_r_fn[rvs_max_index], s_tp_r_fn_per[rvs_max_index], s_tn_r_fp[rvs_max_index], s_tn_r_fp_per[rvs_max_index], r_tp_s_fn[rvs_max_index], r_tp_s_fn_per[rvs_max_index], r_tn_s_fp[rvs_max_index], r_tn_s_fp_per[rvs_max_index]
    rvs_min, rvs_p_min, s_tp_r_fn_min, s_tp_r_fn_per_min, s_tn_r_fp_min, s_tn_r_fp_per_min, r_tp_s_fn_min, r_tp_s_fn_per_min, r_tn_s_fp_min, r_tn_s_fp_per_min = rvs[rvs_min_index], rvs_p[rvs_min_index], s_tp_r_fn[rvs_min_index], s_tp_r_fn_per[rvs_min_index], s_tn_r_fp[rvs_min_index], s_tn_r_fp_per[rvs_min_index], r_tp_s_fn[rvs_min_index], r_tp_s_fn_per[rvs_min_index], r_tn_s_fp[rvs_min_index], r_tn_s_fp_per[rvs_min_index]
    s_tp_r_fn_per, s_tn_r_fp_per, r_tp_s_fn_per, r_tn_s_fp_per = [(i * 100)+1 for i in s_tp_r_fn_per], [(i * 100)+1 for i in s_tn_r_fp_per], [(i * 100)+1 for i in r_tp_s_fn_per], [(i * 100)+1 for i in r_tn_s_fp_per]
    s_tp_r_fn_avg, s_tn_r_fp_avg, r_tp_s_fn_avg, r_tn_s_fp_avg = gmean(s_tp_r_fn_per)-1, gmean(s_tn_r_fp_per)-1, gmean(r_tp_s_fn_per)-1, gmean(r_tn_s_fp_per)-1
    
    

    
    f=open("Stats/IF_Inconsistency_Max.csv", "a")
    f.write(filename+',Default,'+str(mvr_max)+','+str(mvr_p_max)+','+str(rvs_max)+','+str(rvs_p_max)+','+str(mvs_max)+','+str(mvs_p_max) +",")
    f.write(str(r_tp_m_fn_max)+","+str(r_tp_m_fn_per_max)+","+str(r_tn_m_fp_max)+","+str(r_tn_m_fp_per_max)+","+str(m_tp_r_fn_max)+","+str(m_tp_r_fn_per_max)+","+str(m_tn_r_fp_max)+","+str(m_tn_r_fp_per_max)+",")
    f.write(str(s_tp_r_fn_max)+","+str(s_tp_r_fn_per_max)+","+str(s_tn_r_fp_max)+","+str(s_tn_r_fp_per_max)+","+str(r_tp_s_fn_max)+","+str(r_tp_s_fn_per_max)+","+str(r_tn_s_fp_max)+","+str(r_tn_s_fp_per_max)+",")
    f.write(str(s_tp_m_fn_max)+","+str(s_tp_m_fn_per_max)+","+str(s_tn_m_fp_max)+","+str(s_tn_m_fp_per_max)+","+str(m_tp_s_fn_max)+","+str(m_tp_s_fn_per_max)+","+str(m_tn_s_fp_max)+","+str(m_tn_s_fp_per_max))
    f.write("\n")
    f.close()
    f=open("Stats/IF_Inconsistency_Min.csv", "a")
    f.write(filename+',Default,'+str(mvr_min)+','+str(mvr_p_min)+','+str(rvs_min)+','+str(rvs_p_min)+','+str(mvs_min)+','+str(mvs_p_min) +",")
    f.write(str(r_tp_m_fn_min)+","+str(r_tp_m_fn_per_min)+","+str(r_tn_m_fp_min)+","+str(r_tn_m_fp_per_min)+","+str(m_tp_r_fn_min)+","+str(m_tp_r_fn_per_min)+","+str(m_tn_r_fp_min)+","+str(m_tn_r_fp_per_min)+",")
    f.write(str(s_tp_r_fn_min)+","+str(s_tp_r_fn_per_min)+","+str(s_tn_r_fp_min)+","+str(s_tn_r_fp_per_min)+","+str(r_tp_s_fn_min)+","+str(r_tp_s_fn_per_min)+","+str(r_tn_s_fp_min)+","+str(r_tn_s_fp_per_min)+",")
    f.write(str(s_tp_m_fn_min)+","+str(s_tp_m_fn_per_min)+","+str(s_tn_m_fp_min)+","+str(s_tn_m_fp_per_min)+","+str(m_tp_s_fn_min)+","+str(m_tp_s_fn_per_min)+","+str(m_tn_s_fp_min)+","+str(m_tn_s_fp_per_min))
    f.write("\n")
    f.close()
    f=open("Stats/IF_Inconsistency_Max_Min.csv", "a")
    f.write(filename+',Default,'+str(mvr_p_min)+','+str(mvr_p_max)+','+str(rvs_p_min)+','+str(rvs_p_max)+','+str(mvs_p_min)+','+str(mvs_p_max) +",")
    f.write(str(r_tp_m_fn_per_min)+","+str(r_tp_m_fn_avg)+","+str(r_tp_m_fn_per_max)+","+str(r_tn_m_fp_per_min)+","+str(r_tn_m_fp_avg)+","+str(r_tn_m_fp_per_max)+","+str(m_tp_r_fn_per_min)+","+str(m_tp_r_fn_avg)+","+str(m_tp_r_fn_per_max)+","+str(m_tn_r_fp_per_min)+","+str(m_tn_r_fp_avg)+","+str(m_tn_r_fp_per_max)+",")
    f.write(str(s_tp_r_fn_per_min)+","+str(s_tp_r_fn_avg)+","+str(s_tp_r_fn_per_max)+","+str(s_tn_r_fp_per_min)+","+str(s_tn_r_fp_avg)+","+str(s_tn_r_fp_per_max)+","+str(r_tp_s_fn_per_min)+","+str(r_tp_s_fn_avg)+","+str(r_tp_s_fn_per_max)+","+str(r_tn_s_fp_per_min)+","+str(r_tn_s_fp_avg)+","+str(r_tn_s_fp_per_max)+",")
    f.write(str(s_tp_m_fn_per_min)+","+str(s_tp_m_fn_avg)+","+str(s_tp_m_fn_per_max)+","+str(s_tn_m_fp_per_min)+","+str(s_tn_m_fp_avg)+","+str(s_tn_m_fp_per_max)+","+str(m_tp_s_fn_per_min)+","+str(m_tp_s_fn_avg)+","+str(m_tp_s_fn_per_max)+","+str(m_tn_s_fp_per_min)+","+str(m_tn_s_fp_avg)+","+str(m_tn_s_fp_per_max))
    f.write("\n")
    f.close()
    
    ## Mod
    labels_sk = []
    for i in range(runs):
        clustering = IsolationForest(n_estimators=param_sk_mod[0], max_samples=param_sk_mod[1], 
                                      max_features=param_sk_mod[3], bootstrap=param_sk_mod[4], 
                                      n_jobs=param_sk_mod[5], warm_start=param_sk_mod[6]).fit(X)
            
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels_sk.append(l)
    
    labelFile_r_mod = filename + "_" + str(param_r_mod[0]) + "_" + str(param_r_mod[1]) + "_" + str(param_r_mod[2]) + "_" + str(param_r_mod[3])
    labelFile_mat_mod = filename + "_" + str(param_mat_mod[0]) + "_" + str(param_mat_mod[1]) + "_" + str(param_mat_mod[2])
    
    
    if os.path.exists("Labels/IF_R/"+labelFile_r_mod+".csv") == 0:
        try:
            subprocess.call((["/usr/local/bin/Rscript", "--vanilla", "RIF_Rerun.r"]))
            if os.path.exists("Labels/IF_R/"+labelFile_r_mod+".csv") == 0:      
                print("\nFaild to run Rscript from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Rscript from Python.\n")
            exit(0)  
    if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat_mod+".csv") == 0:        
        try:
            eng.MatIF_Rerun(runs)
            if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat_mod+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
            
    
    labels_r = pd.read_csv("Labels/IF_R/"+labelFile_r_mod+".csv").to_numpy()
    # labels_r = np.int64((labels_r[0][1:])*1)
    labels_mat = pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile_mat_mod+".csv", header=None).to_numpy()
    
    
    mvs, mvs_p, s_tp_m_fn, s_tp_m_fn_per, s_tn_m_fp, s_tn_m_fp_per, m_tp_s_fn, m_tp_s_fn_per, m_tn_s_fp, m_tn_s_fp_per = [], [], [], [], [], [], [], [], [], []
    mvr, mvr_p, r_tp_m_fn, r_tp_m_fn_per, r_tn_m_fp, r_tn_m_fp_per, m_tp_r_fn, m_tp_r_fn_per, m_tn_r_fp, m_tn_r_fp_per = [], [], [], [], [], [], [], [], [], []
    rvs, rvs_p, s_tp_r_fn, s_tp_r_fn_per, s_tn_r_fp, s_tn_r_fp_per, r_tp_s_fn, r_tp_s_fn_per, r_tn_s_fp, r_tn_s_fp_per = [], [], [], [], [], [], [], [], [], []
    
    for m in range(runs):
        for n in range(runs):
            a,b,c,d,e,f,g,h,i,j = drawGraphs(gt, labels_sk[m], labels_mat[n])
            mvs.append(a)
            mvs_p.append(b)
            s_tp_m_fn.append(c)
            s_tp_m_fn_per.append(d)
            s_tn_m_fp.append(e)
            s_tn_m_fp_per.append(f)
            m_tp_s_fn.append(g)
            m_tp_s_fn_per.append(h)
            m_tn_s_fp.append(i)
            m_tn_s_fp_per.append(j)
            
    mvs_max_index = mvs.index(max(mvs))
    mvs_min_index = mvs.index(min(mvs))
    mvs_max, mvs_p_max, s_tp_m_fn_max, s_tp_m_fn_per_max, s_tn_m_fp_max, s_tn_m_fp_per_max, m_tp_s_fn_max, m_tp_s_fn_per_max, m_tn_s_fp_max, m_tn_s_fp_per_max = mvs[mvs_max_index], mvs_p[mvs_max_index], s_tp_m_fn[mvs_max_index], s_tp_m_fn_per[mvs_max_index], s_tn_m_fp[mvs_max_index], s_tn_m_fp_per[mvs_max_index], m_tp_s_fn[mvs_max_index], m_tp_s_fn_per[mvs_max_index], m_tn_s_fp[mvs_max_index], m_tn_s_fp_per[mvs_max_index]
    mvs_min, mvs_p_min, s_tp_m_fn_min, s_tp_m_fn_per_min, s_tn_m_fp_min, s_tn_m_fp_per_min, m_tp_s_fn_min, m_tp_s_fn_per_min, m_tn_s_fp_min, m_tn_s_fp_per_min = mvs[mvs_min_index], mvs_p[mvs_min_index], s_tp_m_fn[mvs_min_index], s_tp_m_fn_per[mvs_min_index], s_tn_m_fp[mvs_min_index], s_tn_m_fp_per[mvs_min_index], m_tp_s_fn[mvs_min_index], m_tp_s_fn_per[mvs_min_index], m_tn_s_fp[mvs_min_index], m_tn_s_fp_per[mvs_min_index]
    s_tp_m_fn_per, s_tn_m_fp_per, m_tp_s_fn_per, m_tn_s_fp_per = [(i * 100)+1 for i in s_tp_m_fn_per], [(i * 100)+1 for i in s_tn_m_fp_per], [(i * 100)+1 for i in m_tp_s_fn_per], [(i * 100)+1 for i in m_tn_s_fp_per]
    s_tp_m_fn_avg, s_tn_m_fp_avg, m_tp_s_fn_avg, m_tn_s_fp_avg = gmean(s_tp_m_fn_per)-1, gmean(s_tn_m_fp_per)-1, gmean(m_tp_s_fn_per)-1, gmean(m_tn_s_fp_per)-1
            

    for m in range(runs):
        for n in range(runs):
            a,b,c,d,e,f,g,h,i,j = drawGraphs(gt, np.int64((labels_r[m][1:])*1), labels_mat[n])
            mvr.append(a)
            mvr_p.append(b)
            r_tp_m_fn.append(c)
            r_tp_m_fn_per.append(d)
            r_tn_m_fp.append(e)
            r_tn_m_fp_per.append(f)
            m_tp_r_fn.append(g)
            m_tp_r_fn_per.append(h)
            m_tn_r_fp.append(i)
            m_tn_r_fp_per.append(j)


    mvr_max_index = mvr.index(max(mvr))
    mvr_min_index = mvr.index(min(mvr))
    mvr_max, mvr_p_max, r_tp_m_fn_max, r_tp_m_fn_per_max, r_tn_m_fp_max, r_tn_m_fp_per_max, m_tp_r_fn_max, m_tp_r_fn_per_max, m_tn_r_fp_max, m_tn_r_fp_per_max = mvr[mvr_max_index], mvr_p[mvr_max_index], r_tp_m_fn[mvr_max_index], r_tp_m_fn_per[mvr_max_index], r_tn_m_fp[mvr_max_index], r_tn_m_fp_per[mvr_max_index], m_tp_r_fn[mvr_max_index], m_tp_r_fn_per[mvr_max_index], m_tn_r_fp[mvr_max_index], m_tn_r_fp_per[mvr_max_index]
    mvr_min, mvr_p_min, r_tp_m_fn_min, r_tp_m_fn_per_min, r_tn_m_fp_min, r_tn_m_fp_per_min, m_tp_r_fn_min, m_tp_r_fn_per_min, m_tn_r_fp_min, m_tn_r_fp_per_min = mvr[mvr_min_index], mvr_p[mvr_min_index], r_tp_m_fn[mvr_min_index], r_tp_m_fn_per[mvr_min_index], r_tn_m_fp[mvr_min_index], r_tn_m_fp_per[mvr_min_index], m_tp_r_fn[mvr_min_index], m_tp_r_fn_per[mvr_min_index], m_tn_r_fp[mvr_min_index], m_tn_r_fp_per[mvr_min_index]
    r_tp_m_fn_per, r_tn_m_fp_per, m_tp_r_fn_per, m_tn_r_fp_per = [(i * 100)+1 for i in r_tp_m_fn_per], [(i * 100)+1 for i in r_tn_m_fp_per], [(i * 100)+1 for i in m_tp_r_fn_per], [(i * 100)+1 for i in m_tn_r_fp_per]
    r_tp_m_fn_avg, r_tn_m_fp_avg, m_tp_r_fn_avg, m_tn_r_fp_avg = gmean(r_tp_m_fn_per)-1, gmean(r_tn_m_fp_per)-1, gmean(m_tp_r_fn_per)-1, gmean(m_tn_r_fp_per)-1
    
    for m in range(runs):
        for n in range(runs):
            a,b,c,d,e,f,g,h,i,j = drawGraphs(gt, labels_sk[m], np.int64((labels_r[n][1:])*1))
            rvs.append(a)
            rvs_p.append(b)
            s_tp_r_fn.append(c)
            s_tp_r_fn_per.append(d)
            s_tn_r_fp.append(e)
            s_tn_r_fp_per.append(f)
            r_tp_s_fn.append(g)
            r_tp_s_fn_per.append(h)
            r_tn_s_fp.append(i)
            r_tn_s_fp_per.append(j)


    rvs_max_index = rvs.index(max(rvs))
    rvs_min_index = rvs.index(min(rvs))
    rvs_max, rvs_p_max, s_tp_r_fn_max, s_tp_r_fn_per_max, s_tn_r_fp_max, s_tn_r_fp_per_max, r_tp_s_fn_max, r_tp_s_fn_per_max, r_tn_s_fp_max, r_tn_s_fp_per_max = rvs[rvs_max_index], rvs_p[rvs_max_index], s_tp_r_fn[rvs_max_index], s_tp_r_fn_per[rvs_max_index], s_tn_r_fp[rvs_max_index], s_tn_r_fp_per[rvs_max_index], r_tp_s_fn[rvs_max_index], r_tp_s_fn_per[rvs_max_index], r_tn_s_fp[rvs_max_index], r_tn_s_fp_per[rvs_max_index]
    rvs_min, rvs_p_min, s_tp_r_fn_min, s_tp_r_fn_per_min, s_tn_r_fp_min, s_tn_r_fp_per_min, r_tp_s_fn_min, r_tp_s_fn_per_min, r_tn_s_fp_min, r_tn_s_fp_per_min = rvs[rvs_min_index], rvs_p[rvs_min_index], s_tp_r_fn[rvs_min_index], s_tp_r_fn_per[rvs_min_index], s_tn_r_fp[rvs_min_index], s_tn_r_fp_per[rvs_min_index], r_tp_s_fn[rvs_min_index], r_tp_s_fn_per[rvs_min_index], r_tn_s_fp[rvs_min_index], r_tn_s_fp_per[rvs_min_index]
    s_tp_r_fn_per, s_tn_r_fp_per, r_tp_s_fn_per, r_tn_s_fp_per = [(i * 100)+1 for i in s_tp_r_fn_per], [(i * 100)+1 for i in s_tn_r_fp_per], [(i * 100)+1 for i in r_tp_s_fn_per], [(i * 100)+1 for i in r_tn_s_fp_per]
    s_tp_r_fn_avg, s_tn_r_fp_avg, r_tp_s_fn_avg, r_tn_s_fp_avg = gmean(s_tp_r_fn_per)-1, gmean(s_tn_r_fp_per)-1, gmean(r_tp_s_fn_per)-1, gmean(r_tn_s_fp_per)-1
    
    f=open("Stats/IF_Inconsistency_Max.csv", "a")
    f.write(filename+',Mod,'+str(mvr_max)+','+str(mvr_p_max)+','+str(rvs_max)+','+str(rvs_p_max)+','+str(mvs_max)+','+str(mvs_p_max) +",")
    f.write(str(r_tp_m_fn_max)+","+str(r_tp_m_fn_per_max)+","+str(r_tn_m_fp_max)+","+str(r_tn_m_fp_per_max)+","+str(m_tp_r_fn_max)+","+str(m_tp_r_fn_per_max)+","+str(m_tn_r_fp_max)+","+str(m_tn_r_fp_per_max)+",")
    f.write(str(s_tp_r_fn_max)+","+str(s_tp_r_fn_per_max)+","+str(s_tn_r_fp_max)+","+str(s_tn_r_fp_per_max)+","+str(r_tp_s_fn_max)+","+str(r_tp_s_fn_per_max)+","+str(r_tn_s_fp_max)+","+str(r_tn_s_fp_per_max)+",")
    f.write(str(s_tp_m_fn_max)+","+str(s_tp_m_fn_per_max)+","+str(s_tn_m_fp_max)+","+str(s_tn_m_fp_per_max)+","+str(m_tp_s_fn_max)+","+str(m_tp_s_fn_per_max)+","+str(m_tn_s_fp_max)+","+str(m_tn_s_fp_per_max))
    f.write("\n")
    f.close()
    f=open("Stats/IF_Inconsistency_Min.csv", "a")
    f.write(filename+',Mod,'+str(mvr_min)+','+str(mvr_p_min)+','+str(rvs_min)+','+str(rvs_p_min)+','+str(mvs_min)+','+str(mvs_p_min) +",")
    f.write(str(r_tp_m_fn_min)+","+str(r_tp_m_fn_per_min)+","+str(r_tn_m_fp_min)+","+str(r_tn_m_fp_per_min)+","+str(m_tp_r_fn_min)+","+str(m_tp_r_fn_per_min)+","+str(m_tn_r_fp_min)+","+str(m_tn_r_fp_per_min)+",")
    f.write(str(s_tp_r_fn_min)+","+str(s_tp_r_fn_per_min)+","+str(s_tn_r_fp_min)+","+str(s_tn_r_fp_per_min)+","+str(r_tp_s_fn_min)+","+str(r_tp_s_fn_per_min)+","+str(r_tn_s_fp_min)+","+str(r_tn_s_fp_per_min)+",")
    f.write(str(s_tp_m_fn_min)+","+str(s_tp_m_fn_per_min)+","+str(s_tn_m_fp_min)+","+str(s_tn_m_fp_per_min)+","+str(m_tp_s_fn_min)+","+str(m_tp_s_fn_per_min)+","+str(m_tn_s_fp_min)+","+str(m_tn_s_fp_per_min))
    f.write("\n")
    f.close()
    f=open("Stats/IF_Inconsistency_Max_Min.csv", "a")
    f.write(filename+',Mod,'+str(mvr_p_min)+','+str(mvr_p_max)+','+str(rvs_p_min)+','+str(rvs_p_max)+','+str(mvs_p_min)+','+str(mvs_p_max) +",")
    f.write(str(r_tp_m_fn_per_min)+","+str(r_tp_m_fn_avg)+","+str(r_tp_m_fn_per_max)+","+str(r_tn_m_fp_per_min)+","+str(r_tn_m_fp_avg)+","+str(r_tn_m_fp_per_max)+","+str(m_tp_r_fn_per_min)+","+str(m_tp_r_fn_avg)+","+str(m_tp_r_fn_per_max)+","+str(m_tn_r_fp_per_min)+","+str(m_tn_r_fp_avg)+","+str(m_tn_r_fp_per_max)+",")
    f.write(str(s_tp_r_fn_per_min)+","+str(s_tp_r_fn_avg)+","+str(s_tp_r_fn_per_max)+","+str(s_tn_r_fp_per_min)+","+str(s_tn_r_fp_avg)+","+str(s_tn_r_fp_per_max)+","+str(r_tp_s_fn_per_min)+","+str(r_tp_s_fn_avg)+","+str(r_tp_s_fn_per_max)+","+str(r_tn_s_fp_per_min)+","+str(r_tn_s_fp_avg)+","+str(r_tn_s_fp_per_max)+",")
    f.write(str(s_tp_m_fn_per_min)+","+str(s_tp_m_fn_avg)+","+str(s_tp_m_fn_per_max)+","+str(s_tn_m_fp_per_min)+","+str(s_tn_m_fp_avg)+","+str(s_tn_m_fp_per_max)+","+str(m_tp_s_fn_per_min)+","+str(m_tp_s_fn_avg)+","+str(m_tp_s_fn_per_max)+","+str(m_tn_s_fp_per_min)+","+str(m_tn_s_fp_avg)+","+str(m_tn_s_fp_per_max))
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
    
    return flipped, flipped/len(labels1), l1_tp_l2_fn, l1_tp_l2_fn_per, l1_tn_l2_fp, l1_tn_l2_fp_per, l2_tp_l1_fn, l2_tp_l1_fn_per, l2_tn_l1_fp, l2_tn_l1_fp_per



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
    if os.path.exists("Stats/IF_Inconsistency_Max_Min.csv"):
        df = pd.read_csv("Stats/IF_Inconsistency_Max_Min.csv")
        done_files = df["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
    
    if os.path.exists("Stats/IF_Inconsistency_Max_Min.csv")==0:
        f=open("Stats/IF_Inconsistency_Max.csv", "w")
        f.write('Filename,Mode,MatvR,MatvR_Percentage,RvSk,RvSk_Percentage,MatvSk,MatvSk_Percentage,')
        f.write("r_tp_m_fn"+","+"r_tp_m_fn_per"+","+"r_tn_m_fp"+","+"r_tn_m_fp_per"+","+"m_tp_r_fn"+","+"m_tp_r_fn_per"+","+"m_tn_r_fp"+","+"m_tn_r_fp_per"+",")
        f.write("s_tp_r_fn"+","+ "s_tp_r_fn_per"+","+ "s_tn_r_fp"+","+ "s_tn_r_fp_per"+","+ "r_tp_s_fn"+","+ "r_tp_s_fn_per"+","+"r_tn_s_fp"+","+"r_tn_s_fp_per"+",")
        f.write("s_tp_m_fn"+","+ "s_tp_m_fn_per"+","+ "s_tn_m_fp"+","+ "s_tn_m_fp_per"+","+ "m_tp_s_fn"+","+ "m_tp_s_fn_per"+","+ "m_tn_s_fp"+","+ "m_tn_s_fp_per")
        f.write("\n")    
        f.close()
        
        f=open("Stats/IF_Inconsistency_Min.csv", "w")
        f.write('Filename,Mode,MatvR,MatvR_Percentage,RvSk,RvSk_Percentage,MatvSk,MatvSk_Percentage,')
        f.write("r_tp_m_fn"+","+"r_tp_m_fn_per"+","+"r_tn_m_fp"+","+"r_tn_m_fp_per"+","+"m_tp_r_fn"+","+"m_tp_r_fn_per"+","+"m_tn_r_fp"+","+"m_tn_r_fp_per"+",")
        f.write("s_tp_r_fn"+","+ "s_tp_r_fn_per"+","+ "s_tn_r_fp"+","+ "s_tn_r_fp_per"+","+ "r_tp_s_fn"+","+ "r_tp_s_fn_per"+","+"r_tn_s_fp"+","+"r_tn_s_fp_per"+",")
        f.write("s_tp_m_fn"+","+ "s_tp_m_fn_per"+","+ "s_tn_m_fp"+","+ "s_tn_m_fp_per"+","+ "m_tp_s_fn"+","+ "m_tp_s_fn_per"+","+ "m_tn_s_fp"+","+ "m_tn_s_fp_per")
        f.write("\n")    
        f.close()
        
        f=open("Stats/IF_Inconsistency_Max_Min.csv", "w")
        f.write('Filename,Mode,MatvR_Min,MatvR_Max,RvSk_Min,RvSk_Max,MatvSk_Min,MatvSk_Max,')
        f.write("r_tp_m_fn_per_min,r_tp_m_fn_avg,r_tp_m_fn_per_max"+","+ "r_tn_m_fp_per_min,r_tn_m_fp_avg,r_tn_m_fp_per_max"+","+ "m_tp_r_fn_per_min,m_tp_r_fn_avg,m_tp_r_fn_per_max"+","+ "m_tn_r_fp_per_min,m_tn_r_fp_avg,m_tn_r_fp_per_max"+",")
        f.write("s_tp_r_fn_per_min,s_tp_r_fn_avg,s_tp_r_fn_per_max"+","+ "s_tn_r_fp_per_min,s_tn_r_fp_avg,s_tn_r_fp_per_max"+","+ "r_tp_s_fn_per_min,r_tp_s_fn_avg,r_tp_s_fn_per_max"+","+ "r_tn_s_fp_per_min,r_tn_s_fp_avg,r_tn_s_fp_per_max"+",")
        f.write("s_tp_m_fn_per_min,s_tp_m_fn_avg,s_tp_m_fn_per_max"+","+ "s_tn_m_fp_per_min,s_tn_m_fp_avg,s_tn_m_fp_per_max"+","+ "m_tp_s_fn_per_min,m_tp_s_fn_avg,m_tp_s_fn_per_max"+","+ "m_tn_s_fp_per_min,m_tn_s_fp_avg,m_tn_s_fp_per_max")
        f.write("\n")    
        f.close()
    
    
    for fname in master_files:
        frr=open("GD_ReRun/RIF.csv", "w")
        frr.write('Filename,ntrees,standardize_data,sample_size,ncols_per_tree\n')
        frr.close()
        
        frr=open("GD_ReRun/MatIF.csv", "w")
        frr.write('Filename,ContaminationFraction,NumLearners,NumObservationsPerLearner\n')
        frr.close()
        
        isolationforest(fname)
        
    eng.quit()
    