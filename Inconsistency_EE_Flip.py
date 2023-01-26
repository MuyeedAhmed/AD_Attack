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
from sklearn.covariance import EllipticEnvelope

from sklearn.metrics import f1_score
from scipy.stats import gmean

import subprocess
import matlab.engine
eng = matlab.engine.start_matlab()

datasetFolderDir = 'Dataset/'


withGT = True

def ee(filename):
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
    
    
    param_mat_default = ["fmcd", 0.5, 500, 1, 2, "tauscale", "rfch", 10, "classical"]
    param_mat_mod = ["fmcd", if_cont, 500, 1, 2, "tauscale", "rfch", 10, "classical"]
    
    param_sk_mod = [True, False, None, if_cont]
    
    frr=open("GD_ReRun/MatEE.csv", "a")
    frr.write(filename+","+str(param_mat_default[0])+","+str(param_mat_default[1])+","+str(param_mat_default[2])+","+str(param_mat_default[3])+","+str(param_mat_default[4])+","+str(param_mat_default[5])+","+str(param_mat_default[6])+","+str(param_mat_default[7])+","+str(param_mat_default[8])+'\n')
    frr.write(filename+","+str(param_mat_mod[0])+","+str(param_mat_mod[1])+","+str(param_mat_mod[2])+","+str(param_mat_mod[3])+","+str(param_mat_mod[4])+","+str(param_mat_mod[5])+","+str(param_mat_mod[6])+","+str(param_mat_mod[7])+","+str(param_mat_mod[8])+'\n')
    frr.close()
    
    
    ## Default
    labels_sk = []
    for i in range(runs):
        clustering = EllipticEnvelope().fit(X)
        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels_sk.append(l)
        
    labelFile_mat_default = filename + "_" + str(param_mat_default[0]) + "_" + str(param_mat_default[1]) + "_" + str(param_mat_default[2]) + "_" + str(param_mat_default[3]) + "_" + str(param_mat_default[4]) + "_" + str(param_mat_default[5]) + "_" + str(param_mat_default[6]) + "_" + str(param_mat_default[7]) + "_" + str(param_mat_default[8])
    
    
    if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat_default+".csv") == 0:        
        try:
            eng.MatEE_Rerun(runs)
            if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat_default+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
    
    labels_mat = pd.read_csv("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat_default+".csv", header=None).to_numpy()
    
    
    mvs, mvs_p, s_tp_m_fn, s_tp_m_fn_per, s_tn_m_fp, s_tn_m_fp_per, m_tp_s_fn, m_tp_s_fn_per, m_tn_s_fp, m_tn_s_fp_per = [], [], [], [], [], [], [], [], [], []
    
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
            

    
    
    f=open("Stats/EE_Inconsistency_Max_Min.csv", "a")
    f.write(filename+',Default,'+str(mvs_p_min)+','+str(mvs_p_max) +",")
    f.write(str(s_tp_m_fn_per_min)+","+str(s_tp_m_fn_avg)+","+str(s_tp_m_fn_per_max)+","+str(s_tn_m_fp_per_min)+","+str(s_tn_m_fp_avg)+","+str(s_tn_m_fp_per_max)+","+str(m_tp_s_fn_per_min)+","+str(m_tp_s_fn_avg)+","+str(m_tp_s_fn_per_max)+","+str(m_tn_s_fp_per_min)+","+str(m_tn_s_fp_avg)+","+str(m_tn_s_fp_per_max))
    f.write("\n")
    f.close()
    
    ## Mod
    labels_sk = []
    for i in range(runs):
        clustering = EllipticEnvelope(store_precision=param_sk_mod[0], assume_centered=param_sk_mod[1], 
                                     support_fraction=param_sk_mod[2], contamination=param_sk_mod[3]).fit(X)

        l = clustering.predict(X)
        l = [0 if x == 1 else 1 for x in l]
        labels_sk.append(l)
    
    labelFile_mat_mod = filename + "_" + str(param_mat_mod[0]) + "_" + str(param_mat_mod[1]) + "_" + str(param_mat_mod[2])
    
    
    if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat_mod+".csv") == 0:        
        try:
            eng.MatEE_Rerun(runs)
            if os.path.exists("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat_mod+".csv") == 0:      
                print("\nFaild to run Matlab Engine from Python.\n")
                exit(0)
        except:
            print("\nFaild to run Matlab Engine from Python.\n")
            exit(0) 
            
    labels_mat = pd.read_csv("Labels/EE_Matlab/Labels_Mat_EE_"+labelFile_mat_mod+".csv", header=None).to_numpy()
    
    
    mvs, mvs_p, s_tp_m_fn, s_tp_m_fn_per, s_tn_m_fp, s_tn_m_fp_per, m_tp_s_fn, m_tp_s_fn_per, m_tn_s_fp, m_tn_s_fp_per = [], [], [], [], [], [], [], [], [], []
    
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
            

    f=open("Stats/EE_Inconsistency_Max_Min.csv", "a")
    f.write(filename+',Mod,'+str(mvs_p_min)+','+str(mvs_p_max) +",")
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
    if os.path.exists("Stats/EE_Inconsistency_Max_Min.csv"):
        df = pd.read_csv("Stats/EE_Inconsistency_Max_Min.csv")
        done_files = df["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
    
    if os.path.exists("Stats/EE_Inconsistency_Max_Min.csv")==0:
        f=open("Stats/EE_Inconsistency_Max_Min.csv", "w")
        f.write('Filename,Mode,MatvSk_Min,MatvSk_Max,')
        f.write("s_tp_m_fn_per_min,s_tp_m_fn_avg,s_tp_m_fn_per_max"+","+ "s_tn_m_fp_per_min,s_tn_m_fp_avg,s_tn_m_fp_per_max"+","+ "m_tp_s_fn_per_min,m_tp_s_fn_avg,m_tp_s_fn_per_max"+","+ "m_tn_s_fp_per_min,m_tn_s_fp_avg,m_tn_s_fp_per_max")
        f.write("\n")    
        f.close()
    
    
    for fname in master_files:
        frr=open("GD_ReRun/MatEE.csv", "w")
        frr.write('Filename,Method,OutlierFraction,NumTrials,BiasCorrection,NumOGKIterations,UnivariateEstimator,ReweightingMethod,NumConcentrationSteps,StartMethod\n')
        frr.close()
        
        ee(fname)
        
    eng.quit()
    