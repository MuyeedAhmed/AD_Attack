import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
from time import process_time
import matlab.engine
eng = matlab.engine.start_matlab()
from numpy import unravel_index

datasetFolderDir = 'Dataset/'

def isolationforest(filename, optSettings):
    print(filename)
    folderpath = datasetFolderDir
    
    
    if os.path.exists(folderpath+filename+".mat") == 1:
        if os.path.getsize(folderpath+filename+".mat") > 200000: # 200KB
            return
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
        if os.path.getsize(folderpath+filename+".csv") > 200000: # 200KB
            print("Didn\'t run -> Too large - ", filename)    
            return
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
    
    runs = 50
    
    parameters_default = [0.1, 100, 'auto']
    parameters_fast = [0.1, 50, 'auto']
    ReRun_CSV(filename, parameters_fast)
    # ReRun_CSV(filename, parameters_default)
    # ReRun_CSV(filename, optSettings)
    # ReRun_CSV(filename, parameters_fast)
    
    time = eng.MatIF_Rerun(runs)
    print(time)
    '''
    Default
    '''
    # runIF(filename, X, gt, parameters_default, runs, 'Default', time)

    '''
    Optimal
    '''
    # runIF(filename, X, gt, optSettings, runs, "Optimal", time[0][1])
    '''
    Fast
    '''
    runIF(filename, X, gt, parameters_fast, runs, 'Fast', time)
    

def ReRun_CSV(filename, params):
    frr=open("GD_ReRun/MatIF.csv", "a")
    frr.write(filename+","+str(params[0])+","+str(params[1])+","+str(params[2])+'\n')
    frr.close()

  
def runIF(filename, X, gt, params, runs, mode, t):
    labelFile = filename + "_" + str(params[0]) + "_" + str(params[1]) + "_" + str(params[2])
    
    if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv"):
    
        labels =  pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv", header=None).to_numpy()
    else:
        return
    
    flipped, runNumber50p, avgFlippedPerRun, avgFlippedPerRunPercentage, ti_fo_per_all, to_fi_per_all, ti_fo_per_avg, to_fi_per_avg = drawGraphs(filename, gt, labels, runs, mode)
    
    f=open("Stats/MatIF.csv", "a")
    f.write(filename+','+mode+','+str(t)+','+str(flipped)+','+str(flipped/len(gt))+','+str(runNumber50p)+','+str(avgFlippedPerRun)+','+str(avgFlippedPerRunPercentage))
    f.write(","+str(ti_fo_per_all)+","+str(to_fi_per_all)+","+str(ti_fo_per_avg)+","+str(to_fi_per_avg)+'\n')
    f.close()



def drawGraphs(filename, gt, labels, runs, mode):
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
    
    if flipped == 0:
        return flipped, -1, 0, 0, 0, 0, 0, 0
    
    inlier = len(gt) - sum(gt)
    outlier = sum(gt)
    ti_fo = 0
    to_fi = 0
    for i in range(len(gt)):
        if flippable[i] == True:
            if gt[i] == 0:
                ti_fo += 1
            else:
                to_fi += 1
    ti_fo_per_all = ti_fo/inlier
    to_fi_per_all = to_fi/outlier
    
    # f = plt.figure()
    # avgs = np.array(avgs)
    # sns.displot(avgs, kde=True, stat='count')
    # # plt.savefig("FlipFig/"+filename+"_"+mode+"_NormDist.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()

    '''
    50th percetile
    '''
    variables_iter = []
    for h in range(1,runs):
        variables = 0
        for i in range(len(labels[0])):
            s = 0
            for j in range(h):
                s+=labels[j][i]
            avg = s/h
            if avg != 0 and avg != 1:
                variables += 1
        variables_iter.append(variables)
        
    probability = [(x / flipped)*100 for x in variables_iter]
    for i in range(runs-1):
        if probability[i] < 0.5:
            continue
        else:
            runNumber50p = i
            break

    # g = plt.figure
    # plt.plot(variables_iter)
    # plt.axhline(y=0.5*flipped, color='r', linestyle='-')
    # plt.savefig("FlipFig/MatIF/"+filename+"_MatIF_"+mode+"_Count.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    
    # g = plt.figure
    # plt.plot(probability)
    # plt.xlabel("Run")
    # plt.ylabel("Previously Uniscovered Flipped Points (%)")
    # # plt.axhline(y=0.5*flipped, color='r', linestyle='-')
    # plt.savefig("FlipFig/MatIF/"+filename+"_MatIF_"+mode+"_Count_percentage.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    
    '''
    Flipped in a single run (mean)
    '''
    ti_fo_avg = []
    to_fi_avg = []
    flippedIn2Runs = []
    for i in range(runs):
        for j in range(i+1,runs):
            ti_fo = 0
            to_fi = 0
            norms = 0
            outliers = 0
            variables = 0
            for n in range(len(gt)):
                
                if labels[i][n] != labels[j][n]:
                    variables += 1
                    if gt[n] == 0:
                       ti_fo += 1
                    else:
                       to_fi += 1
            ti_fo_avg.append(ti_fo)
            to_fi_avg.append(to_fi)
            flippedIn2Runs.append(variables)
    
    
    ti_fo_per_avg = np.mean(ti_fo_avg)/inlier
    to_fi_per_avg = np.mean(to_fi_avg)/outlier
    avgFlippedPerRun = sum(flippedIn2Runs)/len(flippedIn2Runs)
    # print(mode, "- flipped in a single run ", avgFlippedPerRun, " - ", avgFlippedPerRun/len(labels[0]))

    
    # '''
    # Flipped in a single run sorted graph
    # '''
    
    # # f_g = []
    
    # # print("----")
    # # while sum(flippable) != 0:
    # #     flippedIn2Runs = np.array([[0]*runs for i in range(runs)])
    # #     print(sum(flippable), end='')
    # #     for i in range(runs):
    # #         for j in range(i+1,runs):
    # #             variables = 0
    # #             for n in range(len(flippable)):
    # #                 if flippable[n]:
    # #                     s = labels[i][n] + labels[j][n]                
    # #                     avg = s/2
    # #                     if avg != 0 and avg != 1:
    # #                         variables += 1
    # #             flippedIn2Runs[i][j] = variables
    # #     maxInd = unravel_index(flippedIn2Runs.argmax(), flippedIn2Runs.shape)
    # #     for n in range(len(flippable)):
    # #         if flippable[n]:
    # #             s = labels[maxInd[0]][n] + labels[maxInd[1]][n]                
    # #             avg = s/2
    # #             if avg != 0 and avg != 1:
    # #                 flippable[n] = False
    # #     f_g.append(flippedIn2Runs[maxInd[0]][maxInd[1]])
    # # print("-----")
    # # print(f_g)
    
    # # f_g_percentage = [(x/flipped)*100 for x in f_g]
    
    # # g = plt.figure
    # # plt.plot(f_g)
    # # plt.xlabel("Run")
    # # plt.ylabel("Discovered Flipped Points")
    # # plt.savefig("FlipFig/"+filename+"_MatIF_"+mode+"_FlippableVRun.pdf", bbox_inches="tight", pad_inches=0)
    # # plt.show()

    # # g = plt.figure
    # # plt.plot(f_g_percentage)
    # # plt.xlabel("Run")
    # # plt.ylabel("Discovered Flipped Points Percentage")
    # # plt.savefig("FlipFig/"+filename+"_MatIF_"+mode+"_FlippableVRun_Percentage.pdf", bbox_inches="tight", pad_inches=0)
    # # plt.show()
    # '''
    # Flipped in a single run unsorted
    # '''
    
    # # flippedIn2Runs = []
    
    # # for i in range(runs-1):
    # #     variables = 0
    # #     for n in range(len(labels[0])):
    # #         s = labels[i][n] + labels[i+1][n]                
    # #         avg = s/2
    # #         if avg != 0 and avg != 1:
    # #             variables += 1
    # #     flippedIn2Runs.append(variables)
    
    # # g = plt.figure
    # # plt.plot(flippedIn2Runs)
    # # plt.xlabel("Run")
    # # plt.ylabel("Points Flipped")
    # # plt.savefig("FlipFig/MatIF/"+filename+"_MatIF_"+mode+"_FlippableVRun_rand.pdf", bbox_inches="tight", pad_inches=0)
    # # plt.show()
    
    # '''
    # Flipped in a single run unsorted barchart
    # '''
    
    # # flippedIn2Runs = [0]
    # flippedIn2Runs = np.array([[0]*2 for i in range(runs-1)])
    # for i in range(runs-1):
    #     variables = 0
    #     tp_fn = 0
    #     tn_fp = 0
    #     for n in range(len(labels[0])):
    #         s = labels[i][n] + labels[i+1][n]                
    #         avg = s/2
    #         if avg != 0 and avg != 1:
    #             if labels[i][n] == 0:
    #                 tp_fn += 1
    #             else:
    #                 tn_fp += 1
                
    #     flippedIn2Runs[i][0] = tp_fn
    #     flippedIn2Runs[i][1] = tn_fp
        
    
    # bar = flippedIn2Runs.reshape((2, runs-1))
    # # create plot
    # fig, ax = plt.subplots()
    # ax.grid(False)
    # index = np.arange(runs-1)
    # bar_width = 0.5
    # # opacity = 0
    
    # rects1 = plt.bar(index, bar[0], bar_width,
    # # alpha=opacity,
    # # color='b',
    # label='TP to FN')
    
    # rects2 = plt.bar(index + bar_width, bar[1], bar_width,
    # # alpha=opacity,
    # # color='g',
    # label='TN to FP')
    
    # plt.xlabel('Runs')
    # plt.ylabel('Flipped')
    # plt.legend()
    # plt.savefig("FlipFig/MatIF/"+filename+"_MatIF_"+mode+"_FlippableVRun_rand_pn_bar.pdf", bbox_inches="tight", pad_inches=0)

    # plt.tight_layout()
    # plt.show()
    
    # '''
    # Flipped in a single run sorted - asc
    # '''
    # f_g = [0]
    # doneRun = []
    # flippable_temp = flippable.copy()
    # print("----")
    # while sum(flippable_temp) != 0:
    #     flippedIn2Runs = np.array([[99999]*runs for i in range(runs)])
    #     print(sum(flippable_temp), end=' ')
    #     for i in range(runs):
    #         for j in range(i+1,runs):
    #             if i in doneRun or j in doneRun:
    #                 continue
    #             variables = 0
    #             for n in range(len(flippable_temp)):
    #                 if flippable_temp[n]:
    #                     s = labels[i][n] + labels[j][n]                
    #                     avg = s/2
    #                     if avg != 0 and avg != 1:
    #                         variables += 1
    #             flippedIn2Runs[i][j] = variables
        
    #     minInd = unravel_index(flippedIn2Runs.argmin(), flippedIn2Runs.shape)
    #     doneRun.append(minInd[0])
    #     # if flippedIn2Runs[minInd[0]][minInd[1]] == 0:
    #     #     continue
    #     for n in range(len(flippable_temp)):
    #         if flippable_temp[n]:
    #             s = labels[minInd[0]][n] + labels[minInd[1]][n]                
    #             avg = s/2
    #             if avg != 0 and avg != 1:
    #                 flippable_temp[n] = False
    #     f_g.append(f_g[-1] + flippedIn2Runs[minInd[0]][minInd[1]])
    # # print("-----")
    # # print(f_g)
    # # print(doneRun)

    # f_g_percentage = [(x/flipped)*100 for x in f_g]
    
    # g = plt.figure
    # plt.plot(f_g)
    # plt.xlabel("Run")
    # plt.ylabel("Previously Uniscovered Flipped Points")
    # plt.savefig("FlipFig/MatIF/"+filename+"_MatIF_"+mode+"_FlippableVRun_asc.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()

    # g = plt.figure
    # plt.plot(f_g_percentage)
    # plt.xlabel("Run")
    # plt.ylabel("Previously Uniscovered Flipped Points (%)")
    # plt.savefig("FlipFig/MatIF/"+filename+"_MatIF_"+mode+"_FlippableVRun_Percentage_asc.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    
    
    # '''
    # Flipped in a single run sorted - dsc
    # '''
    # f_g = [0]
    # doneRun = []
    # flippable_temp = flippable.copy()
    # print("\n----")
    # # while sum(flippable_temp) != 0:
    # for _ in range(runs-1):
    #     flippedIn2Runs = np.array([[0]*runs for i in range(runs)])
    #     print(sum(flippable_temp), end=' ')
    #     for i in range(runs):
    #         for j in range(i+1,runs):
    #             if i in doneRun or j in doneRun:
    #                 continue
    #             variables = 0
    #             for n in range(len(flippable_temp)):
    #                 if flippable_temp[n] and labels[i][n] != labels[j][n]:
    #                     variables += 1
    #             flippedIn2Runs[i][j] = variables
        
    #     maxInd = unravel_index(flippedIn2Runs.argmax(), flippedIn2Runs.shape)
    #     doneRun.append(maxInd[0])
    #     # if flippedIn2Runs[maxInd[0]][maxInd[1]] == 0:
    #     #     continue
    #     for n in range(len(flippable_temp)):
    #         if flippable_temp[n]:
    #             if labels[maxInd[0]][n] != labels[maxInd[1]][n]:
    #                 flippable_temp[n] = False
    #     f_g.append(f_g[-1] + flippedIn2Runs[maxInd[0]][maxInd[1]])
    # print("\n-----")
    # print(f_g)
    # print(doneRun)

    # f_g_percentage = [(x/flipped)*100 for x in f_g]
    
    # g = plt.figure
    # plt.plot(f_g)
    # plt.xlabel("Run")
    # plt.ylabel("Previously Uniscovered Flipped Points")
    # plt.savefig("FlipFig/MatIF/"+filename+"_MatIF_"+mode+"_FlippableVRun_dsc.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()

    # g = plt.figure
    # plt.plot(f_g_percentage)
    # plt.xlabel("Run")
    # plt.ylabel("Previously Uniscovered Flipped Points (%)")
    # plt.savefig("FlipFig/MatIF/"+filename+"_MatIF_"+mode+"_FlippableVRun_Percentage_dsc.pdf", bbox_inches="tight", pad_inches=0)
    # plt.show()
    
    return flipped, runNumber50p, avgFlippedPerRun, avgFlippedPerRun/len(labels[0]), ti_fo_per_all, to_fi_per_all, ti_fo_per_avg, to_fi_per_avg
    
    


    
if __name__ == '__main__':
    folderpath = datasetFolderDir
    master_files1 = glob.glob(folderpath+"*.mat")
    master_files2 = glob.glob(folderpath+"*.csv")
    master_files = master_files1 + master_files2
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    if os.path.exists("Stats/MatIF.csv"):
        df = pd.read_csv("Stats/MatIF.csv")
        done_files = df["Filename"].to_numpy()
        master_files = [item for item in master_files if item not in done_files]
    master_files.sort()
    
    optimalSettingsUni = pd.read_csv("OptimalSettings/MatIF_Uni.csv")
    
    if os.path.exists("Stats/MatIF.csv")==0:
        f=open("Stats/MatIF.csv", "w")
        f.write('Filename,Mode,AvgTimeElapsed,Flipped,FlippedPercentage,RunNumber50p,AvgFlippedPerRun,AvgFlippedPerRunPercentage,ti_fo_per_all,to_fi_per_all,ti_fo_per_avg,to_fi_per_avg\n')
        f.close()
    
    for fname in master_files:
        frr=open("GD_ReRun/MatIF.csv", "w")
        frr.write('Filename,ContaminationFraction,NumLearners,NumObservationsPerLearner\n')
        frr.close()
        # if fname == 'smapbase':
        #     continue
        # optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == fname].to_numpy()[0][1:]
        isolationforest(fname, [])
    eng.quit()
    
    # optSettings = optimalSettingsUni[optimalSettingsUni['Filename'] == 'breastw'].to_numpy()[0][1:]
        
    # isolationforest('breastw', optSettings)
    # isolationforest('breastw')
    # isolationforest("arsenic-female-lung")
    