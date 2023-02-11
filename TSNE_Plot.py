import os
import glob
import pandas as pd
import mat73
from scipy.io import loadmat
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from sklearn.manifold import TSNE

import numpy as np
  
def calculateAccuracy(filename, Algo):
    folderpath = 'Dataset/'
    print(filename)
    
    if os.path.exists(folderpath+filename+".mat") == 1:
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
    
    
    X_embedded = TSNE(n_components=2, learning_rate='auto',perplexity = 20,
                       init='random', random_state=(0)).fit_transform(X)
    
    X_embedded = pd.DataFrame(X_embedded)
    
    # drawPlot(filename, Algo, "Matlab", "R", X_embedded, gt)
    # drawPlot(filename, Algo, "Sklearn", "R", X_embedded, gt)
    drawPlot(filename, Algo, "Sklearn", "Matlab", X_embedded, gt, X)


def runAlgo(filename, X):
    clustering = IsolationForest(random_state=42).fit(X)
    l1 = clustering.predict(X)
    l1 = [0 if x == 1 else 1 for x in l1]
    
    clustering = IsolationForest(random_state=400).fit(X)
    l2 = clustering.predict(X)
    l2 = [0 if x == 1 else 1 for x in l2]

    clustering = IsolationForest(n_estimators=50, max_samples=64, random_state=400).fit(X)
    l3 = clustering.predict(X)
    l3 = [0 if x == 1 else 1 for x in l3]
    
    params = [0.005, 100, 'auto']
    frr=open("GD_ReRun/MatIF.csv", "w")
    frr.write('Filename,ContaminationFraction,NumLearners,NumObservationsPerLearner\n')
    frr.write(filename+","+str(params[0])+","+str(params[1])+","+str(params[2])+'\n')
    frr.close()
    
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.MatIF_Rerun(1)
    eng.quit()
    
    labelFile = filename + "_" + str(params[0]) + "_" + str(params[1]) + "_" + str(params[2])
    
    if os.path.exists("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv"):
        labels =  pd.read_csv("Labels/IF_Matlab/Labels_Mat_IF_"+labelFile+".csv", header=None).to_numpy()
    else:
        return
    
    # return np.array(l1), np.array(l2), np.array(labels[0])
    return np.array(l1), np.array(l2), np.array(l3), np.array(labels[0])
    
def drawPlot(filename, Algo, tool1, tool2, x, y, X):
    l1, l2, l3, l4 = runAlgo(filename, X)
    
    # cat = [0] * len(y)
    # for i in range(len(y)):
    #     if l1[i] != l2[i]:
    #         if l1[i] == 0:
    #             cat[i] = 2
    #         if l2[i] == 0:
    #             cat[i] = 3
    #     else:
    #         cat[i] = 0
    # cat = np.array(cat)
    
    # fig = plt.figure()
    # plt.rcParams['figure.figsize'] = [10, 10]
    
    
    # plt.title(filename, fontsize=15)
    
    # indicesToKeep = cat == 0
    # plt0 = plt.scatter(x.loc[indicesToKeep,1]
    #  ,x.loc[indicesToKeep,0]
    #  ,s = 25, color='grey')
    
    # # indicesToKeep = cat == 1
    # # plt1 = plt.scatter(x.loc[indicesToKeep,1]
    # #  ,x.loc[indicesToKeep,0]
    # #  ,s = 50, color='green')
    
    # indicesToKeep = cat == 2
    # plt2 = plt.scatter(x.loc[indicesToKeep,1]
    #  ,x.loc[indicesToKeep,0]
    #  ,s = 50, color='blue')
    
    # indicesToKeep = cat == 3
    # plt3 = plt.scatter(x.loc[indicesToKeep,1]
    #  ,x.loc[indicesToKeep,0]
    #  ,s = 50, color='red')
    
    
    
    # plt.legend([plt0, plt2, plt3],["Both Toolkits Predicted Same Output", "Only "+tool2+" Predicted as Anomaly", "Only "+tool1+" Predicted as Anomaly"], prop={'size': 15})
    # # plt.legend([plt0, plt1, plt2, plt3],["Both Toolkits Predicted as Normal", "Both Toolkits Predicted as Anomaly", "Only "+tool2+" Predicted as Anomaly", "Only "+tool1+" Predicted as Anomaly"], prop={'size': 15})
    # plt.grid(False)
    # plt.xticks(fontsize = 25)
    # plt.yticks(fontsize = 25)
    
    # # plt.savefig('Fig_InterTool_tsne/'+tool1+'_'+tool2+"_"+Algo+'_'+filename+'_default_Box.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    # plt.show()

    
    plt.rcParams['figure.figsize'] = [5, 5]
    
    fig = plt.figure()
    indicesToKeep = (y == 0)
    plt0 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 50, color='green')
    
    indicesToKeep = y == 1
    plt2 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 75, marker='X', color='red')
    plt.grid(False)
    # plt.xticks([])
    # plt.yticks([])
    plt.ylim(0, 25)
    plt.xlim(0, 10)
    
    plt.show()

    '''
    Original - L1 Run1
    '''
    fig = plt.figure()
    # plt.rcParams['figure.figsize'] = [5, 10]
    
    # plt.title(filename, fontsize=15)
    
    indicesToKeep = (l1 == 0)
    plt0 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 50, color='green')
    
    indicesToKeep = l1 == 1
    plt2 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 75, marker='X', color='red')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(5, 11)
    plt.xlim(1, 7.5)
    
    plt.annotate("A: Outlier",
            xy=(6.92, 10.6), xycoords='data',
            xytext=(6.92, 9.4), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='red'
            )
    plt.annotate("B: Outlier",
            xy=(2.75, 6.9), xycoords='data',
            xytext=(3.5, 6.9), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='red'
            )
    plt.annotate("C: Outlier",
            xy=(2.73, 7.4), xycoords='data',
            xytext=(2.73, 9), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='red'
            )
    plt.annotate("D: Inlier",
            xy=(2.2, 6.1), xycoords='data',
            xytext=(2.2, 8), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='green'
            )
    plt.annotate("E: Outlier",
            xy=(1.30, 6.4), xycoords='data',
            xytext=(1.45, 7.5), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='red'
            )
    # plt.text(10, 7.1, "A: Outlier", fontsize=15)
    # plt.text(8.5, 3.0, "B: Outlier", fontsize=15)
    # plt.text(8.5, 2.6, "C: Outlier", fontsize=15)
    # plt.text(8.5, 2.2, "D: Inlier", fontsize=15)
    # plt.text(7.0, 1.2, "E: Outlier", fontsize=15)
    plt.savefig('Fig/'+filename+'_SkIF_R1_Box.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    plt.show()

    '''
    Restart - L2
    '''
    fig = plt.figure()
    # plt.rcParams['figure.figsize'] = [5, 10]
    indicesToKeep = l2 == 0
    plt0 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 50, color='green')
    
    indicesToKeep = l2 == 1
    plt2 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 75, marker='X', color='red')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(5, 11)
    plt.xlim(1, 7.5)
    plt.annotate("A: Inlier",
            xy=(6.92, 10.6), xycoords='data',
            xytext=(6.92, 9.4), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='green'
            )
    plt.annotate("B: Inlier",
            xy=(2.75, 6.9), xycoords='data',
            xytext=(3.5, 6.9), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='green'
            )
    plt.annotate("C: Outlier",
            xy=(2.73, 7.4), xycoords='data',
            xytext=(2.73, 9), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='red'
            )
    plt.annotate("D: Inlier",
            xy=(2.2, 6.1), xycoords='data',
            xytext=(2.2, 8), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='green'
            )
    plt.annotate("E: Inlier",
            xy=(1.30, 6.4), xycoords='data',
            xytext=(1.45, 7.5), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='green'
            )
    plt.savefig('Fig/'+filename+'_SkIF_R2_Box.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    plt.show()
    
    
    '''
    Fast - L3
    '''
    fig = plt.figure()
    # plt.rcParams['figure.figsize'] = [5, 10]
    indicesToKeep = l3 == 0
    plt0 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 50, color='green')
    
    indicesToKeep = l3 == 1
    plt2 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 75, marker='X', color='red')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(5, 11)
    plt.xlim(1, 7.5)
    plt.annotate("A: Outlier",
            xy=(6.92, 10.6), xycoords='data',
            xytext=(6.92, 9.4), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='red'
            )
    plt.annotate("B: Outlier",
            xy=(2.75, 6.9), xycoords='data',
            xytext=(3.5, 6.9), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='red'
            )
    plt.annotate("C: Outlier",
            xy=(2.73, 7.4), xycoords='data',
            xytext=(2.73, 9), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='red'
            )
    plt.annotate("D: Outlier",
            xy=(2.2, 6.1), xycoords='data',
            xytext=(2.2, 8), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='red'
            )
    plt.annotate("E: Inlier",
            xy=(1.30, 6.4), xycoords='data',
            xytext=(1.45, 7.5), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='green'
            )
    plt.savefig('Fig/'+filename+'_SkIF_Fast_Box.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    plt.show()
    
    '''
    Incon - L4
    '''
    fig = plt.figure()
    indicesToKeep = l4 == 0
    plt0 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 50, color='green')
    
    indicesToKeep = l4 == 1
    plt2 = plt.scatter(x.loc[indicesToKeep,1]
     ,x.loc[indicesToKeep,0]
     ,s = 75, marker='X', color='red')
    # plt.grid(True)
    plt.xticks([])
    plt.yticks([])
    plt.ylim(5, 11)
    plt.xlim(1, 7.5)
    plt.annotate("A: Outlier",
            xy=(6.85, 10.7), xycoords='data',
            xytext=(4.8, 10.7), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='red'
            )
    plt.annotate("B: Inlier",
            xy=(2.75, 6.93), xycoords='data',
            xytext=(4.5, 6.93), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='green'
            )
    plt.annotate("C: Outlier",
            xy=(2.73, 7.3), xycoords='data',
            xytext=(2.73, 9), textcoords='data',
            arrowprops=dict(color='Red', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='top', size = 15,color='red'
            )
    plt.annotate("D: Inlier",
            xy=(2.2, 6.05), xycoords='data',
            xytext=(4.2, 6.05), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center', size = 15,color='green'
            )
    # plt.annotate("E: Inlier",
    #         xy=(1.30, 6.4), xycoords='data',
    #         xytext=(1.75, 7.5), textcoords='data',
    #         arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
    #         horizontalalignment='left', verticalalignment='top', size = 15,color='green'
    #         )
    # plt.annotate("E: Inlier",
    #         xy=(1.30, 6.4), xycoords='data',
    #         xytext=(1.75, 7.5), textcoords='data',
    #         arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
    #         horizontalalignment='right', verticalalignment='bottom', size = 15,color='green'
    #         )
    plt.annotate("E: Inlier",
            xy=(1.30, 6.4), xycoords='data',
            xytext=(1.75, 7.5), textcoords='data',
            arrowprops=dict(color='Green', shrink=0.1, linewidth=0.01),
            horizontalalignment='center', verticalalignment='center_baseline', size = 15,color='green'
            )
    plt.savefig('Fig/'+filename+'_MatIF_Def_Box.pdf', dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    plt.show()

if __name__ == '__main__':
    folderpath = 'Dataset/'
    master_files = glob.glob(folderpath+"*.mat")
    
    for i in range(len(master_files)):
        master_files[i] = master_files[i].split("/")[-1].split(".")[0]
    master_files.sort()

    # Algos = ["EE", "IF", "LF", "OCSVM"]
    # for Algo in Algos:    
    #     for FileNumber in range(len(master_files)):
    #         calculateAccuracy(master_files[FileNumber], Algo)
    # calculateAccuracy("arsenic-male-lung", "IF")
        
    calculateAccuracy("spambase", "IF")
        



