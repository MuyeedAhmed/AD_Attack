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

class draw:
    def __init__(self, filename, tool, algo, gt, labels, run_count, mode):
        """Init vars"""
        self.filename = filename
        self.tool = tool
        self.algoName = algo
        self.gt = gt
        self.labels = labels
        self.run_count= run_count
        self.mode = mode
        
        self.savePath = "FlipFig/"+self.tool+self.algoName+"/"+self.filename+"_"+self.tool+"_"+self.algoName+"_"+self.mode
        """Plot vars"""
        self.figsize_x = 10
        self.figsize_y = 6
        self.labelSize = 32
        self.tickSize = 32
        """Calculated vars"""
        # # From Flip Summary
        self.flipped = -1
        self.flippable = [False]*len(self.labels[0])
        self.inlier = -1
        self.outlier = -1
        self.ti_fo_per_all = -1
        self.to_fi_per_all = -1
        
        # # From  avgFlippedInSingleRun
        self.ti_fo_per_avg = -1
        self.to_fi_per_avg = -1
        self.avgFlippedPerRun = -1
        
    '''Flip Summary'''  
    def getFlipSummary(self, save=False):
        norms = 0
        outliers = 0
        avgs = []
        
        for i in range(len(self.labels[0])):
            s = 0
            for j in range(self.run_count):
                s+=self.labels[j][i]
            avg = s/self.run_count
            avgs.append(avg)
            if avg == 0:
                norms += 1
            elif avg == 1:
                outliers += 1
            else:
                self.flipped += 1
                self.flippable[i] = True
        # print("*****")
        # print(norms, outliers, self.flipped)
        # if self.flipped == 0:
        #     return self.flipped, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
        self.inlier = len(self.gt) - sum(self.gt)
        self.outlier = sum(self.gt)
        ti_fo = 0
        to_fi = 0
        for i in range(len(self.gt)):
            if self.flippable[i] == True:
                if self.gt[i] == 0:
                    ti_fo += 1
                else:
                    to_fi += 1
        self.ti_fo_per_all = ti_fo/self.inlier
        self.to_fi_per_all = to_fi/self.outlier
        
        
        f = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        avgs = np.array(avgs)
        sns.displot(avgs, kde=True, stat='count')
        if save:
            plt.savefig(self.savePath+"_NormDist.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()
    
    ''' 50th percetile '''
    def get50thPercentile(self, save=False):
        if self.flipped == -1:
            self.getFlipSummary()
        variables_iter = []
        for h in range(1,self.run_count):
            variables = 0
            for i in range(len(self.labels[0])):
                s = 0
                for j in range(h):
                    s+=self.labels[j][i]
                avg = s/h
                if avg != 0 and avg != 1:
                    variables += 1
            variables_iter.append(variables)
            
        probability = [(x / (self.flipped+1))*100 for x in variables_iter]
        for i in range(self.run_count-1):
            if probability[i] < 0.5:
                continue
            else:
                runNumber50p = i
                break
            
        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        plt.plot(probability)
        plt.xticks(fontsize=self.tickSize)
        plt.yticks(fontsize=self.tickSize)
        plt.xlabel("Run",fontsize=self.labelSize)
        plt.ylabel("Previously Undiscovered \nFlipped Points (%)",fontsize=self.labelSize)
        # plt.axhline(y=0.5*flipped, color='r', linestyle='-')
        if save:
            plt.savefig(self.savePath+"_Count_percentage.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()
        return runNumber50p, probability
    
    '''Flipped in a single run (mean)'''
    def avgFlippedInSingleRun(self):
        if self.inlier == -1:
            self.getFlipSummary()
        ti_fo_avg = []
        to_fi_avg = []
        flippedIn2Runs = []
        for i in range(self.run_count):
            for j in range(i+1,self.run_count):
                ti_fo = 0
                to_fi = 0
                norms = 0
                outliers = 0
                variables = 0
                for n in range(len(self.gt)):
                    
                    if self.labels[i][n] != self.labels[j][n]:
                        variables += 1
                        if self.gt[n] == 0:
                           ti_fo += 1
                        else:
                           to_fi += 1
                ti_fo_avg.append(ti_fo)
                to_fi_avg.append(to_fi)
                flippedIn2Runs.append(variables)
        
        
        self.ti_fo_per_avg = np.mean(ti_fo_avg)/self.inlier
        self.to_fi_per_avg = np.mean(to_fi_avg)/self.outlier
        self.avgFlippedPerRun = sum(flippedIn2Runs)/len(flippedIn2Runs)
        
        return ti_fo, to_fi, np.mean(ti_fo_avg), np.mean(to_fi_avg)
    
    ''' Flipped in a single run sorted graph '''
    def flippedInSingleRunSorted(self, save=False):
        if self.flipped == -1:
            self.getFlipSummary()
        f_g = []
        while sum(self.flippable) != 0:
            flippedIn2Runs = np.array([[0]*self.run_count for i in range(self.run_count)])
            print(sum(self.flippable), end=' ')
            for i in range(self.run_count):
                for j in range(i+1,self.run_count):
                    variables = 0
                    for n in range(len(self.flippable)):
                        if self.flippable[n]:
                            s = self.labels[i][n] + self.labels[j][n]                
                            avg = s/2
                            if avg != 0 and avg != 1:
                                variables += 1
                    flippedIn2Runs[i][j] = variables
            maxInd = unravel_index(flippedIn2Runs.argmax(), flippedIn2Runs.shape)
            for n in range(len(self.flippable)):
                if self.flippable[n]:
                    s = self.labels[maxInd[0]][n] + self.labels[maxInd[1]][n]                
                    avg = s/2
                    if avg != 0 and avg != 1:
                        self.flippable[n] = False
            f_g.append(flippedIn2Runs[maxInd[0]][maxInd[1]])

        
        f_g_percentage = [(x/self.flipped)*100 for x in f_g]
        
        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        plt.plot(f_g)
        plt.xlabel("Run")
        plt.ylabel("Discovered Flipped Points")
        if save:
            plt.savefig(self.savePath+"_FlippableVRun.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()

        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        plt.plot(f_g_percentage)
        plt.xlabel("Run")
        plt.ylabel("Discovered Flipped Points Percentage")
        if save:
            plt.savefig(self.savePath+"_FlippableVRun_Percentage.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()
    
    ''' Flipped in a single run unsorted '''
    def flippedInSingleRunUnsorted(self, save=False):
        flippedIn2Runs = []
        
        for i in range(self.run_count-1):
            variables = 0
            for n in range(len(self.labels[0])):
                s = self.labels[i][n] + self.labels[i+1][n]                
                avg = s/2
                if avg != 0 and avg != 1:
                    variables += 1
            flippedIn2Runs.append(variables)
        
        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        plt.plot(flippedIn2Runs)
        plt.xticks(fontsize=self.tickSize)
        plt.yticks(fontsize=self.tickSize)
        plt.xlabel("Run",fontsize=self.labelSize)
        plt.ylabel("Points Flipped",fontsize=self.labelSize)
        if save:
            plt.savefig(self.savePath+"_FlippableVRun_rand.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()
    
    ''' Flipped in a single run unsorted barchart '''
    def flippedInSingleRunUnsortedBarchart(self, save=False):        
        flippedIn2Runs = np.array([[0]*2 for i in range(self.run_count-1)])
        for i in range(self.run_count-1):
            variables = 0
            tp_fn = 0
            tn_fp = 0
            for n in range(len(self.labels[0])):
                s = self.labels[i][n] + self.labels[i+1][n]                
                avg = s/2
                if avg != 0 and avg != 1:
                    if self.labels[i][n] == 0:
                        tp_fn += 1
                    else:
                        tn_fp += 1
                    
            flippedIn2Runs[i][0] = tp_fn
            flippedIn2Runs[i][1] = tn_fp
            
        
        bar = flippedIn2Runs.reshape((2, self.run_count-1))
        # create plot
        fig, ax = plt.subplots(figsize=(self.figsize_x,self.figsize_y))
        ax.grid(False)
        index = np.arange(self.run_count-1)
        bar_width = 0.5
        # opacity = 0
        
        rects1 = plt.bar(index, bar[0], bar_width,
        # alpha=opacity,
        # color='b',
        label='TI to FO')
        
        rects2 = plt.bar(index + bar_width, bar[1], bar_width,
        # alpha=opacity,
        # color='g',
        label='TO to FI')
        ax.fill(True)
        plt.xticks(fontsize=self.tickSize)
        plt.yticks(fontsize=self.tickSize)
        plt.xlabel('Runs',fontsize=self.labelSize)
        plt.ylabel('Flipped',fontsize=self.labelSize)
        plt.xlim([-0.5,29])
        plt.legend(fontsize=self.tickSize)
        plt.savefig(self.savePath+"_FlippableVRun_rand_pn_bar.pdf", bbox_inches="tight", pad_inches=0)

        plt.tight_layout()
        plt.show()
        
    ''' Flipped in a single run sorted - asc '''
    def flippedInSingleRunSortedASC(self, save=False):
        if self.flipped == -1:
            self.getFlipSummary()
        f_g = [0]
        doneRun = []
        flippable_temp = self.flippable.copy()
        # print("----")
        while sum(flippable_temp) != 0:
            flippedIn2Runs = np.array([[99999]*self.run_count for i in range(self.run_count)])
            # print(sum(flippable_temp), end=' ')
            for i in range(self.run_count):
                for j in range(i+1,self.run_count):
                    if i in doneRun or j in doneRun:
                        continue
                    variables = 0
                    for n in range(len(flippable_temp)):
                        if flippable_temp[n]:
                            s = self.labels[i][n] + self.labels[j][n]                
                            avg = s/2
                            if avg != 0 and avg != 1:
                                variables += 1
                    flippedIn2Runs[i][j] = variables
            
            minInd = unravel_index(flippedIn2Runs.argmin(), flippedIn2Runs.shape)
            doneRun.append(minInd[0])
            # if flippedIn2Runs[minInd[0]][minInd[1]] == 0:
            #     continue
            for n in range(len(flippable_temp)):
                if flippable_temp[n]:
                    s = self.labels[minInd[0]][n] + self.labels[minInd[1]][n]                
                    avg = s/2
                    if avg != 0 and avg != 1:
                        flippable_temp[n] = False
            f_g.append(f_g[-1] + flippedIn2Runs[minInd[0]][minInd[1]])

        f_g_percentage = [(x/(self.flipped+1))*100 for x in f_g]
        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        plt.plot(f_g)
        plt.xlabel("Run")
        plt.ylabel("Previously Undiscovered Flipped Points")
        if save:
            plt.savefig(self.savePath+"_FlippableVRun_asc.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()

        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        plt.plot(f_g_percentage)
        plt.xticks(fontsize=self.tickSize)
        plt.yticks(fontsize=self.tickSize)
        plt.xlabel("Run",fontsize=self.labelSize)
        plt.ylabel("Previously Undiscovered \nFlipped Points (%)",fontsize=self.labelSize)
        if save:
            plt.savefig(self.savePath+"_FlippableVRun_Percentage_asc.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()
        return f_g_percentage        
        
    ''' Flipped in a single run sorted - dsc '''
    def flippedInSingleRunSortedDSC(self, save=False):
        if self.flipped == -1:
            self.getFlipSummary()
        f_g = [0]
        doneRun = []
        flippable_temp = self.flippable.copy()
        # print("\n----")
        # while sum(flippable_temp) != 0:
        for _ in range(self.run_count-1):
            flippedIn2Runs = np.array([[0]*self.run_count for i in range(self.run_count)])
            print(sum(flippable_temp), end=' ')
            for i in range(self.run_count):
                for j in range(i+1,self.run_count):
                    if i in doneRun or j in doneRun:
                        continue
                    variables = 0
                    for n in range(len(flippable_temp)):
                        if flippable_temp[n] and self.labels[i][n] != self.labels[j][n]:
                            variables += 1
                    flippedIn2Runs[i][j] = variables
            
            maxInd = unravel_index(flippedIn2Runs.argmax(), flippedIn2Runs.shape)
            doneRun.append(maxInd[0])
            # if flippedIn2Runs[maxInd[0]][maxInd[1]] == 0:
            #     continue
            for n in range(len(flippable_temp)):
                if flippable_temp[n]:
                    if self.labels[maxInd[0]][n] != self.labels[maxInd[1]][n]:
                        flippable_temp[n] = False
            f_g.append(f_g[-1] + flippedIn2Runs[maxInd[0]][maxInd[1]])
        # print("\n-----")
        # print(f_g)
        # print(doneRun)

        f_g_percentage = [(x/(self.flipped+1))*100 for x in f_g]
        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        plt.plot(f_g)
        plt.xticks(fontsize=self.tickSize)
        plt.yticks(fontsize=self.tickSize)
        plt.xlabel("Run",fontsize=self.labelSize)
        plt.ylabel("Previously Undiscovered Flipped Points",fontsize=self.labelSize)
        if save:
            plt.savefig(self.savePath+"_FlippableVRun_dsc.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()

        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        plt.plot(f_g_percentage)
        plt.xticks(fontsize=self.tickSize)
        plt.yticks(fontsize=self.tickSize)
        plt.xlabel("Run",fontsize=self.labelSize)
        plt.ylabel("Previously Undiscovered \nFlipped Points (%)",fontsize=self.labelSize)
        if save:
            plt.savefig(self.savePath+"_FlippableVRun_Percentage_dsc.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()
        
        return f_g_percentage
    def MergeEffectsOfRunOrder(self, a, b, c,save=False):
        g = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        # plt.plot(a, "o-", linewidth=4, markersize=15, label="Typical scenario")
        # plt.plot(b, "D-", linewidth=4, markersize=15, label="Most optimistic scenario")
        # plt.plot(c, "s-", linewidth=4, markersize=15, label="Most pessimistic scenario")
        plt.plot(a,linewidth=4, label="Typical scenario")
        plt.plot(b,linewidth=4, label="Most optimistic scenario")
        plt.plot(c,linewidth=4, label="Most pessimistic scenario")
        
        plt.xticks(fontsize=self.tickSize)
        plt.yticks(fontsize=self.tickSize)
        plt.xlabel("Run",fontsize=self.labelSize)
        plt.ylabel("Previously Undiscovered \nFlipped Points (%)",fontsize=self.labelSize)
        plt.legend(fontsize=self.labelSize)
        if save:
            plt.savefig("FlipFig/Paper/"+self.filename+"_FlipOrder.pdf", bbox_inches="tight", pad_inches=0)
        plt.show()
        
        