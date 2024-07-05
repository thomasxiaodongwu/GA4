
import numpy as np
import geatpy as ea
import pandas as pd
import re
import copy
from Model1_NSGA2.EncodeInitPop import Encode

def decodeData(machineSupport, machineTime):
    machineInfo = {}
    jobInfo = {}
    numSubJob = 0  
    allMeachine = []  
    job_subJobInfo = {}  
    maxNumSubJobChain = 0  
    for i in machineSupport.index:  
        iJob = {}
        iJob_subJob = []
        for j in machineSupport.columns:
            subJob = {}
            if len(machineSupport.loc[i, j]) != 0:
                subJob['可选机器'] = sorted(machineSupport.loc[i, j].split(','))
                subJob['可选机器耗时'] = list(map(int, machineTime.loc[i, j].split(',')))
                iJob[j] = subJob
                numSubJob += 1
                allMeachine.extend(subJob['可选机器'])
                iJob_subJob.append(j)
        machineInfo['job%d' % i] = iJob
        job_subJobInfo['job%d' % i] = iJob_subJob
        maxNumSubJobChain = max(maxNumSubJobChain, len(iJob_subJob))
    jobInfo['所有工序的总数量'] = numSubJob
    jobInfo['所有机器的总数量'] = len(set(allMeachine))
    tmp = list(set(allMeachine))
    jobInfo['所有机器列表'] = sorted(tmp, key=lambda item: int(item[1:]))

    jobInfo['所有工件的工序明细'] = job_subJobInfo
    jobInfo['最长工序链的数量'] = maxNumSubJobChain

    return machineInfo, jobInfo


def getsOSIdx_2_SubJobIdx_JobIdx(job_subJobInfo):

    OSIdx = 0  
    jobIdx = -1
    OSIdx_2_JobIdx_Map = {}
    OSIdx_2_subJobIdx_Map = {}
    for ijob, subjob in job_subJobInfo.items():
        jobIdx += 1
        for subJobIdx in range(len(subjob)):
            OSIdx_2_JobIdx_Map[OSIdx] = jobIdx
            OSIdx_2_subJobIdx_Map[OSIdx] = subJobIdx
            OSIdx += 1
    return OSIdx_2_JobIdx_Map, OSIdx_2_subJobIdx_Map


def getsIdx_2_SubJobIdx_JobIdx(job_subJobInfo):

    idx = 0  
    jobIdx = -1
    Idx_2_JobIdx_subJobIdx_Map = {}
    for ijob, subjob in job_subJobInfo.items():
        jobIdx += 1
        for subJobIdx in range(len(subjob)):
            Idx_2_JobIdx_subJobIdx_Map[idx] = (jobIdx, subJobIdx)
            idx += 1
    return Idx_2_JobIdx_subJobIdx_Map


class MyProblem(ea.Problem):  
    def __init__(self, popSize, GSRate=0.6, LSRate=0.2):

        self.machineSupport = pd.read_csv('../data/machineSupport.csv', encoding='gbk', index_col=0)
        self.machineTime = pd.read_csv('../data/machineTime.csv', encoding='gbk', index_col=0)
        assert self.machineSupport.index.min() == 0, '要求工件的序号从0开始！'
        assert '0' in self.machineSupport.columns.min(), '要求工序的序号从0开始！'

        self.machineInfo, jobInfo = decodeData(self.machineSupport, self.machineTime)
        self.numJob = len(self.machineSupport)  
        self.numMachine = jobInfo['所有机器的总数量']
        self.numSubJob = jobInfo['所有工序的总数量']
        self.allMeachineName = jobInfo['所有机器列表']
        assert '1' in min(self.allMeachineName), '要求机器的序号从1开始！'

        self.job_subJobInfo = jobInfo['所有工件的工序明细']
        self.maxNumSubJobChain = jobInfo['最长工序链的数量']  

        self.OSIdx_2_JobIdx_Map, self.OSIdx_2_subJobIdx_Map = getsOSIdx_2_SubJobIdx_JobIdx(self.job_subJobInfo)
        self.Idx_2_JobIdx_subJobIdx_Map = getsIdx_2_SubJobIdx_JobIdx(self.job_subJobInfo)

        self.mExecQueue = self.initMeachineExecQueue(self.allMeachineName)

        self.workEnergyUnit = {'M1':{'工序0': 1}, 'M2':{'工序0':0.9},'M3':{'工序0':0.9},
                               'M4':{'工序1': 1}, 'M5':{'工序1':1},'M6':{'工序1':1},
                               'M7':{'工序2': 1}, 'M8':{'工序2':1},
                               'M9':{'工序3':1}, 'M10':{'工序3':1}}

        J = {}
        jobIdx = 0  
        for ijob, subjob in self.job_subJobInfo.items():
            J[jobIdx] = len(subjob)
            jobIdx += 1
        RSRate = 1 - GSRate - LSRate
        encodePop = Encode(self.machineInfo, popSize, J, self.numJob, self.numMachine, GSRate=GSRate, LSRate=LSRate, RSRate=RSRate)
        CHS_GS = encodePop.global_selection()
        CHS_LS = encodePop.local_selection()
        CHS_RS = encodePop.random_selection()
        self.initChrom = np.vstack([CHS_GS, CHS_LS, CHS_RS])
        lb1 = [0] * self.numSubJob
        ub1 = [1] * self.numSubJob
        lb2 = [0] * self.numSubJob
        ub2 = [self.numMachine - 1] * self.numSubJob
        name = 'MyProblem'  
        M = 2  
        maxormins = [1] * M  
        lb = lb1 + lb2  
        ub = ub1 + ub2  
        Dim = len(lb)  
        varTypes = [0] * self.numSubJob + [1] * self.numSubJob  
        lbin = [1] * Dim  
        ubin = [1] * Dim
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def initMeachineExecQueue(self, meachineList):
        mExecQueue = {}
        for iMachineName in meachineList:
            mExecQueue[iMachineName] = {'时间轴': 0, '执行队列': [], '起始时刻': []}   
        return mExecQueue

    def evalVars(self, x):  
        vars = x  
        allmExecQueue = self.decode(vars)
        F1, F2, allmExecQueue, allOptDetail = self.core(allmExecQueue)
        F = np.hstack([F1, F2])
        return F

    def core(self, allmExecQueue):
        machineInfo = copy.deepcopy(self.machineInfo)

        F1 = []  
        F2 = []  
        allOptDetail = []  
        for i, mQueue in enumerate(allmExecQueue):
            f1, f2 = 0, 0  
            iMachineState = [-1 for i in range(self.numMachine)]  
            finishTime = 0  
            istate = [-1 for i in range(self.numJob)]  

            while not self.isQueueAllExecuted(iMachineState):
                for j, jMachineName in enumerate(self.allMeachineName):
                    imExecQueue = mQueue[jMachineName]
                    nextTaskIdx = iMachineState[j] + 1  
                    if nextTaskIdx > (len(imExecQueue['执行队列'])) - 1:
                        continue
                    nextTask = imExecQueue['执行队列'][nextTaskIdx]  

                    if self.checkWeatherTaskCanExecute(jMachineName, nextTask, istate, mQueue):
                        nextJobIdx = nextTask[0]  
                        nextSubJobIdx = nextTask[1]  


                        nextTask_start = imExecQueue['时间轴']  

                        nextJobInfo = machineInfo['job%d' % (nextJobIdx)]['工序%d' % (nextSubJobIdx)]
                        machindIdx = np.where(np.array(nextJobInfo['可选机器']) == jMachineName)[0][0]
                        costTime = nextJobInfo['可选机器耗时'][machindIdx]

                        imExecQueue['时间轴'] += float(costTime)
                        nextTask_end = imExecQueue['时间轴']  
                        imExecQueue['起始时刻'].append((nextTask_start, nextTask_end))

                        iMachineState[j] += 1
                        istate[nextJobIdx] = nextSubJobIdx

                        finishTime = max(finishTime, imExecQueue['时间轴'])
                    else:
                        continue

            totalCostTime = 0  
            workTime = np.zeros((self.maxNumSubJobChain, len(self.allMeachineName)))  
            workCost = 0    
            standbyCost = 0 
            for j, jMachineName in enumerate(self.allMeachineName):
                imExecQueue = mQueue[jMachineName]
                if len(imExecQueue['起始时刻']) == 0:
                    continue
                totalCostTime += imExecQueue['起始时刻'][-1][1]
                jMachineWorkTime = 0
                kSubJobIdx = 0
                for k in range(len(imExecQueue['执行队列'])):
                    kTask = imExecQueue['执行队列'][k]
                    kJobIdx = kTask[0]  
                    kSubJobIdx = kTask[1]  
                    kTask_time = imExecQueue['起始时刻'][k]
                    workTime[kSubJobIdx, j] += kTask_time[1] - kTask_time[0]
                    workCost += (kTask_time[1] - kTask_time[0]) * self.workEnergyUnit[jMachineName]['工序%d'%kSubJobIdx]

            f1 += workCost
            f2 = np.array(workTime).max(axis=0).max()

            F1.append([f1])
            F2.append([f2])
            allOptDetail.append([workCost, standbyCost])
        return np.array(F1), np.array(F2), allmExecQueue, allOptDetail

    def decode(self, vars):

        allmExecQueue = []
        for item in vars:
            mExecQueue = copy.deepcopy(self.mExecQueue)
            layer1 = item[0:self.numSubJob]  
            minMachineIdx = self.machineSupport.index.min()
            layer1 = np.argsort(layer1)  
            layer2 = np.round(item[self.numSubJob:]).astype(int)  

            job_subJob_map = {}  
            for idx in range(len(layer2)):

                iMachineIdx = int(layer2[idx])

                iJobIdx, iSubJobIdx = self.Idx_2_JobIdx_subJobIdx_Map[idx]

                iJobName = 'job%d' % (iJobIdx)

                iSubJobName = '工序%d' % (iSubJobIdx)

                candMachine = self.machineInfo[iJobName][iSubJobName]['可选机器']

                iMachineName = candMachine[iMachineIdx % len(candMachine)]

                job_subJob_map[(iJobIdx, iSubJobIdx)] = iMachineName

            istate = [-1 for i in range(self.numJob)]
            for i in range(len(layer1)):
                OSidx = layer1[i]

                iJobIdx = self.OSIdx_2_JobIdx_Map[OSidx]

                iJobName = 'job%d' % (iJobIdx)

                iSubJobIdx = self.OSIdx_2_subJobIdx_Map[OSidx]
                iSubJobName = '工序%d' % (iSubJobIdx)

                if iSubJobIdx != (istate[iJobIdx] + 1):
                    layer1[i] = OSidx - (iSubJobIdx - (istate[iJobIdx] + 1))
                    iSubJobIdx = istate[iJobIdx] + 1
                    istate[iJobIdx] = iSubJobIdx

                else:  
                    istate[iJobIdx] = iSubJobIdx

                iMachineName = job_subJob_map[(iJobIdx, iSubJobIdx)]
                imQueue = mExecQueue[iMachineName]['执行队列']

                imQueue.append((iJobIdx, iSubJobIdx))

            allmExecQueue.append(mExecQueue)
        return allmExecQueue

    def isQueueAllExecuted(self, iMachineState):

        if self.numSubJob == sum(iMachineState) + len(iMachineState):
            return True
        else:
            return False

    def checkWeatherTaskCanExecute(self, iMachineName, iTask, istate, mExecQueue):

        iJobIdx = iTask[0]  
        iSubJobIdx = iTask[1]  
        if iSubJobIdx == 0:
            return True

        iJobIdx_curExecSubJob = istate[iJobIdx]  
        if iJobIdx_curExecSubJob != (iSubJobIdx - 1):  
            return False


        iTask_StartTime = mExecQueue[iMachineName]['时间轴']  

        for jMachineName in self.allMeachineName:
            jmExecQueue = mExecQueue[jMachineName]['执行队列']  
            jmExecQueueTime = mExecQueue[jMachineName]['起始时刻']  
            for k in range(len(jmExecQueue)):
                if jmExecQueue[k] == (iJobIdx, iSubJobIdx - 1):
                    preTask_endTime = jmExecQueueTime[k][1]  
                    mExecQueue[iMachineName]['时间轴'] = max(iTask_StartTime, preTask_endTime)
                    return True
        assert 0, '有问题！'