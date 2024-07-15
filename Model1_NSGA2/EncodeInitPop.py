import numpy as np
import random


class Encode:
    def __init__(self, Matrix, popSize, J, jobNum, machineNum, GSRate=0.6, LSRate=0.2, RSRate=0.2):
        self.Matrix = Matrix
        self.GSNum = int(GSRate * popSize)  
        self.LSNum = int(LSRate * popSize)  
        self.RSNum = int(RSRate * popSize)  

        self.J = J  
        self.jobNum = jobNum  
        self.machineNum = machineNum  
        self.lenChrome = 0
        for cnt in self.J.values():  
            self.lenChrome += cnt

    def OSList(self):
        return np.random.random(self.lenChrome)

    def Site(self, job, op):
        O_num = 0
        for i in range(len(self.J)):
            if i == job:
                return O_num + op
            else:
                O_num = O_num + self.J[i]
        return O_num

    def initPopulation(self, num):
        return np.zeros((num, self.lenChrome))

    def global_selection(self):
        OS = self.initPopulation(self.GSNum)  
        MS = self.initPopulation(self.GSNum)  
        for it in range(self.GSNum):  
            OS[it] = self.OSList()
            machineTimearr = np.zeros(self.machineNum, dtype=int)  
            jobList = [x for x in range(self.jobNum)]  
            random.shuffle(jobList)  
            for jobi in jobList:  
                timei = self.Matrix['job%d'%jobi] 
                for i in range(len(timei)):  
                    candMachineOperation = timei['machines%d'%i]['可选机器耗时']  
                    candMachine = timei['machines%d'%i]['可选机器']
                    operationTime = np.ones(self.machineNum, dtype=int) * np.inf
                    for idx, M in enumerate(candMachine):
                        MIdx = int(M[1:]) - 1   
                        operationTime[MIdx] = candMachineOperation[idx]

                    sumTime = machineTimearr + operationTime

                    minTime = min(sumTime)  
                    k = np.where(sumTime==minTime)[0][0]  
                    machineTimearr[k] = minTime  

                    site = self.Site(jobi, i)  
                    MS[it][site] = k
        CHS_GS = np.hstack([OS, MS])  
        return CHS_GS

    def local_selection(self):
        '''
        '''
        OS = self.initPopulation(self.LSNum)  
        MS = self.initPopulation(self.LSNum)  
        for it in range(self.LSNum):
            OS[it] = self.OSList()
            jobList = [x for x in range(self.jobNum)]  
            for jobi in jobList:  
                machineTimearr = np.zeros(self.machineNum, dtype=int)
                timei = self.Matrix['job%d'%jobi] 
                for i in range(len(timei)):  
                    candMachineOperation = timei['machines%d'%i]['可选机器耗时']  
                    candMachine = timei['machines%d'%i]['可选机器']
                    operationTime = np.ones(self.machineNum, dtype=int) * np.inf
                    for idx, M in enumerate(candMachine):
                        MIdx = int(M[1:]) - 1   
                        operationTime[MIdx] = candMachineOperation[idx]

                    sumTime = machineTimearr + operationTime

                    minTime = min(sumTime)  
                    k = np.where(sumTime==minTime)[0][0]  
                    machineTimearr[k] = minTime  

                    site = self.Site(jobi, i)  
                    MS[it][site] = k
        CHS_LS = np.hstack([OS, MS])  
        return CHS_LS

    def random_selection(self):
        '''
        '''
        OS = self.initPopulation(self.RSNum)  
        MS = self.initPopulation(self.RSNum)  
        for it in range(self.RSNum):
            OS[it] = self.OSList()  
            jobList = [x for x in range(self.jobNum)]  
            random.shuffle(jobList)  
            for jobi in jobList:  
                timei = self.Matrix['job%d'%jobi] 
                for i in range(len(timei)):
                    machinePosition = []
                    candMachineOperation = timei['machines%d'%i]['可选机器耗时']  
                    candMachine = timei['machines%d'%i]['可选机器']  
                    for idx, M in enumerate(candMachine):
                        MIdx = int(M[1:]) - 1   
                        machinePosition.append(MIdx)

                    machineNum = random.choice(machinePosition)  
                    k = machinePosition.index(machineNum)
                    site = self.Site(jobi, i)
                    MS[it][site] = k
        CHS_RS = np.hstack([OS, MS])  
        return CHS_RS


if __name__ == '__main__':
    matrix = [[[2, 6, 5, 3, 4],
               [9999, 8, 9999, 4, 9999],
               [4, 9999, 3, 6, 2]],
              [[3, 9999, 6, 9999, 5],
               [4, 6, 5, 9999, 9999],
               [9999, 7, 11, 5, 8]]]
    J = {1: 3, 2: 3}  
    e = Encode(matrix, 10, J, 2, 5)  
    CHS_GS = e.global_selection()
    CHS_LS = e.local_selection()
    CHS_RS = e.random_selection()
    CHS = np.vstack([CHS_GS, CHS_LS, CHS_RS])
    print(CHS)
