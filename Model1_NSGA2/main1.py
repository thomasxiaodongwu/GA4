

import os
import re
import sys
sys.path.append('../')
import subprocess
import shutil
try:
    import numpy as np
    import pandas as pd
    import geatpy as ea
    import matplotlib.pyplot as plt
    from Model1_NSGA2.MyProblem1 import MyProblem  
    from Model1_NSGA2.moea_NSGA2_templet import moea_NSGA2_templet

except:
    print('尚未安装第三方依赖库')
    import numpy as np
    import pandas as pd
    import geatpy as ea
    import matplotlib.pyplot as plt
    from Model1_NSGA2.MyProblem1 import MyProblem  
    from Model1_NSGA2.moea_NSGA2_templet import moea_NSGA2_templet
np.seterr(invalid='ignore')

def clearDir(dir='Log'):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def draw(pop, problem, dirName):
    if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
        figureName = 'Pareto Front Plot'
        plotter = ea.PointScatter(problem.M, grid=True, legend=True, title=figureName,
                                  saveName=dirName + figureName)
        plotter.add(problem.ReferObjV, color='gray', alpha=0.1, label='True PF')
        plotter.add(pop.ObjV, color='red', label='MOEA PF')
        plotter.draw()
    else:
        figureName = 'Parallel Coordinate Plot'
        plotter = ea.ParCoordPlotter(problem.M, grid=True, legend=True, title=figureName,
                                     saveName=dirName + figureName)
        plotter.add(problem.TinyReferObjV, color='gray', alpha=0.5, label='True Objective Value')
        plotter.add(pop.ObjV, color='red', label='MOEA Objective Value')
        plotter.draw()
    plotter.fig.savefig(plotter.saveName + '.png', dpi=400, bbox_inches='tight')
    plotter.close()


def drawGanttChartType2(ST_ET, machineName, jobList, fileName='甘特图.png', productStartTime=[]):
    colors = ['cyan', 'yellow']
    plt.figure(figsize=(13, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  
    end_time = 0
    y_machineName_ticks = []
    for i in range(len(jobList)):
        if len(productStartTime) == 0:
            productstart = 0
        else:
            productstart = productStartTime[i]
        product = jobList[i]  
        iST_ET = ST_ET[i]  
        y = i * 2 + 1  
        y_machineName_ticks.append(y)
        for k, processId in enumerate(product):  

            ST = iST_ET[k][0]
            ET = iST_ET[k][1]
            width = ET - ST
            end_time = max(end_time, iST_ET[k][1])

            if processId == 0:  
                plt.barh(y=y, width=width, height=0.8, left=ST, edgecolor="black", color="red")
            else:
                plt.barh(y=y, width=width, height=0.8, left=ST, edgecolor="black", color="cyan")  

            y_time_ticks = y - 0.5
            plt.text(x=ST, y=y_time_ticks, s='%0.1f' % ST, fontsize=8)
            plt.text(x=ET, y=y_time_ticks, s='%0.1f' % ET, fontsize=8)

            plt.text(x=ST, y=y, s='%s' % str(processId))

            if k == 0:
                pre_ET = productstart
                width = ST - productstart
            else:
                pre_ET = iST_ET[k - 1][1]
                width = ST - pre_ET
            plt.barh(y=y, width=width, height=0.8, left=pre_ET, edgecolor="black", color="yellow")

    plt.yticks(y_machineName_ticks, machineName, rotation=0)
    plt.xlim(0, end_time + 1)
    plt.xlabel('任务执行时间/小时')

    import matplotlib.patches as mpatches
    labels = ['执行任务', '待机']
    patches = [mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(len(colors))]
    plt.legend(handles=patches, ncol=2, loc='upper right')  
    plt.savefig('%s' % fileName, dpi=600, format="png", bbox_inches='tight')

    plt.close()


def GA(popSize=50, maxGen=200, pm=0.3, pc=0.7, GSRate=0.6, LSRate=0.2):
    rootPath = os.getcwd()
    if 'Model1_NSGA2' not in rootPath:
        rootPath = rootPath + '/Model1_NSGA2/'  
    rootPath = rootPath + '/popSize[%d]_maxGen[%d]_pm[%0.1f]_pc[%0.1f]_GSRate[%0.1f]_LSRate[%0.1f]/' % (popSize, maxGen, pm, pc, GSRate, LSRate)
    problem = MyProblem(popSize, GSRate, LSRate)
    algorithm = moea_NSGA2_templet(problem,
                                   ea.Population(Encoding='RI', NIND=popSize),
                                   MAXGEN=maxGen,  
                                   logTras=50,  
                                   trappedValue=1e-6,  
                                   maxTrappedCount=np.inf)  
    algorithm.mutOper.Pm = pm  
    algorithm.recOper.XOVR = pc
    prophet = problem.initChrom
    res = ea.optimize(algorithm, prophet=prophet, verbose=True, drawing=0, outputMsg=False, drawLog=False,
                      saveFlag=True,
                      dirName=rootPath)
    uniqueIndex = []
    uniqueObjv = []
    res['optPop'].ObjV = np.round(res['optPop'].ObjV, 4)
    for i, item in enumerate(res['optPop'].ObjV):
        if tuple(item) not in uniqueObjv:
            uniqueIndex.append(i)
            uniqueObjv.append(tuple(item))
    res['optPop'] = res['optPop'][uniqueIndex]
    res['optPop'].save('%s/optPop' % rootPath)  
    assert res['success'], '未找到可行解，请增大种群规模或最大进化代数！'

    optPop = res['optPop']
    fileName = '/结果_帕累托前沿'
    fileName = rootPath + fileName
    draw(optPop, problem, fileName)

    vars = res['optPop'].Phen
    allmExecQueue = problem.decode(vars)
    F1, F2, allmExecQueue, allOptDetail = problem.core(allmExecQueue)
    clearDir(rootPath + '/非支配解/')
    f = open(rootPath + '/结果_Log.txt', 'w')
    f.write('Solving time consuming: %s min\n' % (algorithm.passTime / 60))
    for j in range(len(F1)):
        f1 = F1[j][0]
        f2 = F2[j][0]
        mQueue = allmExecQueue[j]
        optDetial = allOptDetail[j]
        workCost, standbyCost = optDetial
        f.write('-' * 30 + ' Nondominated Solution[%d] ' % (j + 1) + '-' * 30 + '\n')
        f.write('F1[%f] F2[%f]\n' % (f1, f2))
        f.write('Decision Variable: %s\n' % list(res['Vars'][j]))
        f.write('优先级：%s  请忽略：%s\n' % (workCost, standbyCost))

        fileName = rootPath + '/非支配解/帕累托[%d]_甘特图.png' % (j + 1)
        allMeachineName = problem.allMeachineName
        ST_ET = []
        jobList = []
        for j, jMachineName in enumerate(allMeachineName):
            imExecQueueTime = mQueue[jMachineName]['起始时刻']
            ST_ET.append(imExecQueueTime)
            jobList.append(mQueue[jMachineName]['执行队列'])
        drawGanttChartType2(ST_ET, allMeachineName, jobList, fileName=fileName, productStartTime=[])
    f.close()


if __name__ == '__main__':
    GA(popSize=50, maxGen=500, pm=0.3, pc=0.7, GSRate=0.3, LSRate=0.4)
