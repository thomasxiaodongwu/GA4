
import numpy as np
import geatpy as ea  


class moea_NSGA2_templet(ea.MoeaAlgorithm):
    """
    """
    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 dirName=None,
                 **kwargs):

        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing,
                         dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'NSGA2'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS  
        else:
            self.ndSort = ea.ndsortTNS  
        self.selFunc = 'tour'  
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  
            self.mutOper = ea.Mutinv(Pm=1)  
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  
            self.mutOper = ea.Mutbin(Pm=None)  
        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')

    def reinsertion(self, population, offspring, NUM):
        """
        """
        population = population + offspring

        [levels, _] = self.ndSort(population.ObjV, NUM, None, population.CV, self.problem.maxormins)  
        dis = ea.crowdis(population.ObjV, levels)  
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  
        return population[chooseFlag]

    def run(self, prophetPop=None):  

        population = self.population
        NIND = population.sizes
        self.initialization()
        population.initChrom()
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  
        self.call_aimFunc(population)  
        [levels, _] = self.ndSort(population.ObjV, NIND, None, population.CV, self.problem.maxormins)  
        population.FitnV = (1 / levels).reshape(-1, 1)  

        while not self.terminated(population):
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            tmpPop = [tuple(item) for item in offspring.ObjV]
            offspring.Chrom = self.recOper.do(offspring.Chrom)
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  
            self.call_aimFunc(offspring)  
            population = self.reinsertion(population, offspring, NIND)  
        return self.finishing(population)  
