#!/usr/bin/env python

"""
This program uses implements a simple steady-state GA.

The purpose of this tutorial is to show you how to distribute the runs of the GA. You can follow the same steps if you
wish to distribute the runs of other stochastic algorithms.
"""


__author__ = "Ahmed Hassan"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ahmed Hassan"
__email__ = "ahmedhassan@aims.ac.za"
__status__ = "Development"



import numpy as np
import numpy.random as rng

from problem.sphere import Sphere

class GeneticAlg:

    def __init__(self, seed, popSize, tSize):
        self.seed = seed
        rng.seed(seed)
        self.popSize = popSize
        self.tSize = tSize
        self.best_fitness = float('inf')
        self.worst_fitness = float('-inf')
        self.worst_index = -1
        self.pop = []
        self.fitness = []
        self._problem = None # Will be set


    def getproblem(self):
        return self._problem


    def setproblem(self, problem):
        self._problem = problem


    def delproblem(self):
        del self._problem

    problem = property(getproblem, setproblem, delproblem)


    def _evaluate(self, index):
        self.fitness[index] = self.problem.evaluate(self.pop[index])


    def _init_pop(self):
        for idx in range(self.popSize):
            ind = self.problem.b*rng.random(self.problem.DIMENSION) \
                - self.problem.a*rng.random(self.problem.DIMENSION)  
            self.pop.append(ind)
            self.fitness.append(None)
            self._evaluate(idx)
            if self.fitness[-1] < self.best_fitness:
                self.best_fitness = self.fitness[-1]
            if self.fitness[-1] > self.worst_fitness:
                self.worst_fitness = self.fitness[-1]
                self.worst_index = idx
        self.pop += [None, None]
        #print(f"initial best fitness = {self.best_fitness}")


    def _select(self):
        winner = -1
        winner_fitness = float('inf')
        indexes = rng.randint(low=0, high=self.popSize, size=self.tSize) #high excl
        for index in indexes:
            if self.fitness[index] < winner_fitness:
                winner = index
                winner_fitness = self.fitness[index]
        return winner


    def _crossover(self, parent1, parent2):
        ind1 = self.pop[parent1]
        ind2 = self.pop[parent2]
        point = rng.randint(0, ind1.size)
        child1 = np.zeros(ind1.size)
        child2 = np.zeros(ind2.size)
        child1[:point] = ind1[:point]
        child1[-point:] = ind2[-point:]
        child2[:point] = ind2[:point]
        child2[-point:] = ind1[-point:]
        self.pop[-2] = child1
        self.pop[-1] = child2


    def _mutate(self):
        idx = rng.randint(0, self.pop[-2].size)
        self.pop[-2][idx] = self.problem.b*rng.random() - self.problem.a*rng.random()
        idx = rng.randint(0, self.pop[-2].size)
        self.pop[-1][idx] = self.problem.b*rng.random() - self.problem.a*rng.random()



    def _update(self):
        self._evaluate(-2)
        self._evaluate(-1)
        better_child = -1 if self.fitness[-1] < self.fitness[-2] else -2
        if self.fitness[better_child] < self.worst_fitness:
            self.pop[self.worst_index] = self.pop[better_child]
            self._find_worst_ind()

            if self.fitness[better_child] < self.best_fitness:
                self.best_fitness = self.fitness[better_child]
                # print(f"Best fitness updated: {self.best_fitness}")


    def _find_worst_ind(self):
        self.worst_fitness = float('-inf')
        self.worst_index = -1
        for idx in range(self.popSize):
            if self.fitness[-1] > self.worst_fitness:
                self.worst_fitness = self.fitness[-1]
                self.worst_index = idx


    def run(self, gens):
        self._init_pop()
        for _ in range(gens):
            parent1 = self._select()
            parent2 = self._select()
            while parent1 == parent2: parent2 = self._select()
            self._crossover(parent1, parent2)
            self._mutate()
            self._update()
        return self.best_fitness


if __name__ == "__main__":
    pass
# Happy coding :-)
