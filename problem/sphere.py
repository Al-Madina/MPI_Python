#!/usr/bin/env python

"""
This program defines a function of the form: x1^2 + x2^2 + .... +xd^2 where d is the dimension of the problem
"""

__author__ = "Ahmed Hassan"
__copyright__ = "NICOG 2022"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ahmed Hassan"
__email__ = "ahmedhassan@aims.ac.za"
__status__ = "Development"

import numpy as np


class Sphere:


    def __init__(self, seed, DIMENSION=10, a=-1, b=1):
        """Initialize an instance of the problem.

        Parameters
        ----------
        seed: int
            A seed for the random number genertor
        DIMENSION: int
            The dimension of the problem
        a: int
            The starting point of the interval in which the solution is to be found
        b: int 
            The ending point of the interval in which the solution is to be found
        """
        # Not needed but often times problem domains may depend on a random number generator
        self.seed = seed 
        # Dimension of the problem
        self.DIMENSION = DIMENSION 
        # The beginning of the interval in which we look for a solution
        self.a = a 
        # The end of the interval in which we look for a solution
        self.b = b 


    def evaluate(self, solution: np.array):
        """Evaluate a solution. Here we assume each solution is a numpy array of size d where d is the dimension
        of the problem.

        Parameters
        ----------
        solution: numpy.array
            A candidate solution for the problem which is represented as a numpy array
        """

        if solution.shape[0] != self.DIMENSION:
            raise ValueError("The dimension of a candidate solution is not equal to the dimension of the problem")

        # Just to simulate heavy computation. These lines are not part of the solution evaluation
        for i in range(100000):
            x = np.random.uniform(1, 100, size=100)
            try:
                y = np.log(x)
            except: pass
        # End of the dummy HEAVY simulation!

        # This line computes the value of the solution (fitness). It computes x1^2 + x2^2 + .... +xd^2
        # where d is the dimension of the problem
        return sum(solution**2)
# Happy coding :-)