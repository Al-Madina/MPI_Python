#!/usr/bin/env python

"""
This program uses MPI to distribute the fitness evaluation in a generational genetic algorithm (GA).

The generational GA is very simple and utilizes basic genetic operators since the purpose is not to developed the most
sophisticated GA in the world. Rather, the aim is to show you how to use MPI to make your GA run faster. 

I distributed the fitness evalaution since it is ususally the computational bottleneck.
"""

__author__ = "Ahmed Hassan"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ahmed Hassan"
__email__ = "ahmedhassan@aims.ac.za"
__status__ = "Development"


import numpy as np
import numpy.random as rng
# Import mpi4py to use MPI with python
from mpi4py import MPI

# It is useful to have a separate package for coding problem domains
from problem.sphere import Sphere


"""
IMPORTANT:
You can code your parallel GA using an object-oriented approach. I usually do that. However, with MPI I foundit easier 
to code it using sort of functional approach.
"""

def evaluate(mypop, problem):
    """Fitness Evaluation.

    Parameters
    ----------
    mypop: numpy.array
        An array of individuals to be evaluated by this process wherer each individual is a numpy array
    problem: User-defined
        The problem to be solved. Must implement `evaluate` method that takes an indivdual and retruns
    """
    myfitness = np.array(list(map(problem.evaluate, mypop)), dtype=np.float64)
    return myfitness


def init_pop(popSize):
    """Initializes the population. 

    The population is a numpy array of numpy arrays. The outer numpy array is the population and each one of the inner
    array represents an individual (a candidate solution). We use numpy arrays since MPI4PY does not pickle numpy arrays
    but pickles any other python objects including primitive types such as numbers. Pickling objects with large memory 
    will add additional overhead and slows down the performance. Therefore, it is a good practice to try to stick to 
    numpy arrays when working with MPI4PY.

    If numpy arrays does not suit your individual representation you have two options:
        - Use a map function to map your numpy arrays to the individual representation of your choice. In this way,
            MPI4PY works fine with numpy arrays and your map function will translate the numpy arrays to format that 
            your GA will understand.
        - Use python objects (dict, lists, etc) or define your own class for individuals and transfer them using MPI4PY 
            routines that are designed for trasnfering general objects (not numpy arrays). However, be aware that these 
            routines are slower than the routines designed to communicate numpy arrays. Therefore, if your individuals 
            have a large memory print this approach might not be convenient.

    Parameters
    ----------
    popSize: int
        The size of the population
    """
    pop = []
    for idx in range(popSize):
        # Create an individual (a candidate solution)
        ind = rng.uniform(problem.a, problem.b, size=problem.DIMENSION)
        pop.append(ind)
    return np.array(pop, dtype=np.float64)


def select(fitness, tSize):
    """Performs tournament selection.

    Parameters
    ----------
    fitness: numpy.array
        A numpy array where the element at index `i` represents the fitness of individual at index `i` in the population.
    tSize: int
        The tournament size.
    """
    # Draw the individuals participating in the tournament (draw their indexes)
    indexes = rng.randint(low=0, high=len(fitness), size=tSize)
    # Find the index of the winner (the individual with the minimum fitness)
    winner_idx = np.argmin(fitness[indexes])
    # Return the winner of the tournament
    return indexes[winner_idx]


def crossover(parent1, parent2):
    """One-point crossover.

    Parameters
    ----------
    parent1: numpy.array
        The first parent.
    parent2: numpy.array
        The second parent
    """
    point = rng.randint(1, parent1.size-1)
    child1 = np.zeros(parent1.size, dtype=np.float64)
    child2 = np.zeros(parent2.size, dtype=np.float64)
    child1[:point] = parent1[:point]
    child1[-point:] = parent2[-point:]
    child2[:point] = parent2[:point]
    child2[-point:] = parent1[-point:]
    return child1, child2


def mutate(ind):
    """Random mutation that modifies a number of genes at random. 

    This mutation operator changes the individual in place.

    Parameters
    ----------
    ind: numpy.array
        The individual to be mutated (offspring).
    """
    # Choose the number of genes to modify at random
    num_genes = rng.randint(0, ind.size)
    # Choose the indexes of genes to modify (recall that an individual is a numpy array)
    gene_indexes = rng.randint(0, ind.size, num_genes)
    # Modifies the genes in place
    ind[gene_indexes] = rng.uniform(problem.a, problem.b, num_genes)


def regenerate(pop, fitness, tSize, xrate):
    """Regenerates a new population via crossover and mutation. Crossover is perform with a rate.

    pop: numpy.array
        The population which is a numpy array of numpy arrays.
    fitness: numpy.array
        A numpy array representing the fitness of individuals.
    tSize: int
        Tournament size.
    xrate: float
        Crossover rate.
    """

    # A temporary population
    tmpPop = []
    while len(tmpPop) < len(pop):
        p = rng.random()
        # Check if crossover is to be used
        if p < xrate:
            parent1 = select(fitness, tSize)
            parent2 = select(fitness, tSize)
            while parent1 == parent2: parent2 = select(fitness, tSize)
            child1, child2 = crossover(pop[parent1], pop[parent2])
            tmpPop.append(child1)
            if len(tmpPop) == len(pop): break
            tmpPop.append(child2)
        # Use mutation instead
        else:
            idx = select(fitness, tSize)
            ind = pop[idx]
            mutate(ind)
            tmpPop.append(ind)
    return np.array(tmpPop, dtype=np.float64)


if __name__ == "__main__":

    # Get the communicator that will use all processes
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    # Get my rank. Pocesses are identified by integer numbers (rank) from 0 to P - 1 where P is the number of processes
    myrank = comm.Get_rank()


    # Parameters. Practically, you need to read these from the command line. Process with rank 0 shoud do the reading
    # and then broadcasting the arguments to other processes. See "genAlgRunner.py"

    # Seed for random number generator (RNG)
    seed = 12345 
    # Population size
    popSize = 128 
    # Tournament size
    tSize = 5
    # Crossover rate. Mutation rate is 1 - xrate 
    xrate = 0.7 
    # Number of generations
    gens = 20 
    # Dimension of the problem
    DIMENSION = 10 

    # Seed the RNG for reproduciblity
    rng.seed(seed)
    # Create a problem instance
    problem = Sphere(seed, 1, -1)
    problem.DIMENSION = DIMENSION

    # If the population size is not divisible by the number of processes terminate. You do not have to do this. You can
    # distribute the extra individuals across the processes as explained in the presentation (load balancing problem).
    # Here I am trying to keep everything simple and focus on the core idea (distributing the fitness evaluation).
    if popSize % nprocs != 0 and nprocs > 1:
        comm.Abort(errorcode=2) # errorcode is helping for debugging
        # This is an impolite way of aborting MPI program! It forces other process to terminate immediately. For instance, 
        # if resources are open by other processes they might not get closed properly

    # Determines how many individuals each process will get
    myPopSize = popSize//nprocs

    # IMPORTANT: The whole population is None in all processes except in process 0 because process 0 will generate the 
    # initial pop and regenerate new pop. At every generation other processes need to get their share of the pop to 
    # evaluate and do  not need the whole pop
    pop = None
    # The fitness array is also None in all processes except in process 0. As explained above!
    fitness = None
    # Best fitness (we assuming a minimization problem)
    best_fitness = float('inf')

    # The parallel GA starts from here
    start = MPI.Wtime() # To measure the execution time

    # Every process will get part of the population to evaluate (fitness evaluation is the computational bottleneck)
    myPop = np.empty((myPopSize, problem.DIMENSION))

    # Only one process creates the initial population
    if myrank == 0:
        # Create the initial population
        pop = init_pop(popSize)
        # Create the fitness array for the entire population. Fitness[idx] is the fitness of individual at pop[idx]
        fitness = np.empty(popSize, dtype=np.float64)

    # Scatter the population so that each process will get a portion of the population of size `myPopSize`. This portion
    # of the population will be stored in `myPop`.
    # The first part of the pop (of size equals `myPopSize`) will go to process 0, the second part of the pop will go to
    # process 1, and so on.
    # If you want to scatter data of variable sizes use `Scatterv` routine of MPI4PY (note the "v" at the end)
    comm.Scatter(pop, myPop, root=0)

    # Distribute fitness evaluation. Recall that `myPop` is the part of the population allocated for the current process 
    # running the line below
    myfitness = evaluate(myPop, problem)

    # Gather the fitness from all processes in one array `fitness` and this array is only available in process 0.
    # See "genAlgRunner.py" for more details on "Gather"
    comm.Gather(myfitness, fitness, root=0)

    # Reduce on best and worst fitness
    # Update the best fitness by process 0 (note that process 0 will gather the fitness of whole pop)
    if myrank == 0 and min(fitness) < best_fitness:
        best_fitness = min(fitness)
        print("Initial best fitness = ", min(fitness))

    # Entire the optimization cycle (run the GA for a number of generations)
    for g in range(gens):
        # May be a useful print!
        if myrank == 0: 
            print("gen = ", g)

        # Generate a new population by process 0. Other processes do not have to participate in this since the 
        # population regeneration is cheap (assumption!)
        if myrank == 0:
            pop = regenerate(pop, fitness, tSize, xrate)
        else:
            # If I am a process other than 0, I just chill and do nothing!
            pop = None 

        # Scatter the population so that each process will get a portion of the population of size `myPopSize`. This 
        # portion of the population will be stored in myPop
        # See "genAlgRunner.py" for more info on "Scatter"
        comm.Scatter(pop, myPop, root=0)

        # Distribute fitness evaluation. Recall that myPop is the portion of the population allocated for the current 
        # process running the line below
        myfitness = evaluate(myPop, problem)

        # Gather the fitness from all processes in one array and this array is only available in rank 0
        comm.Gather(myfitness, fitness, root=0)

        # Update the best fitness by rank 0 
        # Recall that that process 0 gathered the fitness of whole pop in `fitness`
        if myrank == 0 and min(fitness) < best_fitness:
            best_fitness = min(fitness)
            print("Updated: best fitness = ", min(fitness))
    # Done with all generations

    # Calculate the execution time for this process
    duration = np.array(MPI.Wtime() - start, dtype=np.float64)

    # Need to also calc the execution time for the whole parallel program which is the same as the longest execution time
    # (in fact, your MPI program will not terminate until all processes stop).
    total_duration = np.empty((), dtype=np.float64)

    # The execution time of the entire MPI program is the same as the execution time for the
    # the process that runs the longest.
    # See "genAlgRunner.py" for more info on "Reduce"
    comm.Reduce(duration, total_duration, op=MPI.MAX, root=0)

    # Print out the results
    if myrank == 0:
        print("-"*50)
        print(f"Best fitness = {best_fitness}")
        print(f"Execution time = {total_duration} seconds")
# Happy coding :-)