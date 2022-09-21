#!/usr/bin/env python

"""
This program uses MPI to run a stead-state genetic algorithm (GA) in parallel. This called distributed the algorithm runs
since we run the same algorithm but with different seeds for the random number generator to establish statistical
significance of the results for stochastic algorithms such as GAs.

The program creates `N` GAs where each instance of the GA is seeded with a different 
random number for the random number generator.

Please note that you can use this code with the stochastic algorithm of your choice.
"""

__author__ = "Ahmed Hassan"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ahmed Hassan"
__email__ = "ahmedhassan@aims.ac.za"
__status__ = "Development"


import numpy as np
import numpy.random as rng
# Needs to import mpi4py to use MPI with python
from mpi4py import MPI
from algorithms.GeneticAlg import GeneticAlg
from problem.sphere import Sphere


# This program is just a runner for another program!
if __name__ == "__main__":
    # Get the communicator that will use all processes
    comm = MPI.COMM_WORLD
    # Get my rank. Processes are identified by an integers (rank) from 0 to P - 1 where P is the number of processes
    myrank = comm.Get_rank()

    # A global seed for repeatability
    # Note that we used numpy arrays even if we are dealing with just numbers since MPI4PY does not pickle numpy arrays
    # but pickles any other python objects including primitive types such as numbers. Pickling objects with large memory 
    # will add additional overhead. Therefore, it is a good practice to try to stick to numpy arrays.
    # Python will treat these numpy arrays (below) as numbers. For instance, you can do: global_seed + 1 which will give
    # you 0.
    # You may ask why MPI4PY exclude numpy arrays. The answer is that numpy arrays are buffer-like object. Therefore, 
    # they can be treated as a memory buffer.
    global_seed = np.array(-1, dtype=np.int32)
    # Number of generations
    gens = np.array(-1, dtype=np.int32)
    # Population size
    popSize = np.array(-1, dtype=np.int32)
    # Tournament size
    tSize = np.array(-1, dtype=np.int32)
    # Dimension of the problem. We use a benchmark function (sphere).
    DIMENSION = np.array(-1, dtype=np.int32)

    # Only process 0 will parse the command args. 
    if myrank == 0:
        import sys
        if len(sys.argv) >= 6:
            global_seed = np.array(sys.argv[1], dtype=np.int32)
            gens = np.array(sys.argv[2], dtype=np.int32)
            popSize = np.array(sys.argv[3], dtype=np.int32)
            tSize = np.array(sys.argv[4], dtype=np.int32)
            DIMENSION = np.array(sys.argv[5], dtype=np.int32)
        elif len(sys.argv) == 5:
            global_seed = np.array(sys.argv[1], dtype=np.int32)
            gens = np.array(sys.argv[2], dtype=np.int32)
            popSize = np.array(sys.argv[3], dtype=np.int32)
            tSize = np.array(sys.argv[4], dtype=np.int32)
            DIMENSION = np.array(10, dtype=np.int32)
        elif len(sys.argv) == 4:
            global_seed = np.array(sys.argv[1], dtype=np.int32)
            gens = np.array(sys.argv[2], dtype=np.int32)
            popSize = np.array(sys.argv[3], dtype=np.int32)
            tSize = np.array(2, dtype=np.int32)
            DIMENSION = np.array(10, dtype=np.int32)
        elif len(sys.argv) == 3:
            global_seed = np.array(sys.argv[1], dtype=np.int32)
            gens = np.array(sys.argv[2], dtype=np.int32)
            popSize = np.array(100, dtype=np.int32)
            tSize = np.array(2, dtype=np.int32)
            DIMENSION = np.array(10, dtype=np.int32)
        elif len(sys.argv) == 2:
            global_seed = np.array(sys.argv[1], dtype=np.int32)
            gens = np.array(1000, dtype=np.int32)
            popSize = np.array(100, dtype=np.int32)
            tSize = np.array(2, dtype=np.int32)
            DIMENSION = np.array(10, dtype=np.int32)
        else: # This else block is redundant if you set these values as default values instead of the -1 used above!
            global_seed = np.array(123, dtype=np.int32)
            gens = np.array(100, dtype=np.int32)
            popSize = np.array(100, dtype=np.int32)
            tSize = np.array(2, dtype=np.int32)
            DIMENSION = np.array(10, dtype=np.int32)


    # Process 0 will broadcast the values of the command args to all other processes
    comm.Bcast(global_seed, root=0)
    comm.Bcast(gens, root=0)
    comm.Bcast(popSize, root=0)
    comm.Bcast(tSize, root=0)
    comm.Bcast(DIMENSION, root=0)


    # Just print info from all processes to ensure that every process gets the correct args and not the -1 used as 
    # default values
    info = f"I am process = {myrank}\nseed = {global_seed}\ngenerations = {gens}\n"\
           + f"pop size = {popSize}\ntournament size = {tSize}\nDimension = {DIMENSION}\n" + "-"*50

    print(info)
    # Sync all processes by using a barrier. 
    # Every process that has called the barrier will be blocked until all other processes call the barrier as well.
    comm.Barrier()

    # Each process will get a unique seed based on the global seed
    # If you fail to do this, then every process will use the same global seed and your processes will generate the same
    # sequence of random numbers leading to the same result eventually which is not what we want!
    myseed = global_seed * (2*myrank + 1)

    # Every process will create a problem instance
    myproblem = Sphere(myseed)
    myproblem.DIMENSION = DIMENSION

    # Every process will create an instance of the GA
    alg = GeneticAlg(myseed, popSize, tSize)
    # Let the GA knows the problem it is solving.
    alg.problem = myproblem

    # For measuring the runtime of the GA
    start = MPI.Wtime()
    # Runs the GA
    result = alg.run(gens)
    # Sticking to numpy arrays
    result = np.array(result, dtype=np.float64)
    # End of execution
    end = MPI.Wtime()
    # Now, measure the runtime of the GA
    duration = np.array(end - start, dtype=np.float64)
    # Useful print out!
    print(f"I am process {myrank}. My execution time = {duration}")

    # Start collecting the results from other processes.
    # Process 0 will ask every other process to send its result
    # In all processes, the results list is not needed except in process 0
    results = None 
    # The total runtime is also not need except in process 0
    total_duration = None 
    if myrank == 0:
        # Get the total number of processes
        nprocs = comm.Get_size() 
        # Initialize an empty array of size equals to the number of processes. Element at index i is the result of process i
        results = np.empty(nprocs, dtype=np.float64)
        # Initialize the total duration as an empty array (just holds a position in memory for it)
        total_duration = np.empty((),dtype=np.float64)

    # Gather the result from all other process into process 0
    comm.Gather(result, results, root=0)

    # Reduce is a global computation function. Here we sum the duration taken by the GA across all processes. Dividing 
    # this by the number of processes will give you the average runtime of the GA.
    # Alternatively, you could gather the duration from all processes as done above for the results and do the summation
    # on your own. But why doing that if you could use the "Reduce" function in one step!
    comm.Reduce(duration, total_duration, op=MPI.SUM, root=0)

    # By now, `results` is an array containing the results from all processes and you can compute some statistics: mean, min, max, std, etc
    if myrank == 0:
        print("\n\n--- Result Summary ---")
        print(f"Number of runs = {nprocs}")
        print(f"Average fitness = {np.mean(results)}")
        print(f"Standard deviation = {np.std(results)}")
        print(f"Minimum = {min(results)}")
        print(f"Maximum = {max(results)}")
        # print all results one result per line
        print("Printing out results from all runs: one result per line")
        print("\n".join(np.array(results, dtype=np.str)))
        print(f"Average execution time = {total_duration/nprocs}")
# Happy coding :-)