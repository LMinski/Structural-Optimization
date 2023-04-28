"""
Backtracking Search Optimization Algorithm (BSA)
Platform: Matlab 2013a   
Cite this algorithm as;
[1]  P. Civicioglu, "Backtracking Search Optimization Algorithm for 
numerical optimization problems", Applied Mathematics and Computation,
219, 8121-8144, 2013.
Copyright Notice
Copyright (c) 2012, Pinar Civicioglu
All rights reserved.
Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions are 
met:
    * Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the copyright 
      notice, this list of conditions and the following disclaimer in 
      the documentation and/or other materials provided with the distribution
      
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import time
from datetime import timedelta


def generate_population(popsize, dim, low, up):
    """
    Generates a random population
    """
    pop = np.zeros((popsize, dim))
    for i in range(popsize):
        pop[i,:] = np.random.rand(1, dim)*(up - low) + low
        
    return pop


def boundary_control(pop, low, up):
    """
    Controls the boundary to stay between low and up
    """
    popsize, dim = pop.shape
    for i in range(popsize):
        for j in range(dim):
            k = np.random.rand() < np.random.rand() # you can change boundary-control strategy
            if pop[i,j] < low[j]:
                if k:
                    pop[i,j] = low[j]
                else:
                    pop[i,j] = np.random.rand()*(up[j] - low[j]) + low[j]
                    
            if pop[i,j] > up[j]:
                if k:
                    pop[i,j] = up[j]
                else:
                    pop[i,j] = np.random.rand()*(up[j] - low[j]) + low[j]
    
    return pop


def feval(fnc, pop):
    """
    Iterates over pop and sends the cost
    of each individual of the population
    """
    cost = np.zeros((pop.shape[0]))
    for idx, p in enumerate(pop):
        cost[idx] = fnc(p)
    
    return cost


def convert_time(s):
    """
    Provided a time 's' in seconds,
    this method converts to a string
    in a format hh:mm:ss
    """
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))


def bsa(fnc, bounds, popsize=32, epoch=100):
    """Backtracking Search Optimization Algorithm (BSA)

    Args:
        fnc (method): insert your objective function
        bounds (nested tuple): bounds as ((a, b), (c, d), (e, f))
            the first index of the nested tuple is for the lower bound
            and the last for the upper bound of the variable
        popsize (int, optional): size of the population. Defaults to 32.
        epoch (int, optional): amount of iterations. Defaults to 100.
            epoch*popsize equals to the number of objective function evaluations

    Returns:
        results (object):
            results.x is the globalminimizer array,
                i.e., optimum values for the variables
            results.y is the globalminimum scalar,
                i.e., the optimum cost with the variables results.x
            results.convergence shows the history of the results.y
    """
    
    # INITIALIZATION
    DIM_RATE = 1
    low, up = np.zeros((len(bounds))), np.zeros((len(bounds)))
    for idx, b in enumerate(bounds):
        low[idx] = b[0]
        up[idx] = b[1]
    
    dim = len(bounds)
    
    pop = generate_population(popsize, dim, low, up) # see Eq.1 in [1]
    fitnesspop = feval(fnc, pop)
    historical_pop = generate_population(popsize, dim, low, up) # see Eq.2 in [1]

    # historical_pop  is swarm-memory of BSA as mentioned in [1].
    
    # ------------------------------------------------------------------------------------------
    elapsed_time = 0
    cost_history = np.zeros((epoch))
    for epk in range(epoch):
        start = time.time()
        # SELECTION-I
        if np.random.rand() < np.random.rand():
            historical_pop = pop  # see Eq.3 in [1]
        
        historical_pop = historical_pop[np.random.permutation(popsize), :] # see Eq.4 in [1]
        
        F = 3*np.random.standard_normal() # see Eq.5 in [1], you can other F generation strategies 
        map_ = np.zeros((popsize, dim)) # see Algorithm-2 in [1]
          
        if np.random.rand() < np.random.rand():
            for i in range(popsize):
                u = np.random.permutation(dim)
                map_[i, u[np.arange(0, np.ceil(DIM_RATE*np.random.rand()*dim), dtype='int32')]] = 1
        else:
            for i in range(popsize):
                map_[i, np.random.randint(dim)] = 1
        
        # RECOMBINATION (MUTATION+CROSSOVER)
        offsprings = pop + (map_*F)*(historical_pop - pop) # see Eq.5 in [1]
        offsprings = boundary_control(offsprings, low, up) # see Algorithm-3 in [1]
        
        # SELECTON-II
        fitnessoffsprings = feval(fnc,offsprings)
        ind = fitnessoffsprings < fitnesspop
        fitnesspop[ind] = fitnessoffsprings[ind]
        pop[ind,:] = offsprings[ind,:]
        globalminimum = np.min(fitnesspop)
        cost_history[epk] = globalminimum
        ind = fitnesspop == globalminimum
        globalminimizer = pop[ind,:]
        globalminimizer = globalminimizer[0,:]
        
        # Calculate reamining time
        porcentagem = (epk/epoch)*100
        end = time.time()
        if porcentagem % 50 == 0:
            duration = end - start
            total_time = duration * epoch
            elapsed_time = duration * epk
            remaining_time = total_time - elapsed_time
            
            # print(f'Best = {globalminimum:20.8f}   ###   ( {porcentagem:3.0f}% completed ): '
            #       f'Elapsed time {convert_time(elapsed_time)} -- '  
            #       f'Reamining time {convert_time(remaining_time)} -- '
            #       f'Predicted time {convert_time(total_time)}')
    
    class results:
        x = globalminimizer
        y = globalminimum
        convergence = cost_history
    
    return results
