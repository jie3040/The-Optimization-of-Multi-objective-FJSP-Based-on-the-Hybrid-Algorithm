#!/usr/bin/env python

# This script contains a high level overview of the proposed hybrid algorithm
# The code is strictly mirroring the section 4.1 of the attached paper

import time
import random
import math
from src.utils import parser
from src.genetic import encoding, genetic, termination,decoding
from src import config
import numpy
import calculate
# Beginning

# Parameters Setting
parameters = parser.parse("1.txt")

n=0
while n<1:
 optimum_solution_function_1_set=[]
 optimum_solution_function_2_set=[]
 optimum_solution_function_3_set=[]
 t0 = time.time()
 time_count=[]
 t4=time.time()
 params = {}
 # Initial temperature
 params['temperature'] = 100.0
 # Cooling factor
 params['alpha'] = 0.9
 # Steps number keeping the temperature
 params['np'] = 20
 # Final temperature
 params['final_temperature'] = 0.01
 # Number of iterations without improving
 params['number_iterations_without_improves'] = 3000
 # Total number of iterations
 params['nnum'] = 0

    # Initialize the Population
 population = encoding.initializePopulation(parameters)
 gen = 1
 iteration_time_list=[]
    # Evaluate the population
 while not termination.shouldTerminate(gen):
    non_dominated_sorted_solution = genetic.fast_non_dominated_sort(genetic.timeTaken_values(population,parameters,config.popSize)[:],genetic.Maximalload_values(population,parameters,config.popSize)[:],genetic.Totalload_values(population,parameters,config.popSize)[:])
        # Genetic Operators
    Front_0=non_dominated_sorted_solution[0]
    Front_0_solution=[]
    for i in range(0,len(Front_0)):
     Front_0_solution.append(population[Front_0[i]])
    Function_1=calculate.timeTaken_values(Front_0_solution,parameters,len(Front_0_solution))
    Function_2=calculate.Maximalload_values(Front_0_solution,parameters,len(Front_0_solution))
    Function_3=calculate.Totalload_values(Front_0_solution,parameters,len(Front_0_solution))
    optimum_solution_function_1_set.append(min(Function_1))
    optimum_solution_function_2_set.append(min(Function_2))
    optimum_solution_function_3_set.append(min(Function_3))
    AA=genetic.elitistSelection(non_dominated_sorted_solution,population)    
    population = AA
    t2=time.time()
    population = genetic.crossover(population, parameters)
    population = genetic.mutation (population, parameters)
    population = population + AA
    t3=time.time()
    choosen_solution=parser.random_choose(population)
    new_solution=parser.generate_neighbor_solution(choosen_solution)
    new_population=parser.update_pe_list(new_solution,population,'1.txt',params['temperature'])
    if new_population == population:
        delta=parser.find_fitness_value(new_solution,population,'1.txt')-parser.find_fitness_value(choosen_solution,population,'1.txt')
        if (random.random() < math.exp(float(delta) / params['temperature'])):
            new_population.remove(choosen_solution)
            new_population.append(new_solution)
    new_population=decoding.deleteDuplicatedElementFromList(new_population,'1.txt')
    if len(new_population)<config.popSize:
        quantity=config.popSize-len(new_population)
        new_individual=encoding.InitializePopulation(parameters,quantity)
        new_population=new_population+new_individual
    population=new_population
    if gen % params['np'] == 0:
        params['temperature'] *= params['alpha']
    iteration_time=t3-t2
    iteration_time_list.append(iteration_time)
    t5=time.time()
    time_count.append(t5-t4)
    gen = gen + 1
    



 t1 = time.time()
 total_time = t1 - t0
 average_iteration_time=numpy.mean(iteration_time_list)
 print("Finished in {0:.2f}s".format(total_time))
 print("Average iteration time is {0:.4f}s".format(average_iteration_time))

    # Termination Criteria Satisfied ?
 non_dominated_sorted_solution = genetic.fast_non_dominated_sort(genetic.timeTaken_values(population,parameters,config.popSize)[:],genetic.Maximalload_values(population,parameters,config.popSize)[:],genetic.Totalload_values(population,parameters,config.popSize)[:])
 front0= non_dominated_sorted_solution[0]
 front0_solution=[]
 for i in range(0,len(front0)):
    front0_solution.append(population[front0[i]])

 front0=front0_solution

 function_1=calculate.timeTaken_values(front0,parameters,len(front0))
 function_2=calculate.Maximalload_values(front0,parameters,len(front0))
 function_3=calculate.Totalload_values(front0,parameters,len(front0))

 print(function_1)
 print(function_2)
 print(function_3)
 #print(time_count)
 #print(optimum_solution_function_1_set)
 #print(optimum_solution_function_2_set)
 #print(optimum_solution_function_3_set)
 print(front0)
 n=n+1

 