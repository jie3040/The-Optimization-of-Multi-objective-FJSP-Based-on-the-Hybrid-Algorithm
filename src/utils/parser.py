#!/usr/bin/env python

# This module parses .fjs files as found in the "Monaldo" FJSP dataset.
# More explanations on this file format can be found in the dataset.


def parse(path):
    file = open(path, 'r')

    firstLine = file.readline()
    firstLineValues = list(map(int, firstLine.split()[0:2]))

    jobsNb = firstLineValues[0]
    machinesNb = firstLineValues[1]

    jobs = []

    for i in range(jobsNb):
        currentLine = file.readline()
        currentLineValues = list(map(int, currentLine.split()))

        operations = []

        j = 1
        while j < len(currentLineValues):
            k = currentLineValues[j]
            j = j+1

            operation = []

            for ik in range(k):
                machine = currentLineValues[j]
                j = j+1
                processingTime = currentLineValues[j]
                j = j+1

                operation.append({'machine': machine, 'processingTime': processingTime})

            operations.append(operation)

        jobs.append(operations)

    file.close()

    return {'machinesNb': machinesNb, 'jobs': jobs}

import random
from math import exp
from src.genetic import genetic


def random_choose(population):
    cho_population=random.choice(population)
    return cho_population

def generate_neighbor_solution(solution):
    (os,ms)=solution
    i=random.randint(0,len(os)-1)
    j=random.randint(0,len(os)-1)
    os[i],os[j]=os[j],os[i]
    return solution

# print(generate_neighbor_solution(([1, 2, 1, 2, 0, 0], [1, 1, 0, 1, 0, 0])))   

def update_pe_list(new_solution, population, path, temperature):
    ns_f1 = genetic.timeTaken(new_solution,parse(path))
    ns_f2 = genetic.Maximalload(new_solution,parse(path))
    ns_f3 = genetic.Totalload(new_solution,parse(path))

    dominatedByOtherSolution = False
    # Search if a solution in a PE vector is dominated by
    for solution in population:
        s_f1 = genetic.timeTaken(solution,parse(path))
        s_f2 = genetic.Maximalload(solution,parse(path))
        s_f3 = genetic.Totalload(solution,parse(path))
        # Check if it is dominated
        if (ns_f1 <= s_f1 and ns_f2 < s_f2 and ns_f3 < s_f3 or 
            ns_f1 < s_f1  and ns_f2 <= s_f2 and ns_f3 < s_f3 or 
            ns_f1 < s_f1  and ns_f2 < s_f2 and ns_f3 <= s_f3 or  
            ns_f1 < s_f1 and ns_f2 <= s_f2 and ns_f3 <= s_f3 or
            ns_f1 <= s_f1 and ns_f2 < s_f2 and ns_f3 <= s_f3 or
            ns_f1 <= s_f1 and ns_f2 <= s_f2 and ns_f3 < s_f3):
            if random.random()<exp(float(-temperature)/10):
              population.remove(solution)
            dominatedByOtherSolution = True
        # and if is it is not dominated by a value ina a PE vector
    # Append to PE border if it is legal
    if dominatedByOtherSolution and new_solution not in population:
        population.append(new_solution)
    return population

def find_fitness_value(object_solution,population,path):
    ob_f1 = genetic.timeTaken(object_solution,parse(path))
    ob_f2 = genetic.Maximalload(object_solution,parse(path))
    ob_f3 = genetic.Totalload(object_solution,parse(path))
    fitness=0
    for solution in population:
        s_f1 = genetic.timeTaken(solution,parse(path))
        s_f2 = genetic.Maximalload(solution,parse(path))
        s_f3 = genetic.Totalload(solution,parse(path))
        if (ob_f1 >= s_f1 and ob_f2 > s_f2 and ob_f3 > s_f3 or 
            ob_f1 > s_f1  and ob_f2 >= s_f2 and ob_f3 > s_f3 or 
            ob_f1 > s_f1  and ob_f2 > s_f2 and ob_f3 >= s_f3 or  
            ob_f1 > s_f1 and ob_f2 >= s_f2 and ob_f3 >= s_f3 or
            ob_f1 >= s_f1 and ob_f2 > s_f2 and ob_f3 >= s_f3 or
            ob_f1 >= s_f1 and ob_f2 >= s_f2 and ob_f3 > s_f3):
            fitness=fitness+1
    return fitness

