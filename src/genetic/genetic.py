#!/usr/bin/env python

# This module contains the detailed implementation of every genetic operators
# The code is strictly mirroring the section 4.3 of the attached paper

from random import randint,sample,choice,random
from itertools import permutations
from src import config
from src.genetic import decoding
from src.utils import parser, gantt

parameters = parser.parse("1.txt")

def timeTaken(os_ms, pb_instance):
    (os, ms) = os_ms
    decoded = decoding.decode(pb_instance, os, ms)

    # Getting the max for each machine
    max_per_machine = []
    for machine in decoded:
        max_d = 0
        for job in machine:
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
        max_per_machine.append(max_d)

    return max(max_per_machine)

def Maximalload(os_ms, pb_instance):
    (os, ms) = os_ms
    decoded = decoding.decode(pb_instance, os, ms)

    total_load_permachine =  []
    for machine in decoded:
        proc_list = []
        for job in machine:
            proc = job[1]
            proc_list.append(proc)
        total_load_permachine.append(sum(proc_list))

    return max(total_load_permachine)  

def Totalload(os_ms, pb_instance):
    (os, ms) = os_ms
    decoded = decoding.decode(pb_instance, os, ms)

    total_load_permachine =  []
    for machine in decoded:
        proc_list = []
        for job in machine:
            proc = job[1]
            proc_list.append(proc)
        total_load_permachine.append(sum(proc_list))

    return sum(total_load_permachine)       

# 4.3.1 Selection
#######################
def fast_non_dominated_sort(values1, values2, values3):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[q] > values1[p] and values2[q] > values2[p] and values3[q] > values3[p]) or (values1[q] >= values1[p] and 
            values2[q] > values2[p] and values3[q] > values3[p]) or (values1[q] > values1[p] and values2[q] >= values2[p] and 
            values3[q] > values3[p])  or (values1[q] > values1[p] and values2[q] > values2[p] and values3[q] >= values3[p]) or(values1[q] > values1[p] and 
            values2[q] >= values2[p] and values3[q] >= values3[p]) or (values1[q] >= values1[p] and values2[q] > values2[p] and 
            values3[q] >= values3[p]) or (values1[q] >= values1[p] and values2[q] >= values2[p] and values3[q] > values3[p]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[p] > values1[q] and values2[p] > values2[q] and values3[p] > values3[q]) or (values1[p] >= values1[q] and 
            values2[p] > values2[q] and values3[p] > values3[q]) or (values1[p] > values1[q] and values2[p] >= values2[q] and 
            values3[p] > values3[q])  or (values1[p] > values1[q] and values2[p] > values2[q] and values3[p] >= values3[q]) or(values1[p] > values1[q] and 
            values2[p] >= values2[q] and values3[p] >= values3[q]) or (values1[p] >= values1[q] and values2[p] > values2[q] and 
            values3[p] >= values3[q]) or (values1[p] >= values1[q] and values2[p] >= values2[q] and values3[p] > values3[q]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

# print(initializePopulation(parse(path)))


def timeTaken_values(population,parameter,popSize):
    timeTaken_values_list=[]
    for i in range(0,popSize):
        timeTaken_values_list.append(timeTaken(population[i],parameter))

    return timeTaken_values_list

def Maximalload_values(population,parameter,popSize):
    Maximalload_values_list=[]
    for i in range(0,popSize):
        Maximalload_values_list.append(Maximalload(population[i],parameter))

    return Maximalload_values_list

def Totalload_values(population,parameter,popSize):
    Totalload_values_list=[]
    for i in range(0,popSize):
        Totalload_values_list.append(Totalload(population[i],parameter))

    return Totalload_values_list

#print(Totalload_values(population,parse(path),popSize))

# print(population[1])

# print(non_dominated_sorted_solution)


import math

#Function to calculate crowding distance
def crowding_distance(values1, values2, values3):
    distance = [0 for i in range(0,len(values1))]
    distance[0] = 4444444444444444
    distance[len(values1) - 1] = 4444444444444444
    for k in range(1,len(values1)-1):
        distance[k] =  math.fabs(values1[k+1]-values1[k-1])+ math.fabs(values2[k+1] - values2[k-1])+math.fabs(values3[k+1] - values3[k-1])
    return distance


# print(crowding_distance(timeTaken_values(population,parse(path),popSize), Maximalload_values(population,parse(path),popSize), Totalload_values(population,parse(path),popSize)))


#print("The best front for Generation number ",gen_no, " is")
#for valuez in non_dominated_sorted_solution[0]:
        #print(np.round(population[valuez],3),end=" ")
#print("\n")

def front_search(position,fronts):
    front_rank = 0
    for i in range(len(fronts)):
        if position in fronts[i]:
           front_rank = i     
    return front_rank 

#print(front_search(0,non_dominated_sorted_solution))
import math

def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(max(values),values) in list1:
            sorted_list.append(index_of(max(values),values))
        values[index_of(max(values),values)] = 0
    return sorted_list

def elitistSelection(fronts,population):
    elitistSelection_result=[]
    ElitistSelection_result_show=[]
    i = 0
    while (len(elitistSelection_result)+ len(fronts[i])) <= 0.5*config.popSize:
        for k in range(len(fronts[i])):
            elitistSelection_result.append((fronts[i])[k])
        i += 1
    F=fronts[i]
    F_population=[]
    for w in range(0,len(F)):
        F_population.append(population[F[w]])
    F_population_timeTaken_values=timeTaken_values(F_population,parameters,len(F_population))
    listss=[]
    for h in range(0,len(F_population_timeTaken_values)):
        listss.append(h)
    sortbyvaluess=sort_by_values(listss,F_population_timeTaken_values)
    F_sort_population_by_timeTaken=[]
    for k in range(0,len(sortbyvaluess)):
        F_sort_population_by_timeTaken.append(F_population[sortbyvaluess[k]])
    F_crowding=crowding_distance(timeTaken_values(F_sort_population_by_timeTaken,parameters,len(F_sort_population_by_timeTaken)), Maximalload_values(F_sort_population_by_timeTaken,parameters,len(F_sort_population_by_timeTaken)), Totalload_values(F_sort_population_by_timeTaken,parameters,len(F_sort_population_by_timeTaken)))
    lists=[]
    for m in range(0,len(F_crowding)):
        lists.append(m)
    sortbyvalues=sort_by_values(lists, F_crowding)
    F_sort_population_by_crowding_distance=[]
    for n in range (0,len(sortbyvalues)):
        F_sort_population_by_crowding_distance.append(F_sort_population_by_timeTaken[sortbyvalues[n]])
    elitistSelection_result_show=[]
    for l in range(0,len(elitistSelection_result)):
        elitistSelection_result_show.append(population[elitistSelection_result[l]])
    total_elitistSelection_result_show=elitistSelection_result_show
    for o in range(0,len(F_sort_population_by_crowding_distance)):
        total_elitistSelection_result_show.append(F_sort_population_by_crowding_distance[o])
    for p in range(0,config.halfSize):
        ElitistSelection_result_show.append(total_elitistSelection_result_show[p])
    return ElitistSelection_result_show


#print(elitistSelection(non_dominated_sorted_solution))



# 4.3.2 Crossover operators
###########################

def precedenceOperationCrossover(p1, p2, parameters):
    J = parameters['jobs']
    jobNumber = len(J)
    jobsRange = range(1, jobNumber+1)
    sizeJobset1 = randint(0, jobNumber)

    jobset1 = sample(jobsRange, sizeJobset1)

    o1 = []
    p1kept = []
    for i in range(len(p1)):
        e = p1[i]
        if e in jobset1:
            o1.append(e)
        else:
            o1.append(-1)
            p1kept.append(e)

    o2 = []
    p2kept = []
    for i in range(len(p2)):
        e = p2[i]
        if e in jobset1:
            o2.append(e)
        else:
            o2.append(-1)
            p2kept.append(e)

    for i in range(len(o1)):
        if o1[i] == -1:
            o1[i] = p2kept.pop(0)

    for i in range(len(o2)):
        if o2[i] == -1:
            o2[i] = p1kept.pop(0)

    return (o1, o2)


def jobBasedCrossover(p1, p2, parameters):
    J = parameters['jobs']
    jobNumber = len(J)
    jobsRange = range(0, jobNumber)
    sizeJobset1 = randint(0, jobNumber)

    jobset1 = sample(jobsRange, sizeJobset1)
    jobset2 = [item for item in jobsRange if item not in jobset1]

    o1 = []
    p1kept = []
    for i in range(len(p1)):
        e = p1[i]
        if e in jobset1:
            o1.append(e)
            p1kept.append(e)
        else:
            o1.append(-1)

    o2 = []
    p2kept = []
    for i in range(len(p2)):
        e = p2[i]
        if e in jobset2:
            o2.append(e)
            p2kept.append(e)
        else:
            o2.append(-1)

    for i in range(len(o1)):
        if o1[i] == -1:
            o1[i] = p2kept.pop(0)

    for i in range(len(o2)):
        if o2[i] == -1:
            o2[i] = p1kept.pop(0)

    return (o1, o2)


def twoPointCrossover(p1, p2):
    pos1 = randint(0, len(p1) - 1)
    pos2 = randint(0, len(p1) - 1)

    if pos1 > pos2:
        pos2, pos1 = pos1, pos2

    offspring1 = p1
    if pos1 != pos2:
        offspring1 = p1[:pos1] + p2[pos1:pos2] + p1[pos2:]

    offspring2 = p2
    if pos1 != pos2:
        offspring2 = p2[:pos1] + p1[pos1:pos2] + p2[pos2:]

    return (offspring1, offspring2)


def crossoverOS(p1, p2, parameters):
    if choice([True, False]):
        return precedenceOperationCrossover(p1, p2, parameters)
    else:
        return jobBasedCrossover(p1, p2, parameters)


def crossoverMS(p1, p2):
    return twoPointCrossover(p1, p2)


def crossover(population, parameters):
    newPop = []
    i = 0
    while i < len(population):
        (OS1, MS1) = population[i]
        (OS2, MS2) = population[i+1]

        if random() < config.pc:
            (oOS1, oOS2) = crossoverOS(OS1, OS2, parameters)
            (oMS1, oMS2) = crossoverMS(MS1, MS2)
            newPop.append((oOS1, oMS1))
            newPop.append((oOS2, oMS2))
        else:
            newPop.append((OS1, MS1))
            newPop.append((OS2, MS2))

        i = i + 2

    return newPop


# 4.3.3 Mutation operators
##########################

def swappingMutation(p):
    pos1 = randint(0, len(p) - 1)
    pos2 = randint(0, len(p) - 1)

    if pos1 == pos2:
        return p

    if pos1 > pos2:
        pos1, pos2 = pos2, pos1

    offspring = p[:pos1] + [p[pos2]] + \
          p[pos1+1:pos2] + [p[pos1]] + \
          p[pos2+1:]

    return offspring


def neighborhoodMutation(p):
    pos3 = pos2 = pos1 = randint(0, len(p) - 1)

    while p[pos2] == p[pos1]:
        pos2 = randint(0, len(p) - 1)

    while p[pos3] == p[pos2] or p[pos3] == p[pos1]:
        pos3 = randint(0, len(p) - 1)

    sortedPositions = sorted([pos1, pos2, pos3])
    pos1 = sortedPositions[0]
    pos2 = sortedPositions[1]
    pos3 = sortedPositions[2]

    e1 = p[sortedPositions[0]]
    e2 = p[sortedPositions[1]]
    e3 = p[sortedPositions[2]]

    Permutations = list(permutations([e1, e2, e3]))
    permutation  = choice(Permutations)

    offspring = p[:pos1] + [permutation[0]] + \
          p[pos1+1:pos2] + [permutation[1]] + \
          p[pos2+1:pos3] + [permutation[2]] + \
          p[pos3+1:]

    return offspring


def halfMutation(p, parameters):
    o = p
    jobs = parameters['jobs']

    size = len(p)
    r = int(size/2)

    positions = sample(range(size), r)

    i = 0
    for job in jobs:
        for op in job:
            if i in positions:
                o[i] = randint(0, len(op)-1)
            i = i+1

    return o


def mutationOS(p):
    if choice([True, False]):
        return swappingMutation(p)
    else:
        return neighborhoodMutation(p)


def mutationMS(p, parameters):
    return halfMutation(p, parameters)


def mutation(population, parameters):
    newPop = []

    for (OS, MS) in population:
        if random() < config.pm:
            oOS = mutationOS(OS)
            oMS = mutationMS(MS, parameters)
            newPop.append((oOS, oMS))
        else:
            newPop.append((OS, MS))

    return newPop