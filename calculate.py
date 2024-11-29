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

path="1.txt"

import random

popSize = 400
halfSize = 200
maxGen = 200
pr = 0.005
pc = 0.8
pm = 0.1
latex_export = True

def generateOS(parameters):
    jobs = parameters['jobs']

    OS = []
    i = 0
    for job in jobs:
        for op in job:
            OS.append(i)
        i = i+1

    random.shuffle(OS)

    return OS
    
def generateMS(parameters):
    jobs = parameters['jobs']

    MS = []
    for job in jobs:
        for op in job:
            randomMachine = random.randint(0, len(op)-1)
            MS.append(randomMachine)

    return MS
 
def initializePopulation(parameters):
    gen1 = []

    for i in range(popSize):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        gen1.append((OS, MS))

    return gen1

import sys


def split_ms(pb_instance, ms):
    jobs = []
    current = 0
    for index, job in enumerate(pb_instance['jobs']):
        jobs.append(ms[current:current+len(job)])
        current += len(job)
    return jobs


def get_processing_time(op_by_machine, machine_nb):
    for op in op_by_machine:
        if op['machine'] == machine_nb:
            return op['processingTime']
    print("[ERROR] Machine {} doesn't to be able to process this task.".format(machine_nb))
    sys.exit(-1)


def is_free(tab, start, duration):
    for k in range(start, start+duration):
        if not tab[k]:
            return False
    return True


def find_first_available_place(start_ctr, duration, machine_jobs):
    max_duration_list = []
    max_duration = start_ctr + duration

    # max_duration is either the start_ctr + duration or the max(possible starts) + duration
    if machine_jobs:
        for job in machine_jobs:
            max_duration_list.append(job[3] + job[1])  # start + process time

        max_duration = max(max(max_duration_list), start_ctr) + duration

    machine_used = [True] * max_duration

    # Updating array with used places
    for job in machine_jobs:
        start = job[3]
        long = job[1]
        for k in range(start, start + long):
            machine_used[k] = False

    # Find the first available place that meets constraint
    for k in range(start_ctr, len(machine_used)):
        if is_free(machine_used, k, duration):
            return k


def decode(pb_instance, os, ms):
    o = pb_instance['jobs']
    machine_operations = [[] for i in range(pb_instance['machinesNb'])]

    ms_s = split_ms(pb_instance, ms)  # machine for each operations

    indexes = [0] * len(ms_s)
    start_task_cstr = [0] * len(ms_s)

    # Iterating over OS to get task execution order and then checking in
    # MS to get the machine
    for job in os:
        index_machine = ms_s[job][indexes[job]]
        machine = o[job][indexes[job]][index_machine]['machine']
        prcTime = o[job][indexes[job]][index_machine]['processingTime']
        start_cstr = start_task_cstr[job]

        # Getting the first available place for the operation
        start = find_first_available_place(start_cstr, prcTime, machine_operations[machine - 1])
        name_task = "{}-{}".format(job, indexes[job]+1)

        machine_operations[machine - 1].append((name_task, prcTime, start_cstr, start))

        # Updating indexes (one for the current task for each job, one for the start constraint
        # for each job)
        indexes[job] += 1
        start_task_cstr[job] = (start + prcTime)

    return machine_operations


def translate_decoded_to_gantt(machine_operations):
    data = {}

    for idx, machine in enumerate(machine_operations):
        machine_name = "Machine-{}".format(idx + 1)
        operations = []
        for operation in machine:
            operations.append([operation[3], operation[3] + operation[1], operation[0]])

        data[machine_name] = operations

    return data

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib import colors as mcolors

colors = []

for name, hex in mcolors.cnames.items():
    colors.append(name)


def parse_data(file):
    try:
        textlist = open(file).readlines()
    except:
        return

    data = {}

    for tx in textlist:
        if not tx.startswith('#'):
            splitted_line = tx.split(',')
            machine = splitted_line[0]
            operations = []

            for op in splitted_line[1::]:
                label = op.split(':')[0].strip()
                l = op.split(':')[1].strip().split('-')
                start = int(l[0])
                end = int(l[1])
                operations.append([start, end, label])

            data[machine] = operations
    return data

import random
import itertools

def timeTaken(os_ms, pb_instance):
    (os, ms) = os_ms
    decoded = decode(pb_instance, os, ms)

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
    decoded = decode(pb_instance, os, ms)

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
    decoded = decode(pb_instance, os, ms)

    total_load_permachine =  []
    for machine in decoded:
        proc_list = []
        for job in machine:
            proc = job[1]
            proc_list.append(proc)
        total_load_permachine.append(sum(proc_list))

    return sum(total_load_permachine)       

a = generateOS(parse(path))
b = generateMS(parse(path))
c = parse(path)



#Function to carry out NSGA-II's fast non dominated sort
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

population = initializePopulation(parse(path))


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

non_dominated_sorted_solution = fast_non_dominated_sort(timeTaken_values(population,parse(path),popSize)[:],Maximalload_values(population,parse(path),popSize)[:],Totalload_values(population,parse(path),popSize)[:])

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

gen_no=0

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

def elitistSelection(fronts):
    elitistSelection_result=[]
    ElitistSelection_result=[]
    
    i = 0
    while (len(elitistSelection_result)+ len(fronts[i])) <= 0.5*popSize:
        for k in range(len(fronts[i])):
            elitistSelection_result.append((fronts[i])[k])
        i += 1
    F=fronts[i]
    F_crowding=[]
    for j in range(0,len(F)):
        F_crowding.append(crowding_distance(timeTaken_values(population,parse(path),popSize), Maximalload_values(population,parse(path),popSize), Totalload_values(population,parse(path),popSize))[F[j]])
    lists=[]
    for m in range(0,len(F_crowding)):
        lists.append(m)
    sortbyvalues=sort_by_values(lists, F_crowding)
    for n in range (0,len(sortbyvalues)):
        elitistSelection_result.append(F[sortbyvalues[n]])
    for p in range(0,halfSize):
        ElitistSelection_result.append(elitistSelection_result[p])
    return ElitistSelection_result

#print(elitistSelection(non_dominated_sorted_solution))


from src import config

import math
import random
import matplotlib.pyplot as plt


#print(len(front0))
#方法二，利用三维轴方法
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义图像和三维格式坐标轴
fig=plt.figure()
ax2 = Axes3D(fig)

#ax2.scatter3D(function_1,function_2,function_3, s=20,cmap='Blues')  #绘制散点图

# 添加坐标轴(顺序是Z, Y, X)
ax2.set_zlabel('Total machine workload', fontdict={'size': 8, 'color': 'red'})
ax2.set_ylabel('Mximal machine workload', fontdict={'size': 8, 'color': 'red'})
ax2.set_xlabel('Makespan', fontdict={'size': 8, 'color': 'red'})

#plt.xlim(1000,1010)
#plt.ylim(660,670)

#plt.show()

def draw_chart(data):
    nb_row = len(data.keys())

    pos = np.arange(0.5, nb_row * 0.5 + 0.5, 0.5)

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(111)

    index = 0
    max_len = []

    for machine, operations in sorted(data.items()):
        for op in operations:
            max_len.append(op[1])
            c = random.choice(colors)
            rect = ax.barh((index * 0.5) + 0.5, op[1] - op[0], left=op[0], height=0.3, align='center',
                           edgecolor=c, color=c, alpha=0.8)

            # adding label
            width = int(rect[0].get_width())
            Str = "OP_{}".format(op[2])
            xloc = op[0] + 0.50 * width
            clr = 'black'
            align = 'center'

            yloc = rect[0].get_y() + rect[0].get_height() / 2.0
            ax.text(xloc, yloc, Str, horizontalalignment=align,
                            verticalalignment='center', color=clr, weight='bold',
                            clip_on=True)
        index += 1

    ax.set_ylim(ymin=-0.1, ymax=nb_row * 0.5 + 0.5)
    ax.grid(color='gray', linestyle=':')
    ax.set_xlim(0, max(10, max(max_len)))

    labelsx = ax.get_xticklabels()
    plt.setp(labelsx, rotation=0, fontsize=10)

    locsy, labelsy = plt.yticks(pos, data.keys())
    plt.setp(labelsy, fontsize=14)

    font = font_manager.FontProperties(size='small')
    ax.legend(loc=1, prop=font)

    ax.invert_yaxis()

    plt.title("Flexible Job Shop Solution")
    plt.savefig('gantt.svg')
    plt.show()







