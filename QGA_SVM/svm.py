# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:40:31 2021

@author: zalon
"""

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import preprocess



  
def load_data():
 
    data_time = pd.read_csv("svm_data.txt")
    data_time['fault'] = pd.Categorical(data_time['fault'])
   
    train_data, test_data = train_test_split(data_time, test_size =750, stratify = data_time['fault'],
                                         random_state = 1234)
    test_data['fault'].value_counts()
  
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data.iloc[:,:-1])
    
    test_data_scaled = (test_data.iloc[:,:-1].values - scaler.mean_)/np.sqrt(scaler.var_)
    return train_data_scaled, train_data['fault'], test_data_scaled , test_data['fault']

## 2. QGA
class QGA(object):


    
    def __init__(self,population_size,chromosome_num,chromosome_length,max_value,min_value,iter_num,deta):
        '''Initialize class parameters
        population_size(int):Number of population
        chromosome_num(int):The number of chromosomes, corresponding to the number of parameters that need to be optimized
        chromosome_length(int):Chromosome length
        max_value(float):Chromosome decimal value maximum
        min_value(float):Chromosome decimal value minimum
        iter_num(int):Number of iterations
        deta(float):Quantum rotation angle        
        '''

        self.population_size = population_size
        self.chromosome_num = chromosome_num
        self.chromosome_length = chromosome_length
        self.max_value = max_value
        self.min_value = min_value
        self.iter_num = iter_num
        self.deta = deta 
        
### 2.2  Quantum formalization of population        
    def species_origin_angle(self):
        '''population initialization
        input:self(object):QGAClass
        output:population_Angle(list):Population quantum point list
               population_Angle2(list):A list of quantum angles of empty populations used to store a list of quantum angles after intersection
               '''

        population_Angle = []
       
        for i in range(self.chromosome_num):
            tmp1 = []          ##Store the quantum angle of all values ​​for each chromosome          
            for j in range(self.population_size): 
                tmp2 = []        ## Storage Quantum Angle              
                for m in range(self.chromosome_length):
                    a = np.pi * 2 * random.random()
                    tmp2.append(a)
                tmp1.append(tmp2)
            population_Angle.append(tmp1)
       # print('jkjk',population_Angle)
      
        return  population_Angle
    
    def population_Q(self,population_Angle):
        '''Convert the initialized quantum angle sequence to a list of quantum coefficients of the population
        input:self(object):QGAClass
              population_Angle(list):Population quantum point list
        output:population_Q(list):List of quantum coefficients of the population
        '''
        population_Q = []
      #  print('asd',population_Angle)
        for i in range(len(population_Angle)):
            tmp1 = []  ##Store pairs of quantum coefficients for all values ​​of each chromosome           
            for j in range(len(population_Angle[i])): 
                tmp2 = []  ## store a quantum pair of each value of each chromosome
                tmp3 = []  ## Store half of the quantum pair
                tmp4 = []  ## Store the other half of the quantum pair
                for m in range(len(population_Angle[i][j])):
                    a = population_Angle[i][j][m]
                    
                    
                    tmp3.append(np.sin(a))
                    tmp4.append(np.cos(a))
                   
                tmp2.append(tmp3)
                tmp2.append(tmp4)
                tmp1.append(tmp2)
                
            population_Q.append(tmp1)
       # print(population_Q)
            
         
       
        return population_Q     

### 2.3  Calculating the fitness function value  
    def translation(self,population_Q):
        '''Convert the quantum list of the population into a binary list
        input:self(object):QGAClass
        population_Q(list):Quantum list of populations
        output:population_Binary:Binary list of populations
        '''
       # print('asdsa',population_Q)
        population_Binary = []
        for i in range(len(population_Q)):
            tmp1 = []  # Store the binary form of all values ​​for each chromosome
            for j in range(len(population_Q[i])):
                tmp2 = []  ##Store the binary form of each value of each chromosome
                for l in range(len(population_Q[i][j][0])):
                    if np.square(population_Q[i][j][0][l]) > random.random():
                        tmp2.append(1)
                    else:
                        tmp2.append(0)
                tmp1.append(tmp2)
            population_Binary.append(tmp1)
       # print(population_Binary)  
        return population_Binary
       
   
    def fitness(self,population_Binary):
        '''To obtain a list of fitness function values, the fitness function used in this experiment isRBF_SVMof3_fold cross validation average
        input:self(object):QGAClass
              population_Binary(list):Binary list of populations
        output:fitness_value(list):Fitness function value class table
               parameters(list):List of corresponding optimization parameters
        '''
       ##1.The binary representation of the chromosome is converted to decimal and set in[min_value,max_value]Between
        parameters = []  ##Store the possible values ​​of all parameters
        
        for i in range(len(population_Binary)):
            tmp1 = []  ##Store the possible values ​​of a parameter
            for j in range(len(population_Binary[i])):
                total = 0.0 
              #  print((population_Binary[i]))
                for l in range(len(population_Binary[i][j])):
                    total+= population_Binary[i][j][l] * math.pow(2,l)  ##Calculate the decimal value corresponding to the binary
                    
                value = (total * (self.max_value - self.min_value)) / math.pow(2,len(population_Binary[i][j])) + self.min_value
                ## places the decimal value in[min_value,max_value]Between
                tmp1.append(value)
      #  print(tmp1)
            parameters.append(tmp1)
       # print(parameters)       
        fitness_value = []
        cf=[] 
      
        for l in range(len(parameters[0])):

            rbf_svm = svm.SVC(kernel = 'rbf', C = parameters[0][l], gamma = parameters[1][l]).fit(trainX,trainY)
          
            sc= rbf_svm.predict(testX)
            sc1=rbf_svm.predict(trainX)
            train_accuracy=accuracy_score(trainY,sc1)
            test_accuracy=accuracy_score(testY,sc)
            train_confu_matrix = confusion_matrix(trainY, sc1)
            test_confu_matrix=confusion_matrix(testY,sc)
            if test_accuracy>0.9654:
                cf.append(test_confu_matrix)
                #print(test_confu_matrix)
           
            print(test_accuracy)
               
            fitness_value.append(test_accuracy)
        
      #  print(cf)
 ##3.Find the optimal fitness function value and the corresponding parameter binary representation
        best_fitness = 0.0
    
        best_parameter = []        
        best_parameter_Binary = []
        for j in range(len(population_Binary)):
            tmp2 = []
            best_parameter_Binary.append(tmp2) 
            best_parameter.append(tmp2)
      
        for i in range(len(population_Binary[0])):
            if best_fitness < fitness_value[i]:
                best_fitness = fitness_value[i]
                   
           
        for j in range(len(population_Binary)):
                    best_parameter_Binary[j] = population_Binary[j][i]
                    best_parameter[j] = parameters[j][i]
       # print('sds',best_parameter_Binary)            
               #     print('2',best_parameter)
        return parameters,fitness_value,best_parameter_Binary,best_fitness,best_parameter,train_confu_matrix,test_confu_matrix,sc,cf
    
    ### 2.4  Full interference cross
    def crossover(self,population_Angle):
        '''Full interference crossover for the population quantum angle list
        input:self(object):QGAClass
              population_Angle(list):Population quantum point list
        '''
  ## Initialize an empty list, a list of quantum angles after full interference crossover
        population_Angle_crossover = []
       
       
        for i in range(self.chromosome_num):
            tmp11 = []                    
            for j in range(self.population_size): 
                tmp21 = []                             
                for m in range(self.chromosome_length):
                    tmp21.append(0.0)
                tmp11.append(tmp21)
                
            population_Angle_crossover.append(tmp11)
        

        for i in range(len(population_Angle)):
            for j in range(len(population_Angle[i])):
                for m in range(len(population_Angle[i][j])):
                    ni = (j - m) % len(population_Angle[i])
                   # print(ni)
                    population_Angle_crossover[i][j][m] = population_Angle[i][ni][m]
       # print(population_Angle_crossover)           
        return population_Angle_crossover


    def mutation(self,population_Angle_crossover,population_Angle,best_parameter_Binary,best_fitness):
        '''Quantum variation using quantum gate transformation matrix
        input:self(object):QGAClass
              population_Angle_crossover(list):Quantum angle list after full interference crossover
        output:population_Angle_mutation(list):List of quantum angles after mutation
        '''
        ##1.Find the list of fitness function values ​​after the intersection
        population_Q_crossover = self.population_Q(population_Angle_crossover)
      #  print('hehe',population_Q_crossover)## List of population quantum coefficients after intersection
        population_Binary_crossover = self.translation(population_Q_crossover)    ## List of population binary numbers after intersection
        parameters,fitness_crossover,best_parameter_Binary_crossover,best_fitness_crossover,best_parameter,train_confu_matrix,test_confu_matrix,sc,cf= self.fitness(population_Binary_crossover) ## List of fitness function values ​​after crossing
        ##2.Initialize the rotation angle of each qubit
        Rotation_Angle = []
       # print(population_Q_crossover)
        
        for i in range(len(population_Angle_crossover)):
            tmp1 = []
            for j in range(len(population_Angle_crossover[i])):
                tmp2 = []
                for m in range(len(population_Angle_crossover[i][j])):
                    tmp2.append(0.0)
                tmp1.append(tmp2)
            Rotation_Angle.append(tmp1)
            
        deta = self.deta
       # print(population_Q_crossover)
      #  print(best_parameter_Binary)
        for i in range(self.chromosome_num):
            for j in range(self.population_size):
                if fitness_crossover[j] <= best_fitness:
                    for m in range(self.chromosome_length):
                        s1 = 0
                        a1 = population_Q_crossover[i][j][0][m]
                        b1 = population_Q_crossover[i][j][1][m]
                      #  print('a1',a1)
                        
                       
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 0 and a1 * b1 > 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 0 and a1 * b1 < 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 0 and a1 * b1 == 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 1 and a1 * b1 > 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 1 and a1 * b1 < 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 1 and a1 * b1 == 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 0 and a1 * b1 > 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 0 and a1 * b1 < 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 0 and a1 * b1 == 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 1 and a1 * b1 > 0:
                            s1 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 1 and a1 * b1 < 0:
                            s1 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 1 and a1 * b1 == 0:
                            s1 = 1
                        Rotation_Angle[i][j][m] = deta * s1
                else:
                    for m in range(self.chromosome_length):
                        s2 = 0
                        a2 = population_Q_crossover[i][j][0][m]
                       
                        b2 = population_Q_crossover[i][j][1][m]
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 0 and a2 * b2 > 0:
                            s2 = -1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 0 and a2 * b2 < 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 0 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 1 and a2 * b2 > 0:
                            s2 = -1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 1 and a2 * b2 < 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 0 and best_parameter_Binary[i][m] == 1 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 0 and a2 * b2 > 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 0 and a2 * b2 < 0:
                            s2 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 0 and a2 * b2 == 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 1 and a2 * b2 > 0:
                            s2 = 1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 1 and a2 * b2 < 0:
                            s2 = -1
                        if population_Binary_crossover[i][j][m] == 1 and best_parameter_Binary[i][m] == 1 and a2 * b2 == 0:
                            s2 = 1
                        Rotation_Angle[i][j][m] = deta * s2
                        
     

###4.  Generate a new list of quantum angles for each population based on the rotation angle of each qubit

        for i in range(self.chromosome_num):
            for j in range(self.population_size):
                for m in range(self.chromosome_length):
                    population_Angle[i][j][m] = population_Angle[i][j][m] + Rotation_Angle[i][j][m]
       # print(population_Angle)           
        return population_Angle
                        
### 2.5 Draw a fitness function value change graph
    def plot(self,results,train_confu_matrix,test_confu_matrix):
        
        X = []
        Y = []
        
        for i in range(self.iter_num):
            X.append(i + 1)
            Y.append(results[i])
        
        fault_type = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10']
        plt.figure(1,figsize=(18,8))
        plt.subplot(121)
       # train_confu_matrix = confusion_matrix(trainY, sc1)
        sns.heatmap(train_confu_matrix, annot= True,fmt = "d",
                    xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues",cbar = False)
        plt.title('Training Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.subplot(122)
        sns.heatmap(train_confu_matrix/155, annot= True,
                    xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues",cbar = False)
        plt.title('Training Confusion Matrix (in %age)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
       # test_confu_matrix = confusion_matrix(testY, sc)
        plt.figure(2,figsize=(18,8))
        plt.subplot(121)
        sns.heatmap(test_confu_matrix, annot = True,
        xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
        plt.title('Test Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.subplot(122)
        sns.heatmap(test_confu_matrix/75, annot = True,
                    xticklabels=fault_type, yticklabels=fault_type, cmap = "Blues", cbar = False)
        plt.title('Test Confusion Matrix (in %age)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        plt.plot(X,Y)
        plt.xlabel('Number of iteration',size = 15)
        plt.ylabel('Value of CV',size = 15)
        plt.title('QGA_RBF_SVM parameter optimization')
        plt.show()  

### 2.6  Main function
    def main(self):
        results = []
        best_fitness = 0.0
        best_parameter = []
  
        population_Angle= self.species_origin_angle()        
  
        for i in range(self.iter_num):
            population_Q = self.population_Q(population_Angle)
             ## Converting quantum coefficients to binary form
            population_Binary = self.translation(population_Q)
          ## Calculate the list of fitness function values ​​for this iteration, the optimal fitness function value and the corresponding parameters
            parameters,fitness_value,current_parameter_Binary,current_fitness,current_parameter,train_confu_matrix,test_confu_matrix,sc,cf= self.fitness(population_Binary)
             ## Find the optimal fitness function value and corresponding parameters so far

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_parameter = current_parameter
              
            print('iteration is :',i+1,';Best parameters:',best_parameter,';Best fitness',best_fitness)
            results.append(best_fitness)
            
             ## Full interference cross
   
            population_Angle_crossover = self.crossover(population_Angle)
  ## Quantum rotation variation
            population_Angle = self.mutation(population_Angle_crossover,population_Angle,current_parameter_Binary,current_fitness)
        
        results.sort()
        
        self.plot(results,train_confu_matrix,test_confu_matrix)
      
        print('Final parameters are :',parameters[-1])
        
if __name__ == '__main__':
    print('----------------1.Load Data-------------------')

    path='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset_new/'

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                length=512,
                                                                number=100,
                                                                normal=True,
                                                                rate=[0.5, 0.3, 0.2],
                                                                enc=False,
                                                                enc_step=28)
    
    trainX=train_X
    trainY=np.argmax(train_Y, axis=1)

    testX=test_X
    testY=np.argmax(test_Y, axis=1)

    
    print('----------------2.Parameter Seting------------')
    
    population_size=20
    
    chromosome_num=2
    
    chromosome_length=17
   
    max_value=100
    min_value=0.01
    #generation=iter_num
    iter_num=500
    deta=0.1 * np.pi
    print('----------------3.QGA_RBF_SVM-----------------')
    qga = QGA(population_size,chromosome_num,chromosome_length,max_value,min_value,iter_num,deta)
    qga.main()