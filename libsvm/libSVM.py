import sys
from numpy import *
from svm import *
from os import listdir
from plattSMO import PlattSMO
import pickle
import preprocess
from keras.models import save_model
import pickle
import os
import joblib
    
import numpy as np
class LibSVM:
    def __init__(self,data=[],label=[],C=0,toler=0,maxIter=0,**kernelargs):
        self.classlabel = unique(label)
        self.classNum = len(self.classlabel)
        self.classfyNum = (self.classNum * (self.classNum-1))/2
        self.classfy = []
        self.dataSet={}
        self.kernelargs = kernelargs
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        m = shape(data)[0]
        for i in range(m):
            if label[i] not in self.dataSet.keys():
                self.dataSet[label[i]] = []
                self.dataSet[label[i]].append(data[i][:])
            else:
                self.dataSet[label[i]].append(data[i][:])
            
    def train(self):
        num = self.classNum
        for i in range(num):
            for j in range(i+1,num):
                data = []
                label = [1.0]*shape(self.dataSet[self.classlabel[i]])[0]
                label.extend([-1.0]*shape(self.dataSet[self.classlabel[j]])[0])
                data.extend(self.dataSet[self.classlabel[i]])
                data.extend(self.dataSet[self.classlabel[j]])
                svm = PlattSMO(array(data),array(label),self.C,self.toler,self.maxIter,**self.kernelargs)
                svm.smoP()
                self.classfy.append(svm)
        self.dataSet = None
    
    def predict(self,data,label):
        m = shape(data)[0]
        num = self.classNum
        classlabel = []
        count = 0.0
        for n in range(m):
            result = [0] * num
            index = -1
            for i in range(num):
                for j in range(i + 1, num):
                    index += 1
                    s = self.classfy[index]
                    t = s.predict([data[n]])[0]
                    if t > 0.0:
                        result[i] +=1
                    else:
                        result[j] +=1
            classlabel.append(result.index(max(result))) 
            if classlabel[-1] != label[n]:
                count +=1
                print(label[n],classlabel[n])
        #print classlabel
        print("error rate:",count / m)
        return classlabel

def main():
    path='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset_1HP/'
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                length=1024,
                                                                number=1000,
                                                                normal=True,
                                                                rate=[0.5, 0.25, 0.25],
                                                                enc=False,
                                                                enc_step=28)
    train_Y=np.argmax(train_Y, axis=1)
    
    test_Y=np.argmax(test_Y, axis=1)
    
    
    
    data,label = train_X,train_Y
    
    test,testlabel=test_X,test_Y
    
    svm = LibSVM(data, label, 100, 0.0001, 10000, name='rbf', theta=20)
    svm.train()
    
    joblib.dump(svm, 'svm_model.joblib')
    svm.predict(test,testlabel)
    #loaded_svm = joblib.load('svm_model.joblib')
    #loaded_svm.predict(test,testlabel)
    
    #loaded_svm.predict(test,testlabel)

if __name__ == "__main__":
    sys.exit(main())
            
                       
                    
            
        
        
            