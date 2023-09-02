import sys
from numpy import *
#from math import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import preprocess
def loadDataSet(filename):
    data = pd.read_excel(filename)  # Read Excel file using pandas
    labels = data.iloc[:, -1]  # Extract labels from the last column
    data = data.iloc[:, :-1]  # Extract data from all other columns except the last one
    return data, labels

data, labels = loadDataSet('cwstep2.xlsx')

#print(labels)

def selectJrand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j

#print(selectJrand(5,6))

def clipAlpha(a_j,H,L):
    if a_j > H:
        a_j = H
    if L > a_j:
        a_j = L
    return a_j

def smoSimple(data, label, C, toler, maxIter,classifier_i):
    dataMatrix = np.array(data)
    labelMatrix = np.array(label).reshape(-1, 1)
    m, n = dataMatrix.shape
    alpha = np.zeros((m, 1))
    b = 0.0
    iter = 0

    def computeError(i):
        fxi = float(np.dot((alpha * labelMatrix).T, np.dot(dataMatrix, dataMatrix[i, :].T))) + b
        Ei = fxi - float(labelMatrix[i])
        return Ei
    
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            Ei = computeError(i)
            if (labelMatrix[i] * Ei < -toler and alpha[i] < C) or (labelMatrix[i] * Ei > toler and alpha[i] > 0):
                j = selectJrand(i, m)
                Ej = computeError(j)
                alphaIOld, alphaJOld = alpha[i].copy(), alpha[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[i] + alpha[j] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    continue
                eta = 2.0 * np.dot(dataMatrix[i, :], dataMatrix[j, :].T) - np.dot(dataMatrix[i, :], dataMatrix[i, :].T) \
                      - np.dot(dataMatrix[j, :], dataMatrix[j, :].T)
                if eta >= 0:
                    continue

                alpha[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alpha[j] = clipAlpha(alpha[j], H, L)

                if np.abs(alpha[j] - alphaJOld) < 0.00001:
                    continue
                alpha[i] += labelMatrix[j] * labelMatrix[i] * (alphaJOld - alpha[j])
                b1 = b - Ei - labelMatrix[i] * (alpha[i] - alphaIOld) * np.dot(dataMatrix[i, :], dataMatrix[i, :].T) \
                     - labelMatrix[j] * (alpha[j] - alphaJOld) * np.dot(dataMatrix[i, :], dataMatrix[j, :].T)
                b2 = b - Ej - labelMatrix[i] * (alpha[i] - alphaIOld) * np.dot(dataMatrix[i, :], dataMatrix[j, :].T) \
                     - labelMatrix[j] * (alpha[j] - alphaJOld) * np.dot(dataMatrix[j, :], dataMatrix[j, :].T)
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1

        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("classifier %d :iteration number: %d" %(classifier_i, iter))
    return b, alpha

import numpy as np



def train_by_smosimple(train_x_for_label_1, train_x_for_label_negative_1, classifier_i,C=0.01,toler=0.001,maxIter=200 ):
    
    train_y_for_label_1=[1]*len(train_x_for_label_1)
    train_y_for_label_negative_1=[-1]*len(train_x_for_label_negative_1)
    
    train_x=train_x_for_label_1+train_x_for_label_negative_1
    train_y=train_y_for_label_1+train_y_for_label_negative_1
    
    b,alpha = smoSimple(train_x,train_y,C,toler,maxIter,classifier_i)
    
    return b,alpha
 
def predict(x_test, classifiers):
    
    X = np.array(x_test)
    num_classes = len(classifiers)
    num_samples = X.shape[0]
    decision_values = np.zeros((num_samples, num_classes))
    
    # Compute decision values for each classifier
    for class_idx, classifier in enumerate(classifiers):
        alpha, dataMatrix, b = classifier
        for i in range(num_samples):
            decision_values[i, class_idx] = np.dot(alpha.T, np.dot(dataMatrix, X[i].T)) + b
            
    predicted_labels = np.argmax(decision_values, axis=1)   
    
    return predicted_labels



def main():
    path='/home/liaowenjie/myfolder/GAN_for_UFD/dataset/'
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                length=164,
                                                                number=100,
                                                                normal=True,
                                                                rate=[0.5, 0.25, 0.25],
                                                                enc=False,
                                                                enc_step=28)
    
    train_Y=np.argmax(train_Y, axis=1)
    train_Y=train_Y.reshape((-1,1))
    test_Y=np.argmax(test_Y, axis=1)
    test_Y=test_Y.reshape((-1,1))
    
    
    train_x_normal=[]
    train_x_fault_1=[]
    train_x_fault_2=[]
    train_x_fault_3=[]
    train_x_fault_4=[]
    train_x_fault_5=[]
    train_x_fault_6=[]
    train_x_fault_7=[]
    train_x_fault_8=[]
    train_x_fault_9=[]
    
    for i in range(train_Y.shape[0]):
        
        if train_Y[i] == 9:
            train_x_normal.append(train_X[i])
        
        elif train_Y[i] == 0:
            train_x_fault_1.append(train_X[i])
            
        elif train_Y[i] == 1:
            train_x_fault_2.append(train_X[i])
            
        elif train_Y[i] == 2:
            train_x_fault_3.append(train_X[i])
            
        elif train_Y[i] == 3:
            train_x_fault_4.append(train_X[i])    
            
        elif train_Y[i] == 4:
            train_x_fault_5.append(train_X[i])
            
        elif train_Y[i] == 5:
            train_x_fault_6.append(train_X[i])
            
        elif train_Y[i] == 6:
            train_x_fault_7.append(train_X[i])
            
        elif train_Y[i] == 7:
            train_x_fault_8.append(train_X[i])
            
        else:
            train_x_fault_9.append(train_X[i])
    
    train_x_fault=train_x_fault_1+train_x_fault_2+train_x_fault_3+train_x_fault_4+train_x_fault_5+train_x_fault_6+train_x_fault_7+train_x_fault_8+train_x_fault_9     
    


    b_1,alpha_1 = train_by_smosimple(train_x_fault_1,train_x_normal,1)
    b_2,alpha_2 = train_by_smosimple(train_x_fault_2,train_x_normal,2)
    b_3,alpha_3 = train_by_smosimple(train_x_fault_3,train_x_normal,3)
    b_4,alpha_4 = train_by_smosimple(train_x_fault_4,train_x_normal,4)
    b_5,alpha_5 = train_by_smosimple(train_x_fault_5,train_x_normal,5)
    b_6,alpha_6 = train_by_smosimple(train_x_fault_6,train_x_normal,6)
    b_7,alpha_7 = train_by_smosimple(train_x_fault_7,train_x_normal,7)
    b_8,alpha_8 = train_by_smosimple(train_x_fault_8,train_x_normal,8)
    b_9,alpha_9 = train_by_smosimple(train_x_fault_9,train_x_normal,9)
    b_10,alpha_10 = train_by_smosimple(train_x_normal,train_x_fault,10)
    
    
    print(b_1,b_2,b_3,b_4,b_5,b_6,b_7,b_8,b_9,b_10)
    print(alpha_1,alpha_2,alpha_3,alpha_4,alpha_5,alpha_6,alpha_7,alpha_8,alpha_9,alpha_10)
    
    classifiers=[
        (alpha_1,np.array(train_x_fault_1+train_x_normal),b_1),
        (alpha_2,np.array(train_x_fault_2+train_x_normal),b_2),
        (alpha_3,np.array(train_x_fault_3+train_x_normal),b_3),
        (alpha_4,np.array(train_x_fault_4+train_x_normal),b_4),
        (alpha_5,np.array(train_x_fault_5+train_x_normal),b_5),
        (alpha_6,np.array(train_x_fault_6+train_x_normal),b_6),
        (alpha_7,np.array(train_x_fault_7+train_x_normal),b_7),
        (alpha_8,np.array(train_x_fault_8+train_x_normal),b_8),
        (alpha_9,np.array(train_x_fault_9+train_x_normal),b_9),
        (alpha_10,np.array(train_x_normal+train_x_fault),b_10)
    ]

    predictions = predict(test_X, classifiers)
    
    #print(len(test_X))
    #print(len(test_Y))
    print(predictions)
    test_Y=[item for sublist in test_Y for item in sublist]
    print(test_Y)
    accuracy = accuracy_score(test_Y, predictions)
    print(accuracy)
    

if __name__ == '__main__':
    sys.exit(main())



