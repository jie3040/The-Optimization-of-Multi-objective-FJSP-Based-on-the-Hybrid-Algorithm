#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
import preprocess

#path='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset_new/'
#train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                #length=1024,
                                                                #number=2000,
                                                                #normal=True,
                                                                #rate=[0.5, 0.4, 0.1],
                                                                #enc=False,
                                                                #enc_step=28)

#np.savez('/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset.npz', train_X=train_X, train_Y=train_Y, valid_X=valid_X, valid_Y=valid_Y, test_X=test_X, test_Y=test_Y)

PATH='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset5.npz'
data = np.load(PATH)

train_X = data['train_X']
train_Y = data['train_Y']
test_X = data['test_X']
test_Y = data['test_Y']


X_train=train_X
Y_train=np.argmax(train_Y, axis=1)

X_test=test_X
Y_test=np.argmax(test_Y, axis=1)

#print(Y_train)
#Feature Selection
estimator = SVR(kernel='linear')
#selector = RFE(estimator, n_features_to_select=30, step=1)
#selector_1 = selector.fit(X_train, Y_train)
#selector_2 = selector.fit(X_test, Y_test)

#selector_1.support_
#selector_1.ranking_

#selector_2.support_
#selector_2.ranking_

#X_train = selector_1.transform(X_train)
#X_test = selector_2.transform(X_test)

#Applying PCA #Please turn off when applying KPCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 37)
#X_train = pca.fit_transform(X_optTrain)
#X_test = pca.transform(X_optTest)
#explained_variance = pca.explained_variance_ratio_

#Applying Kernel PCA #Please Turn Off when applying PCA
#from sklearn.decomposition import KernelPCA
#kpca = KernelPCA(n_components =32, kernel = 'rbf')
#X_train = kpca.fit_transform(X_train)
#X_test = kpca.transform(X_test)

#Applying LDA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#lda = LDA(n_components = 9)
#X_train = lda.fit_transform(X_optTrain, Y_train)
#X_test = lda.fit_transform(X_optTest, Y_test)


#Fitting SVM to the Training Set
from sklearn.svm import SVC
classifier = SVC(C=100, gamma=0.001,kernel = 'rbf', random_state =0) #kernel can be changed to linear for linear SVM
#classifier.fit(X_train, Y_train)
classifier.fit(X_train, Y_train)


#Fitting Decision Tree to the Training Set
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, Y_train)

#Predicting the Test Set Results

#Y_pred = classifier.predict(X_test)
Y_pred = classifier.predict(X_test)


#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

print(cm)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_pred)

print("Test Accuracy:", accuracy)

#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#parameters = [ {'C': [1,10,100], 'kernel': ['rbf'], 'gamma': [0.0001,0.001,0.01]}] 
#grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
#grid_search = grid_search.fit(X_train, Y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

#print(best_accuracy)
#print(best_parameters)