import preprocess
import numpy as np

path='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset_new/'

train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                length=1024,
                                                                number=2000,
                                                                normal=True,
                                                                rate=[0.5, 0.4, 0.1],
                                                                enc=False,
                                                                enc_step=28)

mask_1 = (train_Y == [0,0,0,0,0,0,0,0,1,0]).all(axis=1)

class_9_X = train_X[mask_1]
class_9_Y = train_Y[mask_1]

mask_2 = (test_Y == [0,0,0,0,0,0,0,0,1,0]).all(axis=1)

class_9_x = test_X[mask_2]
class_9_y = test_Y[mask_2]

PATH='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset4.npz'
data = np.load(PATH)

train_x = data['train_X']
train_y = data['train_Y']
test_x = data['test_X']
test_y = data['test_Y']

mask_3 = (train_y == [0,0,0,0,0,0,0,0,1,0]).all(axis=1)

class_X_except_9 = train_x[~mask_3]  # The "~" operator inverts the boolean values in the mask
class_Y_except_9 = train_y[~mask_3]

mask_4 = (test_y == [0,0,0,0,0,0,0,0,1,0]).all(axis=1)

class_x_except_9 = test_x[~mask_4]  # The "~" operator inverts the boolean values in the mask
class_y_except_9 = test_y[~mask_4]


new_train_x= np.vstack((class_X_except_9, class_9_X))
new_train_y=np.vstack((class_Y_except_9, class_9_Y))

new_test_x= np.vstack((class_x_except_9, class_9_x))
new_test_y=np.vstack((class_y_except_9, class_9_y))




mask_5=(train_y == [0,0,0,0,0,0,0,0,1,0]).all(axis=1)
class_9_x_train = train_x[mask_5][0:200]
class_9_y_train = train_y[mask_5][0:200]
new_test_x= np.vstack((class_x_except_9, class_9_x_train))
new_test_y=np.vstack((class_y_except_9, class_9_y_train))


np.savez('/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset5.npz', train_X=train_x, train_Y=train_y, valid_X=valid_X, valid_Y=valid_Y, test_X=new_test_x, test_Y=new_test_y)