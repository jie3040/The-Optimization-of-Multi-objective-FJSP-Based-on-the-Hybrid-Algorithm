import numpy as np

PATH='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset5.npz'

data = np.load(PATH)

train_x = data['train_X']
train_y = data['train_Y']
test_x = data['test_X']
test_y = data['test_Y']

np.savez('/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset5_2.npz', train_X=train_x, train_Y=train_y, test_X=test_x, test_Y=test_y)