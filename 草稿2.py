#from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
#from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
import numpy as np
import tensorflow as tf

#label=[[1,0,0,0,0,0,0,0,0,0]]

#label = tf.constant(label, dtype=tf.int32)

#label = [1]
#label = tf.reshape(label, (1,))

#label_embedding = Flatten()(Embedding(10, 864)(label))

#print(label_embedding)

#A=[[1],[2],[3]]

#A=tf.convert_to_tensor(A)

#idx = np.random.randint(0, A.shape[0], 2)

#B=A[idx]

#gpus = tf.config.list_physical_devices('GPU')

    
print(tf.__version__)
print(tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))