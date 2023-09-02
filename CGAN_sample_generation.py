from keras.models import load_model
import preprocess
import numpy as np

generator = load_model('/home/liaowenjie/myfolder/GAN_for_UFD/generator_model.h5')

path='/home/liaowenjie/myfolder/GAN_for_UFD/dataset/'

train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                length=864,
                                                                number=1000,
                                                                normal=False,
                                                                rate=[0.5, 0.25, 0.25],
                                                                enc=False,
                                                                enc_step=28)

batch_size=128
latent_dim = 100


idx = np.random.randint(0, train_X.shape[0], batch_size)
train_Y=np.argmax(train_Y, axis=1)
train_Y=train_Y.reshape((-1,1))
samples, labels = train_X[idx], train_Y[idx]
noise = np.random.normal(0, 1, (batch_size, latent_dim))
gen_samples = generator.predict([noise,labels])

print(gen_samples)