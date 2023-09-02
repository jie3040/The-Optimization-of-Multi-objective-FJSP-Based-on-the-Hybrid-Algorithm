from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers import LeakyReLU, PReLU, ELU, ReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import save_model
import matplotlib.pyplot as plt
import preprocess
import sys
from keras.models import load_model

import numpy as np

class CGAN():
    def __init__(self):
        self.data_lenth=864
        self.sample_shape=(self.data_lenth,)
        self.num_classes=10
        self.latent_dim = 100
        
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        
         # Build the generator
        self.generator = self.build_generator()
        
         # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        
        g_sample = self.generator([noise, label])
        
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        
        # The discriminator takes generated images as input and determines validity
        # and the label of that image
        
        valid = self.discriminator([g_sample,label])
        
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        
    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.sample_shape), activation='tanh'))
        model.add(Reshape(self.sample_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        
        model_input = multiply([noise, label_embedding])
        
        sample = model(model_input)

        return Model([noise, label], sample)
    
    def build_discriminator(self):

        model = Sequential()
        
        model.add(Dense(512, input_dim=np.prod(self.sample_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        
        sample = Input(shape=self.sample_shape)
        label = Input(shape=(1,), dtype='int32')
        
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.sample_shape))(label))
        
        flat_sample = Flatten()(sample)
        model_input = multiply([flat_sample, label_embedding])
        
        validity = model(model_input)

        return Model([sample, label], validity)
    
    def train(self, epochs, batch_size=128, sample_interval=50):
        
        path='/home/liaowenjie/myfolder/GAN_for_UFD/dataset/'
        
        train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                length=864,
                                                                number=1000,
                                                                normal=False,
                                                                rate=[0.5, 0.25, 0.25],
                                                                enc=False,
                                                                enc_step=28)
        
        # Adversarial ground truths
        train_Y=np.argmax(train_Y, axis=1)
        train_Y=train_Y.reshape((-1,1))
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of samples
            idx = np.random.randint(0, train_X.shape[0], batch_size)
            samples, labels = train_X[idx], train_Y[idx]
            
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Generate a batch of new images
            
            gen_samples = self.generator.predict([noise,labels])
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([samples,labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_samples,labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            # Condition on labels
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            
            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)
            
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            # If at save interval => save generated image samples
            
if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=20000, batch_size=32, sample_interval=200)
    
    # Save the generator model
    save_model(cgan.generator, '/home/liaowenjie/myfolder/GAN_for_UFD/generator_model.h5')
    
    # Save the discriminator model
    save_model(cgan.discriminator, '/home/liaowenjie/myfolder/GAN_for_UFD/discriminator_model.h5')
    



 
    

        


            

        
        
        
        
        
        