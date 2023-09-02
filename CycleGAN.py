from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate,multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D,Embedding
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
#from data_loader import DataLoader
import numpy as np
import os
import preprocess
import tensorflow as tf
from keras.models import save_model

class CycleGAN():
    def __init__(self):
        self.data_lenth=1024
        self.sample_shape=(self.data_lenth,)
        self.num_classes=10
        self.latent_dim = 100
        
        self.dataset_name='fualt_diagnosis'
        
        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss
        
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the discriminators
        
        self.d_1 = self.build_discriminator() # d_1 is for normal/healthy samples
        self.d_2 = self.build_discriminator() # d_2 is for fault samples
        
        self.d_1.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_2.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        
        # Build the generators
        
        self.g_AB = self.build_generator() #g_AB is for generating fault samples
        self.g_BA = self.build_generator() #g_BA is for generating healthy samples
        
        # Input samples from both domains
        # A is healthy domain, B is faulty domain
        
        sample_A=Input(shape=self.sample_shape)
        sample_B=Input(shape=self.sample_shape)
        
        
        # Translate samples to the other domain
        label_B = Input(shape=(1,))
        fake_B=self.g_AB([sample_A,label_B])
        
        label_A= Input(shape=(1,))
        fake_A=self.g_BA([sample_B,label_A])
        
        # Translate samples back to original domain
        reconstr_A=self.g_BA([fake_B,label_A])
        reconstr_B=self.g_AB([fake_A,label_B])
        
        # Identity mapping of samples
        
        sample_A_id=self.g_BA([sample_A,label_A])
        sample_B_id=self.g_AB([sample_B,label_B])
        
        # For the combined model we will only train the generators
        
        self.d_1.trainable = False
        self.d_2.trainable = False
        
        # Discriminators determines validity of translated samples, resonstr_samples and sample_X_id
        
        valid_A_for_fake_A = self.d_1([fake_A,label_A])
        valid_B_for_fake_B = self.d_2([fake_B,label_B])
        
        valid_A_for_reconstr_A=self.d_1([reconstr_A,label_A])
        valid_B_for_reconstr_B=self.d_2([reconstr_B,label_B])
        
        valid_A_for_sample_A_id=self.d_1([sample_A_id,label_A])
        valid_B_for_sample_B_id=self.d_2([sample_B_id,label_B])
        
        # Combined model trains generators to fool discriminators
        
        self.combined = Model(inputs=[[sample_A, label_A],[sample_B,label_B]],
                              outputs=[valid_A_for_fake_A, valid_B_for_fake_B,valid_A_for_reconstr_A, valid_B_for_reconstr_B,valid_A_for_sample_A_id, valid_B_for_sample_B_id])
        
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy',
                                    'binary_crossentropy', 'binary_crossentropy',
                                    'binary_crossentropy', 'binary_crossentropy'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)
        
        
    def build_discriminator(self):
        
        model = Sequential()
        model.add(Dense(512, input_dim=np.prod(self.sample_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(32)) 
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
    
    def build_generator(self):
        """U-Net Generator"""
        
        
        def enDenselayer(layer_nodes,layer_input):
            
            d=Dense(layer_nodes)(layer_input)
            d=LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d
        
        def deDenselayer(layer_input,layer_nodes,skip_input,dropout_rate=0):
            
            u=Dense(layer_nodes)(layer_input)
            u=LeakyReLU(alpha=0.2)(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u
            
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.data_lenth)(label))
        
        model_input=Input(shape=self.sample_shape)
        d0 = multiply([model_input, label_embedding])
        
        #Downsampling
        d1=enDenselayer(512,d0)
        d2=enDenselayer(256,d1)
        d3=enDenselayer(128,d2)
        d4=enDenselayer(64,d3)
        d5=enDenselayer(32,d4)
        
        #Upsampling
        u0=deDenselayer(d5,64,d4)
        u1=deDenselayer(u0,128,d3)
        u2=deDenselayer(u1,256,d2)
        u3=deDenselayer(u2,512,d1)
        
        output=Dense(np.prod(self.sample_shape), activation='tanh')(u3)
        output_sample=Reshape(self.sample_shape)(output)
        
        return Model([model_input,label], output_sample)
    
    def train(self, epochs, batch_size=1): 
        
        start_time = datetime.datetime.now()
        
        # Adversarial loss ground truths
        valid = np.ones((batch_size,1) )
        fake = np.zeros((batch_size,1) )
        
        path='/home/liaowenjie/myfolder/GAN_for_UFD/dataset/'
        
        
        #train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess.prepro(d_path=path,
                                                                #length=1024,
                                                                #number=1000,
                                                                #normal=False,
                                                                #rate=[0.5, 0.25, 0.25],
                                                                #enc=False,
                                                                #enc_step=28)
        

        PATH='/home/liaowenjie/anaconda3/myfolder/GAN_for_UFD/dataset5.npz'
        data = np.load(PATH)

        train_X = data['train_X']
        train_Y = data['train_Y']
        test_X = data['test_X']
        test_Y = data['test_Y']


        
        train_Y=np.argmax(train_Y, axis=1)
        train_Y=train_Y.reshape((-1,1))
        
        
        domain_A_train_X=[]
        domain_A_train_Y=[]
        domain_B_train_X=[]
        domain_B_train_Y=[]
        
        for i in range(train_Y.shape[0]):
            
            if train_Y[i] == 9:
                
                domain_A_train_X.append(train_X[i])
                domain_A_train_Y.append(train_Y[i])
                
            else:
                
                domain_B_train_X.append(train_X[i])
                domain_B_train_Y.append(train_Y[i])
                
        domain_A_train_X=np.array(domain_A_train_X)
        domain_A_train_Y=np.array(domain_A_train_Y)
        domain_B_train_X=np.array(domain_B_train_X)
        domain_B_train_Y=np.array(domain_B_train_Y)
        
        #print(domain_B_train_X.shape)   
             
        num_batches=int(domain_A_train_X.shape[0]/batch_size)
        
        for epoch in range(epochs):
            
            for batch_i in range(num_batches):
                
                # Select a batch of samples from domian A
                
                start_i =batch_i * batch_size
                end_i=(batch_i + 1) * batch_size
                
                batch_samples_A = domain_A_train_X[start_i:end_i]
                batch_labels_A= domain_A_train_Y[start_i:end_i]

                for class_i in range(9):

                
                    # Select a batch of  samples from domian B

                    Start_i=start_i+(class_i*num_batches)
                    End_i=end_i+(class_i*num_batches)


                    class_1_sample=domain_B_train_X[Start_i:End_i]
                    class_1_label=domain_B_train_Y[Start_i:End_i]
                
                
                    batch_samples_B, batch_labels_B = class_1_sample,class_1_label
                
                
                    #print(batch_samples_A.shape)
                    #print(batch_samples_B.shape)
                    #print(batch_labels_A.shape)
                    #print(batch_labels_B.shape)
                    # ----------------------
                    #  Train Discriminators
                    # ----------------------
                
                    # Translate samples to opposite domain
                
                    fake_B= self.g_AB.predict([batch_samples_A,batch_labels_B])
                    fake_A=self.g_BA.predict([batch_samples_B,batch_labels_A])
                    # Train the discriminators
                
                    dA_loss_real=self.d_1.train_on_batch([batch_samples_A,batch_labels_A],valid)
                    dA_loss_fake=self.d_1.train_on_batch([fake_A,batch_labels_A],fake)
                    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
                
                    dB_loss_real=self.d_2.train_on_batch([batch_samples_B,batch_labels_B],valid)
                    dB_loss_fake=self.d_2.train_on_batch([fake_B,batch_labels_B],fake)
                    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
                
                    # Total disciminator loss
                    d_loss = 0.5 * np.add(dA_loss, dB_loss)
                
                    # ------------------
                    #  Train Generators
                    # ------------------
                
                    g_loss = self.combined.train_on_batch([[batch_samples_A,batch_labels_A],[batch_samples_B,batch_labels_B]],
                                                          [valid, valid,
                                                           valid, valid,
                                                           valid, valid])
                    #print(g_loss)
                    elapsed_time = datetime.datetime.now() - start_time
                
                    # Plot the progress
                
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                            % ( epoch, epochs,
                                                                                batch_i, num_batches,
                                                                                d_loss[0], 100*d_loss[1],
                                                                                g_loss[0],
                                                                                np.mean(g_loss[1:3]),
                                                                                np.mean(g_loss[3:5]),
                                                                                np.mean(g_loss[5:7]),
                                                                                elapsed_time))


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=200, batch_size=1)
    
    # Save the discriminator model
    save_model(gan.d_1, '/home/liaowenjie/myfolder/GAN_for_UFD/CycleGAN_d_1_model.h5')
    
    save_model(gan.d_2, '/home/liaowenjie/myfolder/GAN_for_UFD/CycleGAN_d_2_model.h5')
            
    # Save the generator model      
      
    save_model(gan.g_AB, '/home/liaowenjie/myfolder/GAN_for_UFD/CycleGAN_g_AB_model.h5')
    
    save_model(gan.g_BA, '/home/liaowenjie/myfolder/GAN_for_UFD/CycleGAN_g_BA_model.h5')       
    
        
    
        
        
            
        
        
        
    
    
    
    
    
        
        
        