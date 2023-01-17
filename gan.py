from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout,Conv2D
from tensorflow.keras.layers import BatchNormalization,Activation,LeakyReLU,UpSampling2D
from tensorflow.keras.models import Sequential,Model

import numpy as np
import matplotlib.pyplot as plt

# Fixed
row_size = 28
col_size = 28

# Params
num_of_nodes = [128,64]
size_of_dimension = 100
size_of_kernel = 5

# Generator
generator = Sequential()
generator.add(Dense(num_of_nodes[0]*(row_size//4)*(col_size//4),
        input_dim=size_of_dimension,activation=LeakyReLU(0.2)))
generator.add(BatchNormalization())
generator.add(Reshape((row_size//4,col_size//4,num_of_nodes[0])))
generator.add(UpSampling2D())
generator.add(Conv2D(num_of_nodes[1],kernel_size=size_of_kernel,padding='same'))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1,kernel_size=size_of_kernel,padding='same',activation='tanh'))

# Params
num_of_nodes = [64,128]
size_of_stride = 2
size_of_kernel = 5
dropout_rate = 0.3

# Disciminator
discriminator = Sequential()
discriminator.add(Conv2D(num_of_nodes[0],kernel_size=size_of_kernel,
        strides=size_of_stride,input_shape=(row_size,col_size,1),padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(dropout_rate))
discriminator.add(Conv2D(num_of_nodes[1],kernel_size=size_of_kernel,
        strides=size_of_stride,padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(dropout_rate))
discriminator.add(Flatten())
discriminator.add(Dense(1,activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy',optimizer='adam')
discriminator.trainable = False

# Pipeline from Generator to Discriminator
ginput = Input(shape=(size_of_dimension,))
dis_output = discriminator(generator(ginput))

gan = Model(ginput,dis_output)
gan.compile(loss='binary_crossentropy',optimizer='adam')
gan.summary()

def gan_train(epoch,batch_size,saving_interval):
    (X_train,_),(_,_) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0],row_size,col_size,1).astype('float32')
    X_train = (X_train-127.5)/127.5
    true = np.ones((batch_size,1))
    fake = np.zeros((batch_size,1))

    for i in range(epoch):
        idx = np.random.randint(0,X_train.shape[0],batch_size)
        imgs = X_train[idx]
        d_loss_real = discriminator.train_on_batch(imgs,true)

        noise = np.random.normal(0,1,(batch_size,100))
        gen_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_imgs,fake)

        d_loss = 0.5*np.add(d_loss_real,d_loss_fake)
        g_loss = gan.train_on_batch(noise,true)

        print("epoch:%d"%i,"d_loss:%.4f"%d_loss,"g_loss:%.4f"%g_loss)

        if i%saving_interval == 0:
            noise = np.random.normal(0,1,(25,100))
            gen_imgs = generator.predict(noise)
            gen_imgs = 0.5*gen_imgs+0.5
            fig,axs = plt.subplots(5,5)
            count = 0
            for j in range(5):
                for k in range(5):
                    axs[j,k].imshow(gen_imgs[count,:,:,0],cmap='gray')
                    axs[j,k].axis('off')
                    count += 1
                    fig.savefig("gan_images/gan_mnist_%d.png"%i)

gan_train(4000+1,32,200)