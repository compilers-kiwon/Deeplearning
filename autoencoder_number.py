from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.layers import BatchNormalization,Activation,LeakyReLU,UpSampling2D
from tensorflow.keras.models import Sequential,Model

import numpy as np
import matplotlib.pyplot as plt

(X_train,_),(X_test,_) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')/255

autoencoder = Sequential()

autoencoder.add(Conv2D(16,kernel_size=3,padding='same',input_shape=(28,28,1),activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=2,padding='same'))
autoencoder.add(Conv2D(8,kernel_size=3,padding='same',activation='relu'))
autoencoder.add(MaxPooling2D(pool_size=2,padding='same'))
autoencoder.add(Conv2D(8,kernel_size=3,strides=2,padding='same',activation='relu'))

autoencoder.add(Conv2D(8,kernel_size=3,padding='same',activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(8,kernel_size=3,padding='same',activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(16,kernel_size=3,activation='relu'))
autoencoder.add(UpSampling2D())
autoencoder.add(Conv2D(1,kernel_size=3,padding='same',activation='sigmoid'))

autoencoder.summary()

autoencoder.compile(optimizer='adam',loss='binary_crossentropy')
autoencoder.fit(X_train,X_train,epochs=50,batch_size=128,validation_data=(X_test,X_test))

random_test = np.random.randint(X_test.shape[0],size=5)
autoencoder_images = autoencoder.predict(X_test)
plt.figure(figsize=(7,2))

for i,image_idx in enumerate(random_test):
    ax = plt.subplot(2,7,i+1)
    plt.imshow(X_test[image_idx].reshape(28,28))
    ax.axis('off')
    ax = plt.subplot(2,7,7+i+1)
    plt.imshow(autoencoder_images[image_idx].reshape(28,28))
    ax.axis('off')

plt.show()