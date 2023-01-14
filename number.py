from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')/255
X_test = X_test.reshape(X_test.shape[0],28,28,1).astype('float32')/255

Y_train = np_utils.to_categorical(Y_train,10)
Y_test = np_utils.to_categorical(Y_test,10)

# 28x28(=784) data
# =>
# Convolution Y1(32 filters,3x3) + relu
# =>
# Convolution Y2(64 filters,3x3) + relu
# =>
# MaxPooling Y3(pool size:2) + relu
# =>
# Dropout 25%
# =>
# Flatten
# =>
# Y4(128 nodes) + relu
# =>
# Dropout 50%
# =>
# Y_out(size:10) + softmax
from keras.layers import Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),
          input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))          
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10)

history = model.fit(X_train,Y_train,
                    validation_data=(X_test,Y_test),
                    epochs=30,batch_size=200,verbose=1,
                    callbacks=[early_stopping_callback])

print("\n Test Accuracy:%.4f"%(model.evaluate(X_test,Y_test)[1]))

y_test_loss = history.history['val_loss']
y_train_loss = history.history['loss']

x_len = np.arange(len(y_train_loss))
plt.plot(x_len,y_test_loss,marker='.',
         c='red',label="Testset_loss")
plt.plot(x_len,y_train_loss,marker='.',
         c='blue',label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()