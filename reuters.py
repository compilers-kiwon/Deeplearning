from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense,LSTM,Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

(X_train,Y_train),(X_test,Y_test) = reuters.load_data(num_words=1000,test_split=0.2)
num_of_categories = np.max(Y_train)+1

x_train = sequence.pad_sequences(X_train,maxlen=100)
x_test = sequence.pad_sequences(X_test,maxlen=100)
y_train = np_utils.to_categorical(Y_train)
y_test = np_utils.to_categorical(Y_test)

model = Sequential()
model.add(Embedding(1000,100))
model.add(LSTM(100,activation='tanh'))
model.add(Dense(num_of_categories,activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=100,epochs=20,
                        verbose=1,validation_data=(x_test,y_test))

print("\n Test Accuracy:%.4f"%(model.evaluate(x_test,y_test)[1]))

y_vloss = history.history['val_loss']   # test
y_loss = history.history['loss']        # train

x_len = np.arange(len(y_loss))
plt.plot(x_len,y_vloss,marker='.',c="red",label='Testset_loss')
plt.plot(x_len,y_loss,marker='.',c="blue",label='Trainset_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()