# required keras functions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# required libraries
import numpy as np
import tensorflow as tf

# to get a same result for every execution
np.random.seed(3)
tf.random.set_seed(3)

# load dataset
Data_set = np.loadtxt("dataset/ThoraricSurgery.csv",delimiter=",")

# read patient records and surgery result
X = Data_set[:,0:17]
Y = Data_set[:,17]

# configure deeplearning model
model = Sequential()
model.add(Dense(30,input_dim=17,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# run deeplearning model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,Y,epochs=100,batch_size=10)