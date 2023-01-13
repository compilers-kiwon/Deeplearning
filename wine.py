from    keras.models    import  Sequential
from    keras.layers    import  Dense
from    keras.callbacks import  ModelCheckpoint,EarlyStopping

import  os
import  pandas  as pd
import  numpy   as np
import  tensorflow          as  tf
import  matplotlib.pyplot   as  plt

seed_for_random = 3
np.random.seed(seed_for_random)
tf.random.set_seed(seed_for_random)

df_pre = pd.read_csv("dataset/wine.csv",header=None)
df = df_pre.sample(frac=0.15)
dataset = df.values
X = dataset[:,0:12]
Y = dataset[:,12]

model = Sequential()
model.add(Dense(30,input_dim=12,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR) :
    os.mkdir(MODEL_DIR)

modelpath = './model/{epoch:02d}-{val_loss:.4f}.hdf5'

checkpointer = ModelCheckpoint(filepath=modelpath,monitor='val_loss',
                               verbose=1,save_best_only=True)

early_stopping_callback = EarlyStopping(monitor='val_loss',patience=100)

history = model.fit(X,Y,validation_split=0.2,epochs=3500,batch_size=500,
                    verbose=0,callbacks=[checkpointer,early_stopping_callback])

y_vloss = history.history['val_loss']
y_acc = history.history['accuracy']

x_len = np.arange(len(y_acc))
plt.plot(x_len,y_vloss,"o",c="red",markersize=3)
plt.plot(x_len,y_acc,"o",c="blue",markersize=3)

plt.show()