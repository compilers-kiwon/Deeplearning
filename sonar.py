from    keras.models            import  Sequential
from    keras.layers.core       import  Dense
from    sklearn.preprocessing   import  LabelEncoder
from    sklearn.model_selection import  train_test_split

import  pandas      as  pd
import  numpy       as  np
import  tensorflow  as  tf

np.random.seed(0)
tf.random.set_seed(3)

df = pd.read_csv("dataset/sonar.csv",header=None)

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)

model = Sequential()
model.add(Dense(24,input_dim=60,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=200,batch_size=5)
print("\n Accuracy=%.4f"%(model.evaluate(X_test,Y_test)[1]))