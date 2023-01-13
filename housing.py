from    keras.models            import  Sequential
from    keras.layers            import  Dense
from    sklearn.model_selection import  train_test_split

import  numpy   as  np
import  pandas  as  pd
import  tensorflow  as  tf

seed = 0
np.random.seed(seed)
tf.random.set_seed(3)

df = pd.read_csv("dataset/housing.csv",delim_whitespace=True,header=None)
print(df.info())
print(df.head())

dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=seed)

model = Sequential()
model.add(Dense(30,input_dim=13,activation='relu'))
model.add(Dense(6,activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(X_train,Y_train,epochs=200,batch_size=10)

Y_pred = model.predict(X_test).flatten()

for i in range(len(Y_test)) :
#for i in range(10) :
    label = Y_test[i]
    prediction = Y_pred[i]
    print("actual price: {:.3f}, expected price: {:.3f}".format(label,prediction))