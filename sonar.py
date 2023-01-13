# if you can see the below error :
#
#       ...
#       model_config = json.loads(model_config.decode('utf-8'))
#   AttributeError: 'str' object has no attribute 'decode'
#
# please apply the below configration to your environment:
#
#   > pip install h5py==2.10.0

from    keras.models            import  Sequential,load_model
from    keras.layers.core       import  Dense
from    sklearn.preprocessing   import  LabelEncoder
from    sklearn.model_selection import  train_test_split,StratifiedKFold

import  pandas      as  pd
import  numpy       as  np
import  tensorflow  as  tf

np.random.seed(0)
tf.random.set_seed(0)

df = pd.read_csv("dataset/sonar.csv",header=None)

dataset = df.values
X = dataset[:,0:60]
Y_obj = dataset[:,60]

e = LabelEncoder()
e.fit(Y_obj)
Y = e.transform(Y_obj)

n_fold = 10
skf = StratifiedKFold(n_splits=n_fold,shuffle=True,random_state=0)
accuracy = []

for train,test in skf.split(X,Y):
    model = Sequential()
    model.add(Dense(24,input_dim=60,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X[train],Y[train],epochs=100,batch_size=5)
    k_accuracy = "%.4f"%(model.evaluate(X[test],Y[test])[1])
    accuracy.append(k_accuracy)

print("\n %.f fold accuracy : "%n_fold,accuracy)