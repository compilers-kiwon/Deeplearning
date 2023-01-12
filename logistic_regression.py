import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# study_hours : 2   4   6   8   10  12  14
# pass :        0   0   0   1   1   1   1   (0:false,1:true)
data = [[2,0],[4,0],[6,0],[8,1],[10,1],[12,1],[14,1]]
x_data = [i[0] for i in data]
y_data = [i[1] for i in data]

# initialize "a" & "b"
a = b = 0

# learning rate
lr = 0.05

# sigmoid()
def sigmoid(x) :
    return  1/(1+np.e**(-x))

# How many times is it iterated?
epochs = 2000

# Gradient Descent
for i in range(epochs+1) :
    for x,y in data :
        # differentiation for a
        a_diff = x*(sigmoid(a*x+b)-y)
        # differentiation for b
        b_diff = sigmoid(a*x+b)-y
        # update a1 & a2 & b as learning rate
        a = a-lr*a_diff
        b = b-lr*b_diff
        # print a & b for every 1000's calculation
        if i%1000 == 0 :
            print("epoch=%.f,a=%.04f,b=%.04f"%(i,a,b))
        
    
# Graph
plt.scatter(x_data,y_data)
plt.xlim(0,15)
plt.ylim(-.1,1.1)
x_range = (np.arange(0,15,0.1))
plt.plot(np.arange(0,15,0.1),np.array([sigmoid(a*x+b) for x in x_range]))
plt.show()