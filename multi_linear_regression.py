import numpy as np
import pandas as pd

# time :    2   4   6   8
# private:  0   4   2   3
# score :   81  93  91  97
data  =   [[2,0,81],[4,4,93],[6,2,91],[8,3,97]]
time  =   [i[0] for i in data]
private = [i[1] for i in data]
score =   [i[2] for i in data]

# cast numpy data format
time_np = np.array(time)
private_np = np.array(private)
score_np = np.array(score)

# initialize "a1" & "a2" & "b" of "y=a1x1+a2x2+b"
a1 = 0
a2 = 0
b  = 0

# determine learning rate
lr = 0.02

# How many times is it iterated?
epochs = 2000

# Gradient Descent
for i in range(epochs+1) :
    score_pred = a1*time_np+a2*private_np+b    # predict
    error = score_np-score_pred # get error
    # differentiation for a1
    a1_diff = -(2/len(time_np))*sum(time_np*(error))
    # differentiation for a2
    a2_diff = -(2/len(private_np))*sum(private_np*(error))
    # differentiation for b
    b_diff = -(2/len(time_np))*sum(error)
    # update a1 & a2 & b as learning rate
    a1 = a1-lr*a1_diff
    a2 = a2-lr*a2_diff
    b = b-lr*b_diff
    # print a & b for every 100's calculation
    if i%100 == 0 :
        print("epoch=%.f,a1=%.04f,a2=%.04f,b=%.04f"%(i,a1,a2,b))