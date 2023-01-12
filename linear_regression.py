import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# time :    2   4   6   8
# score :   81  93  91  97
data  = [[2,81],[4,93],[6,91],[8,97]]
time  = [i[0] for i in data]
score = [i[1] for i in data]

# Graph
plt.figure(figsize=(8,5))
plt.scatter(time,score)
plt.show()

# cast numpy data format
time_np = np.array(time)
score_np = np.array(score)

# initialize "a" & "b" of "y=ax+b"
a = 0
b = 0

# determine learning rate
lr = 0.03

# How many times is it iterated?
epochs = 2000

# Gradient Descent
for i in range(epochs+1) :
    score_pred = a*time_np+b    # predict
    error = score_np-score_pred # get error
    # differentiation for a
    a_diff = -(2/len(time_np))*sum(time_np*(error))
    # differentiation for b
    b_diff = -(2/len(time_np))*sum(error)
    # update a & b as learning rate
    a = a-lr*a_diff
    b = b-lr*b_diff
    # print a & b for every 100's calculation
    if i%100 == 0 :
        print("epoch=%.f,a=%.04f,b=%.04f"%(i,a,b))

# Update graph with final a & b
score_pred = a*time_np+b
plt.scatter(time,score)
plt.plot([min(time_np),max(time_np)],[min(score_pred),max(score_pred)])
plt.show()
