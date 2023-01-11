import random as r
import numpy as np

a = r.randint(1,10)
b = r.randint(1,100)

print("a =",a)
print("b =",b)

data  = [[2,81],[4,93],[6,91],[8,97]]
time  = [i[0] for i in data]
score = [i[1] for i in data]

predict = [a*i+b for i in time]
error = []

for i in range(len(predict)):
    print("time to study=%.f,actual score=%.f,predicted_score=%.f"%(time[i],score[i],predict[i]))
    error.append((score[i]-predict[i])**2)

mse = np.mean(error)
print("Mean Square Error:",mse)