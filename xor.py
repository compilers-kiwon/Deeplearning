import numpy as np

# W1 = |-2 2|  B1 = | 3|
#      |-2 2|       |-1|
# W2 = | 1|    B2 = |-1|
#      | 1|
w11 = np.array([-2,-2])
w12 = np.array([2,2])
w2 = np.array([1,1])
b1 = 3
b2 = -1
b3 = -1

# Perceptron
def perceptron(x,w,b) :
    y = np.sum(w*x)+b
    if y <= 0 : ret = 0
    else : ret = 1
    return ret

# NAND
def NAND(x1,x2) :
    return perceptron(np.array([x1,x2]),w11,b1)

# OR
def OR(x1,x2) :
    return perceptron(np.array([x1,x2]),w12,b2)

# AND
def AND(x1,x2) :
    return perceptron(np.array([x1,x2]),w2,b3)

# XOR
def XOR(x1,x2) :
    return AND(NAND(x1,x2),OR(x1,x2))

for x in [(0,0),(1,0),(0,1),(1,1)] :
    y = XOR(x[0],x[1])
    print("input:"+str(x)+" output:"+str(y))