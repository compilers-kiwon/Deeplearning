import numpy as np

# time :    2   4   6   8
# score :   81  93  91  97
X = [2,4,6,8]
Y = [81,93,91,97]

# mean
Xm = np.mean(X)
Ym = np.mean(Y)
print("mean of X:",Xm)
print("mean of Y:",Ym)

# Y = aX+b
#
#     sum((X_i-Xm)*(Y_i-Ym))
# a = -----------------------
#         sum((X_i-Xm)^2)
def top(x,mx,y,my) :
    ret = 0
    for i in range(len(x)) :
        ret += (x[i]-mx)*(y[i]-my)
    return ret
divisor = sum([(x-Xm)**2 for x in X])
dividend = top(X,Xm,Y,Ym)
print("divisor:",divisor)
print("dividend:",dividend)

a = dividend/divisor
b = Ym-(Xm*a)

print("y=ax+b")
print("a=",a)
print("b=",b)