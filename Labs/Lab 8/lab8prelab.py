import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 

xeval = np.linspace(0,10,1000)
xint = np.linspace(0,10,11)

def findxeval(xeval,xint):
    subintervals = []
    for i in range(1, len(xint)):
        ind = np.where((xint[i-1]<=xeval) & (xeval<=xint[i]))
        # print(ind)
        subintervals.append(ind)
    return subintervals

result = findxeval(xeval, xint)
print(result)

x0 = 1
x1 = 2
f = lambda x: x**2 + 3
def line(f,x0,x1):
    m = (f(x1)-f(x0))/(x1-x0)
    b = f(x0) - m * x0
    return m,b
m,b = line(f,x0,x1)

print('line is: y=',m,'x + ',b)