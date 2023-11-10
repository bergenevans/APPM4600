import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1 / (1 + x**2)

a = -5
b = 5
n = 10
h = (b-a)/n

# composite Trapezoidal rule
t = 0
for j in range(1,n):
    xj = a + j*h
    t = t + 2*f(xj)
    # print(xj,t)

trap = h/2 * (f(a)+ t +f(b))
# print(trap)

# composite Simpson's rule
# n must be even
s1 = 0
s2 = 0
for j in range(2,n,2):
    xj = a + j*h
    s1 = s1 + 2*f(xj)

for j in range(1,n-1,2):
    xj = a + j*h
    s2 = s2 + 4*f(xj)

simp = h/3 * (f(a) + s1 + s2 + f(b))
print(simp)