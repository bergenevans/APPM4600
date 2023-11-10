import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(x):
    return 1 / (1 + x**2)

a = -5
b = 5
tol = 1e-6
print('tol:',tol)
actual = quad(f,a,b)
print('SCIPY quad value:',actual[0])

# composite Trapezoidal rule
t = 0
# predicted value of n for trapezoidal rule
nt = 1291
ht = (b-a)/nt
for j in range(1,nt):
    xj = a + j*ht
    t = t + 2*f(xj)
    trap = ht/2 * (f(a)+ t +f(b))
    diff = abs(trap - actual[0])
    # print(diff)
    if diff < tol:
        print('Trapezoidal has this many iterations:', j)
        break
    # print(xj,t)
print('Trapezoidal rule value:',trap)
# print(trap)

# composite Simpson's rule
# n must be even
s1 = 0
s2 = 0
# predicted value of n for simpsons rule
ns = 108
hs = (b-a)/ns

for j in range(1,ns):
    xj = a + j*hs
    if j % 2 == 0:
        s1 = s1 + 2*f(xj)
    else:
        s2 = s2 + 4*f(xj)
    simp = hs/3 * (f(a) + s1 + s2 + f(b))
    diff = abs(simp - actual[0])
    if diff < tol:
        print('Simpsons has this many iterations:',j)
        break 
print('Simpsons rule value:',simp)
# print(simp)
