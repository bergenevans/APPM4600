import numpy as np
import matplotlib.pyplot as plt

# this linspace function orders data from 0 to 2pi in 100 increments
X = np.linspace(0, 2 * np.pi, 100)
Ya = np.sin(X)
Yb = np.cos(X)

# this down below creates a plot of X and Y
plt.plot(X, Ya)
plt.plot(X,Yb)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

x = np.linspace(1,10,10)
y = np.arange(1,11)

# : = everything
# :N = everything until the Nth entry
N = 3
print('the first three entries of x are',x[:N])
w = 10**(-np.linspace(1,10,10))
print(w)

# computes the  length of w
l = len(w)

x = np.linspace(1,l,10)

s = 3 * w

# This creates a semilog scale plot along the y axis of x vs w and x vs s
plt.semilogy(x,w)
plt.semilogy(x,s)
plt.xlabel('x')
plt.ylabel('w/s')
plt.show()