import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2,10,100)
f = x - 4*np.sin(2*x)-3

plt.plot(x,f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid()
plt.show()