import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

alpha = 0.138*10**-6
t = 60*24*3600

x = np.linspace(0,5,100)

f = 35*erf(x/(2*np.sqrt(alpha*t)))-15

plt.plot(x,f)
plt.xlabel('x, depth')
plt.ylabel('f')
plt.show()