import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0,2*np.pi,10000) # chose large number of steps to seem continuous
R = 1.2
deltar = 0.1
f = 15
p = 0

x = R*(1+deltar*np.sin(f*theta+p))*np.cos(theta)
y = R*(1+deltar*np.sin(f*theta+p))*np.sin(theta)

plt.plot(x,y)
plt.axis('equal')
plt.xlabel('x(theta)')
plt.ylabel('y(theta)')
plt.show()