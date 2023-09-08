import numpy as np
import matplotlib.pyplot as plt

x1 = np.pi
x2 = 10**6

x = np.arange(-16,1)
delta = 10.0**x

y1 = (np.cos(x1+delta)**2 - 1 + np.sin(x1)**2)/(np.cos(x1+delta)+np.cos(x1))
y2 = (np.cos(x2+delta)**2 - 1 + np.sin(x2)**2)/(np.cos(x2+delta)+np.cos(x2))

y1_new = -delta*np.sin(x1) + (delta**2/2)*np.cos(x1)
y2_new = -delta*np.sin(x2) + (delta**2/2)*np.cos(x2)

print('This is y1:',y1)
print('This is y1_new:',y1_new)
print('This is the difference between them:', y1-y1_new)

print('This is y2:',y2)
print('This is y2_new:',y2_new)
print('This is the difference between them:', y2-y2_new)