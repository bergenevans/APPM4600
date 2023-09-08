import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1.920,2.080,0.001)

p_long = x**9 - 18*x**8 + 144*x**7 - 672**6 + 2016*x**5 - 4032*x**4 + 5376*x**3 - 4608*x**2 + 2304*x - 512
p_short = (x-2)**9

plt.plot(x,p_long,label = 'p expanded')
plt.plot(x,p_short, label = 'p short form')
plt.xlabel('x')
plt.ylabel('p')
plt.legend()
plt.show()