import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,5,100)

def Mac(x):
    return x - x**3/6 + x**5/120

def Pade(x):
    return (x-7*x**3/60)/(1+x**2/20)

error = Mac(x) - Pade(x)

plt.figure()
plt.plot(x,Mac(x), 'ro--', label = 'MacLauren Approximation')
plt.plot(x,Pade(x), 'bs--', label = "Pade Approximation")
plt.xlabel('x')
plt.ylabel('Approximation')
plt.title('6th Order Approximation')
plt.legend()
plt.show()

plt.figure()
plt.plot(x, error, 'o', label='Error')
plt.xlabel('x')
plt.ylabel('Error: MacLauren - Pade Approximation')
plt.title('Error of 6th order Pade Approximation against 6th order MacLauren')
plt.legend()
plt.show()