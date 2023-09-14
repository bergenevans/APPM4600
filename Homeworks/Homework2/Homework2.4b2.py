import matplotlib.pyplot as plt
import numpy as np
import random

def plot_curve(i): # modularizes the plotting of each curve into a function
    theta = np.linspace(0,2*np.pi,10000)
    R = i
    deltar = 0.05
    f = 2 + i
    p = random.uniform(0,2)

    x = R*(1+deltar*np.sin(f*theta+p))*np.cos(theta)
    y = R*(1+deltar*np.sin(f*theta+p))*np.sin(theta)

    plt.plot(x,y)

for i in range(1, 11): # puts the 10 different curves into the plot through for loop 
    plot_curve(i)  

plt.axis('equal')
plt.xlabel('x(theta)')
plt.ylabel('y(theta)')
plt.show()