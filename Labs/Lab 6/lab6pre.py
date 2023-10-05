import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.cos(x)

def forward(s,h):
    return (f(s+h)-f(s))/h

def centered(s,h):
    return (f(s+h)-f(s-h))/(2*h)

h = 0.01 * 2. **(-np.arange(0,10))
s = np.pi/2

for i in h:
    fd = forward(s,i)
    cd = forward(s,i)

# print(fd,cd)
print(f"For h = {i:.10f}: Forward Difference = {fd:.10f}, Centered Difference = {cd:.10f}")
