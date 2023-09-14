import matplotlib.pyplot as plt
import numpy as np

# defines t and y as their respective functions
t = np.linspace(0, np.pi, 30)
y = np.cos(t)

k = 0 # k starts at 0 bc we're going from start to k<N so to get 30 iterations must be from 0-29
N = len(t) # end number for while loop
S = 0 # define as 0 to start

# iterates through the different values of k from 0 to 30 while taking the sum as we go
while(k<N):
    S = S + t[k]*y[k] 
    k = k + 1 # to increment each time

print('the sum is:', S)