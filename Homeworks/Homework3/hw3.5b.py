import matplotlib.pyplot as plt
import numpy as np

def driver():

# test functions 
     f1 = lambda x: -np.sin(2*x)+(5*x/4)-3/4

     Nmax = 100
     tol = 0.5e-10

# test f1 '''
     x0 = 2
     [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
     formatted_num = "{:,.10f}".format(xstar)
     print('my initial guess, x0, was', x0)
     print('the approximate fixed point is:',formatted_num)
   #   print('f1(xstar):',f1(xstar))
   #   print('Error message reads:',ier)


# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier]
       x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier]
    

driver()