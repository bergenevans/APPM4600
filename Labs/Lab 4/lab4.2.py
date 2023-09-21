import numpy as np


# modified code from fixedpt_example.py provided on canvas
def driver():

# test functions 
     f1 = lambda x: (10/(x+4))**0.5
# fixed point is alpha1 = 1.4987....

     # f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09... 

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     print(fixedpt(f1,x0,tol,Nmax))
#      [xstar,ier] = fixedpt(f1,x0,tol,Nmax)
#      print('the approximate fixed point is:',xstar)
#      print('f1(xstar):',f1(xstar))
#      print('Error message reads:',ier)
    
# test f2 '''
#      x0 = 0.0
#      [xstar,ier] = fixedpt(f2,x0,tol,Nmax)
#      print('the approximate fixed point is:',xstar)
#      print('f2(xstar):',f2(xstar))
#      print('Error message reads:',ier)


# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    X = np.zeros((Nmax,1))
    X[0] = x0
    count = 0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       X[count] = x1
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return X
       x0 = x1

    xstar = x1
    ier = 1
    return X
    

driver()


print('2a. 13 iterations')
