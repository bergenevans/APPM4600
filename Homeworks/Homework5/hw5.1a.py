import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# modified Lazy Newton provided from class

def driver():

    x0 = np.array([1,1])
    
    Nmax = 100
    tol = 1e-10
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('the error message reads:',ier)
    print('took this many seconds:',elapsed/20)
    print('number of iterations is:',its)
     
def evalF(x): 

    F = np.zeros(2)
    
    F[0] = 3*(x[0])**2 - (x[1])**2
    F[1] = 3*x[0]*(x[1])**2 - (x[0])**3 - 1

    return F
           
def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = np.array([[1/6,1/18],[0,1/6]])
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - J.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
     
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()       

# check
# def evalF(x): 

#     F = np.zeros(2)
    
#     F[0] = 3*(x[0])**2 - (x[1])**2
#     F[1] = 3*x[0]*(x[1])**2 - (x[0])**3 - 1
#     return F

# x = np.array([0.5,0.8660254])

# print(evalF(x))