import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# modified Newton nonlinear code provided on canvas

def driver():

    x0 = np.array([1, -1])
    
    Nmax = 100
    tol = 1e-15
    
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: intial guesses:',x0)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)
    
def evalF(x): 

    F = np.zeros(2)
    
    F[0] = (x[0])**2 + (x[1])**2 - 4
    F[1] = np.exp(x[0]) + x[1] - 1

    return F
    
def evalJ(x): 

    J = np.array([[2*x[0],2*x[1]],[np.exp(x[0]),1]])

    return J

           
def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv.dot(F)
       
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
