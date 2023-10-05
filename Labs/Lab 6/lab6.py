import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# 3.2 1. 
#   - a good condition to check whether we should recompute the Jacobian is to see if the difference between
#       the two iterations is small enough, and continue until we get there
#   - another condition ... 
# 3.2.4 ... 
# 3.2.5 ... 
# modified LazyNewton from canvas

def driver():

    x0 = np.array([1, 0])
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(50):
      [xstar,ier,its] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',its)
     
    t = time.time()
    for j in range(20):
      [xstar,ier,its] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',its)
     
    # t = time.time()
    # for j in range(20):
    #   [xstar,ier,its] = Broyden(x0, tol,Nmax)     
    # elapsed = time.time()-t
    # print(xstar)
    # print('Broyden: the error message reads:',ier)
    # print('Broyden: took this many seconds:',elapsed/20)
    # print('Broyden: number of iterations is:',its)
     
def evalF(x): 

    F = np.zeros(3)
    
    # F[0] = 3*x[0]-math.cos(x[1]*x[2])-1/2
    F[1] = (4*x[1])**2 + (x[2])**2 - 4
    F[2] = x[1] + x[2] - np.sin(x[1]-x[2])
    return F
    
def evalJ(x): 

    J = np.array([[8*x[1],2*x[2]],[1-np.cos(x[1]-x[2]),1+np.cos(x[1]-x[2])]])
    return J


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv.dot(F)
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]
           
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
       # added this check down below to see if the iterations were close to eachother
       if abs(x1-x0) < tol:
           xstar = x1
           ier = 0
           return[xstar,ier,its]
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
     
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()   