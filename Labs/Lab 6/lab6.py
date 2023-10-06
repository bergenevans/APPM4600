import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# Newton with an approx Jacobian is going to take the Jacobian with approximations (prelab) instead
# deltaF1/deltax1 approx = (f1(x+h,y)-f1(x,y))/h
# deltaf1/deltax2 approx = (f1(x,y+h)-f1(x,y))/h
# can calculate the same thing for different values of h and see how it changes instead of worrying about 3.3.2

# 3.2 1. 
#   - a good condition to check whether we should recompute the Jacobian is to see if the difference between
#       the two iterations is small enough, and continue until we get there
#   - another condition could be to check the convergence rate
# 3.2.4 couldn't compare, we didn't get this far
# 3.2.5 This method does not seem to work very well as it it requiring many iterations and is still failing

# 3.3.2 I cannot get this to converge to values, my approximations worked for the code given in class but not here, it is resulting in a singular matrix
# 3.3.3 the method performed from the original newton from class worked, it displayed the same values, the biggest difference was that it took more time 

# 3.4.2 It gives the same result as 3.3.2, nan and nan
# 3.4.3 The example from class worked well, couldn't get it to work on this one for some reason


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

    F = np.zeros(2)
    
    F[0] = (4*x[0])**2 + (x[1])**2 - 4
    F[1] = x[0] + x[1] - np.sin(x[0]-x[1])
    return F
    
def evalJ(x, h=1e-3, recomputenum=0): 
    J = np.zeros((2,2))
    h = (1.0 / 2.0)**(recomputenum) * h * np.abs(x)

    # deltaF1/deltax1 approx = (f1(x+h,y)-f1(x,y))/h
    # deltaF1/deltax2 approx = (f1(x,y+h)-f1(x,y))/h
    diffx1 = np.array([x[0] + h[0], x[1]])
    diffx2 = np.array([x[0], x[1] + h[1]])
    J[:, 0] = (evalF(diffx1) - evalF(x)) / h[0]
    J[:, 1] = (evalF(diffx2) - evalF(x)) / h[1]

    # J = np.array([[8*x[0],2*x[1]],[1-np.cos(x[0]-x[1]),1+np.cos(x[0]-x[1])]])
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

    recomputenum = 0
    prevnorm = norm(evalF(x0))
    prevsolnorm = norm(x0)
    count = 0

    for its in range(Nmax):

        F = evalF(x0)
        x1 = x0 - Jinv.dot(F)

       # this checks if the current norm is close enough to the previous norm
       # if not we keep computing the Jacobian
        currentnorm = norm(F)
        if abs(currentnorm - prevnorm) < tol:
            recomputenum += 1
            J = evalJ(x0, recomputenum=recomputenum)
            Jinv = inv(J)

        if (norm(x1-x0) < tol):
            xstar = x1
            ier =0
            return[xstar, ier,its]
        
        x0 = x1
        prevnorm = currentnorm
        # prevsolnorm = currentsolnorm
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]   
     
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()   