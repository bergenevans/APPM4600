import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

# combined version of Newton and Steepest Descent method 
# takes Steepest Descent product and uses this as starting point for Newton

# need to combine # of iterations and time and just output one.

def driver():

    Nmax = 100
    x0= np.array([1,0,1])
    tol = 2e-5
    
    # statements from Steepest Descent
    # # [xsteep,gval,ier] = SteepestDescent(x0,tol,Nmax)
    # print("the initial guesses were", x0)
    # print("the steepest descent code found the solution ",xsteep)
    # print("g evaluated at this point is ", gval)
    # print("ier is ", ier)

    t = time.time()
    for j in range(50):
      [xsteep,gval,sdits,ier] = SteepestDescent(x0,tol,Nmax)
      [xstar,ier,its] =  Newton(xsteep,tol,Nmax)
    elapsed = time.time()-t
    print("the intial guesses used in Steepest Descent were:", x0)
    print("the steepest descent code found the solution", xsteep)
    print("the initial guesses used in Newton were:", xsteep)
    print("The error message for Steepest Descent -> Newton was:", ier)
    print("Steepest Descent -> Newton took this many seconds", elapsed/50)
    print("Steepest Descent -> Newton had this number of iterations: ", (sdits + its))
    print("Steepest Descent -> Newton resulted in: ", xstar)

    # print statements from Newton
    # print('Newton: initial guesses:',xsteep)
    # print('Newton: the error message reads:',ier) 
    # print('Newton: took this many seconds:',elapsed/50)
    # print('Netwon: number of iterations is:',its)

#functions:
def evalF(x):  

    F = np.zeros(3)
    
    F[0] = x[0] + np.cos(x[0]*x[1]*x[2]) - 1
    F[1] = (1-x[0])**(1/4) + x[1] + 0.05*(x[2])**2 - 0.15*x[2] - 1
    F[2] = -(x[0])**2 - 0.1*(x[1])**2 + 0.01*x[1] + x[2] - 1
    return F
    
def evalJ(x): 

    J = np.array([[1 - np.sin(x[0]*x[1]*x[2]), - np.sin(x[0]*x[1]*x[2]), - np.sin(x[0]*x[1]*x[2])], 
        [-(1/4)*(1-x[0]), 1, 0.1*x[2] - 0.15], 
        [-2*x[0], -0.2*x[1] + 0.01, 1]])

    return J

def evalg(x):

    F = evalF(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
    F = evalF(x)
    J = evalJ(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg

### steepest descent code

def SteepestDescent(x,tol,Nmax):
    
    for sdits in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)

        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,sdits,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)

        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,sdits,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,sdits,ier]

def Newton(xsteep,tol = 1e-6,Nmax = 100):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
       J = evalJ(xsteep)
       Jinv = inv(J)
       F = evalF(xsteep)
       
       x1 = xsteep - Jinv.dot(F)
       
       if (norm(x1-xsteep) < tol):
           xstar = x1
           ier =0
           return[xstar, ier, its]
           
       xsteep = x1
    
    xstar = x1
    ier = 1
    return[xstar,ier,its]


if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()        
