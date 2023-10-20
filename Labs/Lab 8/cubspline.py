import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import solve


def M(x, f):
    n = len(x)

    # Differences between x values
    h = np.diff(x)

    # Tridiagonal system
    A = np.zeros((n,n))
    B = np.zeros(n)

    # Initialize these because they don't get initialized in
    # the for loop
    A[0][0] = 1
    A[-1, -1] = 1

    for i in range(1, n-1):
        A[i, i-1] = 1/12 * h[i-1]
        A[i, i] = 1/3
        A[i, i+1] = 1/12 * h[i]

        B[i] = (f[i+1] - 2*f[i] + f[i-1]) / (2 * h[i-1]**2)
    M = np.dot(inv(A), B)
    return M

def cubic(M0,M1,x0,x1,f0,f1):
    # h0 = x1 - x0
    # C = f0/h0 - (h0/6)*M0
    # D = f1/h0 - (h0*M1)/6
    # # S = lambda x: ((M0*(x1-x)**3)/(6*h0)) + ((M1*(x-x0)**3)/(6*h0)) + C*(x-x0) + D*(x1-x)
    # S = lambda x: (M0*(x1-x)**3)/(6*h0) + (M1*(x-x0)**3)/(6*h0) + C*(x-x0) + D*(x1-x)
    
    h = x1 - x0
    S = lambda x: (M0 * (x1 - x)**3 + M1 * (x - x0)**3) / (6 * h) + (f0/h - (M0 * h)/6)*(x1 - x) + (f1/h - (M1 * h)/6)*(x - x0)
    return S

def driver():
    
    f = lambda x: 1 / (1 + (10*x)**2)
    a = -1
    b = 1
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 10
    
    '''evaluate the linear spline'''
    yeval = eval_cube_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 
      
    
    plt.figure()
    plt.plot(xeval,fex,'ro-', label = 'actual function')
    plt.plot(xeval,yeval,'bs-', label = 'cubic spline spline')
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.show()


    fvalues = f(xeval)
    Meval = M(xeval,fvalues)
    print(Meval)


def eval_cube_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 

    ''' Evaluate M matrix for interval points '''
    Mmat = M(xint, f(xint))

    for i in range(Neval):
        ''' Find the interval xeval[i] belongs to '''
        """
        This needs to give a yeva[i] for every i
        While mapping an i to the corresponding jint
        """
        for j in range(Nint):
            if xint[j] <= xeval[i] < xint[j + 1] or (j == Nint-1 and xeval[i] == b):
                jint = j
                break
        S = cubic(Mmat[jint], Mmat[jint+1], xint[jint], xint[jint+1], f(xint[jint]), f(xint[jint+1]))
        yeval[i] = S(xeval[i])
    return yeval
           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()

# 3.4 This method seems to work worse, which is not expected, so something is probably wrong with my code
#   Something with my formulas seem off and I cannot get it any better