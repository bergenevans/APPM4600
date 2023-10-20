import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import solve

# need to do section 3.3
# remember to submit plots too
# do where prof got but then divide by 2h

# will end up with number matrix * Mi(vector) =  (yi+1 - 2y + yi-1)/2h^2 ??


def line(x0,f0,x1,f1):
    m = (f1-f0)/(x1-x0)
    b = f0 - m * x0
    return m,b
# m,b = line(x0,f0,x1,f1)

# N-1 x N-1 matrix, 1 x N-1 vector
# number vector should be
    # [[1/3, 1/12 ...], [1/12, 1/3, ...]]


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
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 
      
    
    plt.figure()
    plt.plot(xeval,fex,'ro-', label = 'actual function')
    plt.plot(xeval,yeval,'bs-', label = 'linear spline')
    plt.legend()
    plt.show()
     
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.show()


def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
        ind = np.where((xint[jint]<=xeval) & (xeval<=xint[jint+1]))[0]
        n = len(ind)
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''

        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)


        m, b = line(a1,fa1,b1,fb1)
        yeval[ind] = m * xeval[ind] + b
        
        # for kk in range(n):
        #    '''use your line evaluator to evaluate the lines at each of the points 
        #    in the interval'''
        #    '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with 
        #    the points (a1,fa1) and (b1,fb1)'''
        #    lines = line(a1,fa1,b1,fb1)
        #    yeval(ind(kk)) = line(a1,fa1,b1,fb1)
    print(yeval)
    return yeval
           
           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               


# 3.2 this seems to perform _____ than global interpolation with uniform nodes from looking at the error graph
