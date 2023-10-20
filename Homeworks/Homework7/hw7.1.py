import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import solve

# modified interp.py code provided on canvas

def driver():


    f = lambda x: 1/(1+(10*x)**2)

    N = 16
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
   #  xint = np.linspace(a,b,N+1)
    # xint = np.linspace(a,b,N+1)
    xint = np.zeros(N+1)
    h = 2 / (N - 1)
    for i in range(N+1):
       xint[i] = -1 + (i-1)*h
    print(xint)
    # xint = np.zeros((N+1, 1))
    # for j in range(N+1):
    #    xint[j] = np.cos((2*j-1)*np.pi/(2*N))
    
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_m = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_m[kk] = monomial(f, xeval[kk],xint,yint,N)
          
    # print(yeval_m[kk])
    ''' create vector with exact values'''
    fex = f(xeval)
       
    # approximation
    plt.figure()    
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_m,'o', label = 'Monomial')
    plt.title("Approximation")
    plt.legend()
    plt.show()

    # # absolute error
    # plt.figure() 
    # err_m = abs(yeval_m-fex)
    # plt.semilogy(xeval,err_m, 'c.--', label='Monomial')
    # plt.legend()
    # plt.title("Absolute Error")
    # plt.show()


def monomial(f, xeval,xint,yint,N):
   V = np.zeros((N+1, N+1))
#    print(xint)
#    print(yint)
   for i in range(N+1):
      for j in range(N+1):
         V[i,j] = xint[i]**j
   # print(V)
   Vinv = inv(V)
   c = np.matmul(Vinv,yint)
   yeval = 0
   for j in range(N+1):
      yeval = yeval + c[j] * (xeval**(j))
   return yeval

driver() 