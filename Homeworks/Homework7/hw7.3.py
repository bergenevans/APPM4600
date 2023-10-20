# will be same as 7.2 but with chebychev points

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# modified interp.py code provided on canvas

def driver():


    f = lambda x: 1/(1+(10*x)**2)

    N = 20
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
    # xint = np.linspace(a,b,N+1)
    xint = np.zeros(N+1)
    for j in range(N+1):
       xint[j] = np.cos((2*j+1)*np.pi/(2*(N+1)))
    
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    # y = np.zeros( (N+1, N+1) )
     
    # for j in range(N+1):
    #    y[j][0]  = yint[j]

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
          
    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_l,'bs--', label = 'Lagrange')
    plt.title("Approximation")
    plt.legend()
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    # p = np.one(N+1)
    # for i in range(N+1):
    #    for j in range(N+1):
    #       if (j != i):
    #          c = np.poly([xint[j]]) / (xint[i] - xint[j])
    #          p = np.convolve(p,c)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              if abs(xint[count]-xint[jj]) > 1e-15:
                lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)

  
driver()   