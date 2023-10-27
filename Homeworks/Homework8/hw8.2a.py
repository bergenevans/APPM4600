import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# modified interp.py provided on canvas


def driver():


    f = lambda x: 1/(1+x**2)

    N = 20
    ''' interval'''
    a = -5
    b = 5
   
   
    # ''' create equispaced interpolation nodes'''
    # xint = np.linspace(a,b,N+1)
    # Chevyshev nodes
    xint = np.zeros(N+1)
    ch = np.cos((2*np.arange(1,N+2)-1)*np.pi/(2*(N+1)))
    xint = 0.5 * (a + b) + 0.5 * (b - a) * ch
    print(xint)
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_l= np.zeros(Neval+1)
  

    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
          

    ''' create vector with exact values'''
    fex = f(xeval)
       

    plt.figure()    
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_l,'bs--', label='Lagrange') 
    plt.legend()
    plt.title("Lagrange Approximation for n=20")

    plt.figure() 
    err_l = abs(yeval_l-fex)
    plt.semilogy(xeval,err_l,'ro--',label='Lagrange')
    plt.legend()
    plt.title("Absolute Error")
    plt.show()

def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  
driver()        
