import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from numpy.linalg import inv

def driver():


    f = lambda x: 1/(1+(10*x)**2)

    N = 18
    ''' interval'''
    a = -1
    b = 1
   
   
    ''' create equispaced interpolation nodes'''
   #  xint = np.linspace(a,b,N+1)
    xint = np.zeros((N+1, 1))
    for j in range(N+1):
       xint[j] = np.cos((2*j-1)*np.pi/(2*N))
    
    ''' create interpolation data'''
    yint = f(xint)
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yeval_m = np.zeros(Neval+1)
    yeval_l= np.zeros(Neval+1)
    yeval_dd = np.zeros(Neval+1)
  
    '''Initialize and populate the first columns of the 
     divided difference matrix. We will pass the x vector'''
    y = np.zeros( (N+1, N+1) )
     
    for j in range(N+1):
       y[j][0]  = yint[j]

    y = dividedDiffTable(xint, y, N+1)
    ''' evaluate lagrange poly '''
    for kk in range(Neval+1):
       yeval_m[kk] = monomial(xeval[kk],xint,yint,N)
       yeval_l[kk] = eval_lagrange(xeval[kk],xint,yint,N)
       yeval_dd[kk] = evalDDpoly(xeval[kk],xint,y,N)
          

    


    ''' create vector with exact values'''
    fex = f(xeval)
       
    # approximation
    plt.figure()    
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval_l,'bs--') 
    plt.plot(xeval,yeval_dd,'c.--')
    plt.plot(xeval,yeval_m,'go--')
   #  plt.title("Approximation")
    plt.legend()

    # absolute error
    plt.figure() 
    err_l = abs(yeval_l-fex)
    err_dd = abs(yeval_dd-fex)
    err_m = abs(yeval_m-fex)
    plt.semilogy(xeval,err_l,'ro--',label='lagrange')
    plt.semilogy(xeval,err_dd,'bs--',label='Newton DD')
    plt.semilogy(xeval,err_m, 'c.--', label='Monomial')
    plt.legend()
   #  plt.title("Absolute Error")
    plt.show()


def monomial(xeval,xint,yint,N):
   V = np.zeros((N+1,N+1))
   for i in range(N+1):
      for j in range(N+1):
         V[i,j] = xint[i]**j
   Vinv = inv(V)
   a = np.matmul(Vinv,yint)
   yeval = 0
   for j in range(N+1):
      yeval = yeval + a[j] * (xeval**j)
   return yeval


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
  

''' create divided difference matrix'''
def dividedDiffTable(x, y, n):
 
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) /
                                     (x[j] - x[i + j]));
    return y;
    
def evalDDpoly(xval, xint,y,N):
    ''' evaluate the polynomial terms'''
    ptmp = np.zeros(N+1)
    
    ptmp[0] = 1.
    for j in range(N):
      ptmp[j+1] = ptmp[j]*(xval-xint[j])
     
    '''evaluate the divided difference polynomial'''
    yeval = 0.
    for j in range(N+1):
       yeval = yeval + y[0][j]*ptmp[j]  

    return yeval

driver()        

# answers to questions:
# 3.1 2. The methods seem to be about the same, their absolute error plots are nearly identical for each case
# 3.1 3. when p(x) is about 100, the plot gets more and more crazy
#     the errors for the methods go up by a lot and we can see the errors increase towards each end
#     there still is not a lot of difference for the methods
# 3.2 2. Using this interpolation nodes results in a singular matrix
# 3.2 3. Cannot plot, results in a singular matrix