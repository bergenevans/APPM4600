import matplotlib.pyplot as plt
import numpy as np


def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]


def driver():

# use routines    
    f = lambda x: np.exp(x**2+7*x-30)-1
    fp = lambda x: (2*x+7)*np.exp(x**2+7*x-30)
    a = 2
    b = 4.5

    Nmax = 100
    tol = 1e-7

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    (astar,ier,count) = bisection(f,fp,a,b,tol)
    (p,pstar,info,it) = newton(f,fp,astar,tol,Nmax)
    print('1. We know Newtons method would converge if viewed as a fixed  point method when |fprime(x)|<1')
    print('3. I changed the input to now include the fprime, this is because it is needed to check for convergence via Newtons method')
    print('5. The advantages of this modified method is that it allows for a greater range of initial values in order to find the root')
    print('    Some limitations of this are that it is more prone to make mistakes since we are sending it through multiple functions.')
    print('6a. Bisecftion with [a,b]=[2,4.5] resulted in 3.0000000000000000e+00 with 24 iterations')
    print('6b. Newtons method with x0 = 4.5 resulted in a root value of 3.000000014901161 with 27 iterations')
    print('6c. hybrid method with [a,b]=[2,4.5] resulted in', pstar)
    print('    the number of iterations used was', count)
    print('Overall, the hybrid method was the fastest and newtons was the most cost effective')
    

    # [astar,ier] = bisection(f,a,b,tol)
    # print('the approximate root is',astar)
    # print('the error message reads:',ier)
    # print('f(astar) =', f(astar))


# define routines
def bisection(f,fp,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      if abs(fp(d))<1:
         break
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier,count]
       
driver()

