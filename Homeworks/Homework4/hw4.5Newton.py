import matplotlib.pyplot as plt
import numpy as np

# modified newton_example.py provided on canvas
def driver():

  f = lambda x: x**6-x-1
  fp = lambda x: 6*x**5-1
  p0 = 2

  Nmax = 100
  tol = 1.e-14

  (p,pstar,info,it,e) = newton(f,fp,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)
  print(e)


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
  e = np.zeros((Nmax,1))
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      e[it] = (p1-p0)
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it,e]
      p0 = p1
  pstar = p1
  info = 1
  
  return [p,pstar,info,it,e]

driver()
