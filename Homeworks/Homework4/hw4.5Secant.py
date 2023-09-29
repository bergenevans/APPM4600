import matplotlib.pyplot as plt
import numpy as np

def driver():

  f = lambda x: x**6-x-1
  p0 = 2
  p1 = 1

  Nmax = 100
  tol = 1.e-14

  (p,pstar,ier,j,e) = secant(f,p0,p1,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % ier)
  print('Number of iterations:', '%d' % j)
  print(e)


def secant(f,p0,p1,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f - function and derivative
    p0,p1   - initial guesses for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    ier  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  e = np.zeros((Nmax,1))
  p = np.zeros(Nmax+1);
  p[0] = p0
  if abs(f(p0))==0:
    pstar = p0
    ier = 0
    return
  if abs(f(p1))==0:
    pstar = p1
    ier = 0
    return
  fp1 = f(p1)
  fp0 = f(p0)

  for j in range(Nmax):
    if abs(fp1-fp0)==0:
      ier = 1
      pstar = p1
      return
    p2 = p1 - ((f(p1)*(p1-p0))/(f(p1)-f(p0)))
    if abs(p2-p1) < tol:
      pstar = p2
      ier = 0
      return
    p0 = p1
    fp0 = fp1
    p1 = p2
    fp2 = f(p2)
  pstar = p2
  ier = 1 
  return [p,pstar,ier,j,e]

driver()