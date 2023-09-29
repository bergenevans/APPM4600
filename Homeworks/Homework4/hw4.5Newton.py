import matplotlib.pyplot as plt
import numpy as np

# modified newton_example.py provided on canvas
def driver():

  f = lambda x: x**6-x-1
  fp = lambda x: 6*x**5-1
  p0 = 2

  Nmax = 100
  tol = 1.e-14
  # for the second part of the problem
  return newton(f,fp,p0,tol,Nmax)

  # for the first part of the problem
  # (p,pstar,info,it,e) = newton(f,fp,p0,tol, Nmax)
  # print('the approximate root is', '%16.16e' % pstar)
  # print('the error message reads:', '%d' % info)
  # print('Number of iterations:', '%d' % it)
  # print(e)


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
  e = np.zeros((Nmax,1)) # this is for the errors
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      e[it] = abs(p1-p0)
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it,e]
      p0 = p1
  pstar = p1
  info = 1
  
  return [p,pstar,info,it,e]

driver()

# for the second part of the problem
def plot(p,alpha):
  xk = np.abs(p[:-1] - alpha)
  xk1 = np.abs(p[1:] - alpha)

  # kept getting errors for 0 values so this ensured that everything that was plotted was positive
  maskxk = (xk[:-1]>0)
  maskxk1= (xk1[:-1]>0)
  pos_xk = xk[:-1][maskxk]
  pos_xk1 = xk1[:-1][maskxk1]

  # print(xk[:-1])
  # print(xk1[:-1])
  # print(np.log(pos_xk[:-1]))
  # print(np.log(pos_xk1[:-1]))

  m, b = np.polyfit(np.log(pos_xk[:-1]),np.log(pos_xk1[:-1]),1) #slope

  plt.loglog(xk,xk1)
  plt.xlabel('x_k - alpha')
  plt.ylabel('x_(k+1)-alpha')
  plt.show()

  return m

(p,pstar,info,it,e) = driver()
m = plot(p,pstar)
print('The slope is', m)