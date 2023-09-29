import matplotlib.pyplot as plt
import numpy as np

def driver():

  f = lambda x: x**6-x-1
  p0 = 2
  p1 = 1

  Nmax = 100
  tol = 1.e-14
  # for the second part of the problem
  return secant(f,p0,p1,tol,Nmax)

  # for the first part of the problem:
  # (p,pstar,ier,j,e) = secant(f,p0,p1,tol, Nmax)
  # print('the approximate root is', '%16.16e' % pstar)
  # print('the error message reads:', '%d' % ier)
  # print('Number of iterations:', '%d' % j)
  # print(e)


def secant(f,p0,p1,tol,Nmax):

  e = np.zeros((Nmax,1)) # this is for the errors
  p = np.zeros(Nmax+1);
  p[0] = p0
  p[1] = p1
  ier = 1
  if abs(f(p0))==0:
    pstar = p0
    ier = 0
    return [p,p0,ier,j,e]
  if abs(f(p1))==0:
    pstar = p1
    ier = 0
    return [p,p1,ier,j,e]
  fp1 = f(p1)
  fp0 = f(p0)

  for j in range(Nmax):
    p2 = p1 - ((f(p1)*(p1-p0))/(f(p1)-f(p0)))
    e[j] = abs(p2-p1)
    if abs(p2-p1) < tol:
      pstar = p2
      ier = 0
      return [p,p2,ier,j,e]
    p[j + 2] = p2
    p0 = p1
    p1 = p2
    fp0 = fp1
    fp1 = f(p1)
  pstar = p2
  ier = 1 
  return [p,p1,ier,j,e]

driver()

# for the second part of the problem
def plot(p,alpha):
  xk = np.abs(p[:-1] - alpha)
  xk1 = np.abs(p[1:] - alpha)

  m, b = np.polyfit(np.log(xk),np.log(xk1),1) #slope

  plt.loglog(xk,xk1)
  plt.xlabel('x_k - alpha')
  plt.ylabel('x_(k+1)-alpha')
  plt.show()

  return m

(p,pstar,ier,j,e) = driver()
m = plot(p,pstar)
print('The slope is', m)