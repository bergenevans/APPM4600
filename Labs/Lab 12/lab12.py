# get lgwts routine and numpy
from gauss_legendre import *
from scipy.integrate import quad
# instead of ^^ can do:

# modified adaptive_quad.py code provided on canvas
# using gauss_legendre.py code provided on canvas

# adaptive quad subroutines
# the following three can be passed
# as the method parameter to the main adaptive_quad() function

def driver():
  def f(x):
    return np.sin(1/x)

  a = 0.1
  b = 2
  M = 5
  tol = 1e-3
  actual = quad(f,a,b)

  print('Actual value:', actual[0])

  # this is returning that they only have one iteration?
  trap_break = 0
  simp_break = 0
  gauss_break = 0
  for j in range(1,100):
    trap = eval_composite_trap(j,a,b,f)
    simp = eval_composite_simpsons(j,a,b,f)
    [I_hat,x,w] = eval_gauss_quad(j,a,b,f)
    diff_trap = abs(trap - actual[0])
    diff_simp = abs(simp - actual[0])
    diff_gauss = abs(I_hat - actual[0])
    if diff_trap < tol and trap_break == 0:
      trap_break = j
      print('Composite Trapezoidal:', trap)
      print('Tolerance hit at ', j)
    if diff_simp < tol and simp_break == 0:
      simp_break = j
      print('Composite Simpson', simp)
    if diff_gauss < tol and gauss_break == 0:
      gauss_break = j
      print('gauss quad: ', I_hat)

  print('Composite Trapezoidal has this many iterations at tol break:',trap_break)
  print('Composite Simpsons has this many iterations at tol break:',simp_break)
  print('Gauss guad has this many iterations at tol break:',gauss_break)

def eval_composite_trap(M,a,b,f):
  """
  put code from prelab with same returns as gauss_quad
  you can return None for the weights
  """
  h = (b-a)/M
  t = 0
  for j in range(1,M):
    xj = a + j*h
    t = t + 2*f(xj)
  trap = h/2 * (f(a)+ t +f(b))
  # print(trap)
  return trap
  # diff = trap[j+1] - trap[j]

def eval_composite_simpsons(M,a,b,f):
  """
  put code from prelab with same returns as gauss_quad
  you can return None for the weights
  """
  h = (b-a)/M
  s1 = 0
  s2 = 0

  for j in range(1,M):
    xj = a + j*h
    if j % 2 == 0:
        s1 = s1 + 2*f(xj)
    else:
        s2 = s2 + 4*f(xj)
  simp = h/3 * (f(a) + s1 + s2 + f(b))
  # print(simp)
  return simp
  # diff = simp[j+1] - simp[j]

def eval_gauss_quad(M,a,b,f):
  """
  Non-adaptive numerical integrator for \int_a^b f(x)w(x)dx
  Input:
    M - number of quadrature nodes
    a,b - interval [a,b]
    f - function to integrate
  
  Output:
    I_hat - approx integral
    x - quadrature nodes
    w - quadrature weights

  Currently uses Gauss-Legendre rule
  """
  
  x,w = lgwt(M,a,b) 
  I_hat = np.sum(f(x)*w)
  return I_hat,x,w

def adaptive_quad(a,b,f,tol,M,method):
  """
  Adaptive numerical integrator for \int_a^b f(x)dx
  
  Input:
  a,b - interval [a,b]
  f - function to integrate
  tol - absolute accuracy goal
  M - number of quadrature nodes per bisected interval
  method - function handle for integrating on subinterval
        - eg) eval_gauss_quad, eval_composite_simpsons etc.
  
  Output: I - the approximate integral
          X - final adapted grid nodes
          nsplit - number of interval splits
  """
  # 1/2^50 ~ 1e-15
  maxit = 50
  left_p = np.zeros((maxit,))
  right_p = np.zeros((maxit,))
  s = np.zeros((maxit,1))
  left_p[0] = a; right_p[0] = b;
  # initial approx and grid
  s[0],x,_ = method(M,a,b,f);
  # save grid
  X = []
  X.append(x)
  j = 1;
  I = 0;
  nsplit = 1;
  while j < maxit:
    # get midpoint to split interval into left and right
    c = 0.5*(left_p[j-1]+right_p[j-1]);
    # compute integral on left and right spilt intervals
    s1,x,_ = method(M,left_p[j-1],c,f); X.append(x)
    s2,x,_ = method(M,c,right_p[j-1],f); X.append(x)
    if np.max(np.abs(s1+s2-s[j-1])) > tol:
      left_p[j] = left_p[j-1]
      right_p[j] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j] = s1
      left_p[j-1] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j-1] = s2
      j = j+1
      nsplit = nsplit+1
    else:
      I = I+s1+s2
      j = j-1
      if j == 0:
        j = maxit
  return I,np.unique(X),nsplit

driver()