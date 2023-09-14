# import libraries
import numpy as np


# modified from bisection_example.py provided on canvas
def driver():

# use routines    
    f = lambda x: np.sin(x)
    a = 0.5
    b = 3*np.pi/4

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-5

    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))




# define routines
def bisection(f,a,b,tol):
    
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
    return [astar, ier]
      
driver()  


# print('1.a approx root of 0.999, no error message, and a f(star) = -2.98e-08')
# print('1.b approx root of -1, an error message, and a f(star) = -2')
# print('1.b Method b was not successful because the value for b was too small')
# print('1.c approx root of 0.999, no error message, and a f(star) = -2.98e-08')

# print('1. It is possible to for bisection to find the root x = 0 because it exists within the bounds for c, which was successful')

# print('This is the behaivor I would expect because the second a and b values for sin(x) might just be out of range')