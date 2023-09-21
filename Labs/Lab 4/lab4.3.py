import numpy as np


# modified code from fixedpt_example.py provided on canvas
def driver():

# test functions 
     f1 = lambda x: (10/(x+4))**0.5

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     # print(fixedpt(f1,x0,tol,Nmax))
     print(Aitken(f1,tol,Nmax))

# define routines
# def fixedpt(f,x0,tol,Nmax):

#     ''' x0 = initial guess''' 
#     ''' Nmax = max number of iterations'''
#     ''' tol = stopping tolerance'''
#     X = np.zeros((Nmax,1))
#     X[0] = x0
#     count = 0
#     while (count <Nmax):
#        count = count +1
#        x1 = f(x0)
#        X[count] = x1
#        if (abs(x1-x0) <tol):
#           xstar = x1
#           ier = 0
#           return X
#        x0 = x1

#     xstar = x1
#     ier = 1
#     return X
    
def Aitken(f,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    P_at = np.zeros((Nmax,1))
    for i in range[Nmax]:
        P_n1 = f(P_at)
        P_n2 = f(P_n1)
        P_at = P_at - ((P_n1-P_at)**2/(P_n1-2*P_n1+P_at))

        if abs[P_n2-P_n1] < tol:
            return P_at
    count = 0
    while (count <Nmax):
       count = count +1
       p1 = f(p0)
       P_at[count] = p1
       if (abs(p1-p0) <tol):
          xstar = p1
          ier = 0
          return P_at
       p0 = p1

    xstar = p1
    ier = 1
    return P_at
    
driver()