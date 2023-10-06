import numpy as np
from numpy.linalg import norm 

def driver():
    x0 = np.array([1,1,1])
    Nmax = 100
    tol = 1e-7

    [xn,error] = iterate(x0, Nmax, tol)
    print('The found values of x,y,z are ', xn)

def evalF(x):
    F = np.zeros(3)
    F[0] = (((x[0])**2 + 4 * (x[1])**2 + 4 * (x[2])**2 - 16) * (2 * x[0])) / ((2 * x[0])**2 + (8 * x[1])**2 + (8 * x[2])**2)
    F[1] = (((x[0])**2 + 4 * (x[1])**2 + 4 * (x[2])**2 - 16) * (8 * x[1])) / ((2 * x[0]**2) + (8 * x[1])**2 + (8 * x[2])**2)
    F[2] = (((x[0])**2 + 4 * (x[1])**2 + 4 * (x[2])**2 - 16) * (8 * x[2])) / ((2 * x[0]**2) + (8 * x[1])**2 + (8 * x[2])**2)

    return F

def iterate(x0, Nmax, tol):
    xn = x0
    for its in range(Nmax):
        currxn = xn
        xn = xn - evalF(xn)
        error = norm(xn - currxn)

        print('the current error on', its, 'is', error)
        if norm(error) < tol:
            break
        

    return [xn, error]

driver()


# check
# def evalF(x): 

#     F = np.zeros(3)
    
#     F[0] = (((x[0])**2 + 4 * (x[1])**2 + 4 * (x[2])**2 - 16) * (2 * x[0])) / ((2 * x[0])**2 + (8 * x[1])**2 + (8 * x[2])**2)
#     F[1] = (((x[0])**2 + 4 * (x[1])**2 + 4 * (x[2])**2 - 16) * (8 * x[1])) / ((2 * x[0]**2) + (8 * x[1])**2 + (8 * x[2])**2)
#     F[2] = (((x[0])**2 + 4 * (x[1])**2 + 4 * (x[2])**2 - 16) * (8 * x[2])) / ((2 * x[0]**2) + (8 * x[1])**2 + (8 * x[2])**2)
    
#     return F

# x = np.array([1.09258435,1.36043465,1.36043465])

# print(evalF(x))