import numba 
import math

@numba.njit
def SameSign(a, b) : return ((a== b) & (a==0)) | (a*b>0)

@numba.njit
def bisector(func, xlow, xhigh, tol, z0):
    # Check bounds
    Llow = func(xlow, z0)
    Lhigh = func(xhigh, z0)
    if SameSign(Llow, Lhigh) : return math.nan

    err = math.inf 

    while err > tol : 
        Llow = func(xlow, z0)
        Lmid = func(xlow + (xhigh - xlow)/2.0 , z0)
        Lhigh = func(xhigh, z0)

        # Select the best half for the next bisection 
        if (SameSign(Llow, Lmid) and not SameSign(Lmid, Lhigh)):
            # The solution is in the first half
            xlow = xlow + (xhigh - xlow)/2.0
            err = abs(Lmid - Llow)
        else:
            xhigh = xlow + (xhigh - xlow)/2.0
            err = abs(Lmid - Lhigh)
        
    return xlow + (xhigh - xlow)/2.0

        




