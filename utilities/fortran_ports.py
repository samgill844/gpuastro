import numba 


@numba.njit(fastmath=True)
def sign(a,b) : 
    if b >= 0.0 : return abs(a)
    return -abs(a)


@numba.njit
def SameSign(a, b) : return ((a== b) & (a==0)) | (a*b>0)


@numba.njit(fastmath=True)
def clip(a, b, c):
    if (a < b)   : return b
    elif (a > c) : return c
    else         : return a
