import numba 
import math
import numpy as np

#@numba.njit
def ctoh1(c1, c2) : return 1 - c1*(1 - 2**(-c2))    

#@numba.njit
def ctoh2(c1, c2) : return c1*2**(-c2)    

#@numba.njit
def htoc1(h1, h2) : return 1 - h1 + h2
import numba 

#@numba.njit
def htoc2(c1, h2) : return np.log2(c1 / h2)

@numba.njit(fastmath=True)
def ctoh1_(c1, c2) : return 1 - c1*(1 - 2**(-c2))    

@numba.njit(fastmath=True)
def ctoh2_(c1, c2) : return c1*2**(-c2)    

@numba.njit(fastmath=True)
def htoc1_(h1, h2) : return 1 - h1 + h2

@numba.njit(fastmath=True)
def htoc2_(c1, h2) : return np.log2(c1 / h2)

@numba.njit(fastmath=True)
def htoc2__(c1, h2):
	x = c1 / h2
	return ((x-1) - (x-1)**2 /2 + (x-1)**3/3) / math.log(2.)