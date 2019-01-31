import numba, numba.cuda 
import math


#####################################################################
# BATMAN limb-darkening functions
#####################################################################
@numba.njit(fastmath=True)
def linear(x, ldc):
    x = math.sqrt(1-x**2)
    return 1 - ldc[0]*(1 - x)

@numba.njit(fastmath=True)
def quadratic(x, ldc):
    x = math.sqrt(1-x**2)
    return 1 - ldc[0]*(1 - x) - ldc[1]*(1 - x)**2

@numba.njit(fastmath=True)
def sqrt(x, ldc):
    x = math.sqrt(1-x**2)
    return 1 - ldc[0]*(1 - x) - ldc[1]*(1 - math.sqrt(x))

@numba.njit(fastmath=True)
def logarithmic(x, ldc):
    x = math.sqrt(1-x**2)
    return 1 - ldc[0]*(1 - x) - ldc[1]*x*math.log(x)

@numba.njit(fastmath=True)
def exponential(x, ldc):
    x = math.sqrt(1-x**2)
    return 1 - ldc[0]*(1 - x) - ldc[1]/(1 - math.exp(x))

@numba.njit(fastmath=True)
def sing(x, ldc):
    x = math.sqrt(1-x**2)
    return 1 - ldc[0]*(1 - x**0.5) - ldc[1]*(1 - x**1.5) -ldc[2]*(1 - x**2)

@numba.njit(fastmath=True)
def claret(x, ldc):
    x = math.sqrt(1-x**2)
    return 1 - ldc[0]*(1 - x**0.5) - ldc[1]*(1 - x) - ldc[2]*(1 - x**1.5) -ldc[3]*(1 - x**2)
