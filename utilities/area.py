import math, numba 
from gpuastro.utilities.fortran_ports import clip

@numba.njit(fastmath=True)
def area(z, r1, r2):
    if z==0. : return math.pi*(r1**2-r2**2)
    arg1 = clip((z*z + r1*r1 - r2*r2)/(2.*z*r1),-1,1)
    arg2 = clip((z*z + r2*r2 - r1*r1)/(2.*z*r2),-1,1)
    arg3 = clip(max((-z + r1 + r2)*(z + r1 - r2)*(z - r1 + r2)*(z + r1 + r2), 0.),-1,1)

    if   (r1 <= r2 - z) : return math.pi*r1*r1							                              # planet completely overlaps stellar circle
    elif (r1 >= r2 + z) : return math.pi*r2*r2						                                  # stellar circle completely overlaps planet
    else                : return r1*r1*math.acos(arg1) + r2*r2*math.acos(arg2) - 0.5*math.sqrt(arg3)  # partial overlap