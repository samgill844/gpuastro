# Imports
import numba, numba.cuda
import math, numpy as np 


@numba.njit
def Flux_drop_analytical_quadratic(z, k, c1, c2, tol):
    '''
    Calculate the analytical flux drop from the quadratic limb-darkening law.

    Parameters
    z : double
        Projected seperation of centers in units of stellar radii.
    k : double
        Ratio of the radii.
    c : double
        The first power-2 coefficient.
    a : double
        The second power-2 coefficient.
    f : double
        The flux from which to drop light from.
    eps : double
        Factor (1e-9)
    '''
    # Preliminary
    kap0 = 0.0
    kap1 = 0.0
    omega = 1.0 - c1/3.0 - c2/6.0

    # Check the corner cases
    if abs(k - z) < tol : z = k
    if abs(k - 1.0 - z) < tol : z = k - 1.0
    if abs(1.0- k - z) < tol  : z = 1.0 - k 
    if z < tol : z = 0.0 

    # Preliminary 
    x1 = (k-z)**2
    x2 = (k+z)**2
    x3 = k**2 - z**2

    # Source is unocculted
    if (z >= 1.0 + k) : return 1.0
    
    # Source is completely occullted
    if (k >= 1.0) and (z <= k-1.0):
        lambdad = 0.0
        etad = 0.5       # error in Fortran code corrected here, following Jason Eastman's python code
        lambdae = 1.0
        return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*(lambdad + 2.0/3.0) + c2*etad)/omega;

    # Source is partly occulted and occulted object crsses the limb
    if (z >= abs(1.0 - k)) and (z <= 1.0 + k):
        kap1 = math.acos(min((1.0 - k*k + z*z)/2.0/z, 1.0))
        kap0 = math.acos(min((k*k + z*z - 1.0)/2.0/k/z, 1.0))
        lambdae = k*k*kap0 + kap1
        lambdae = (lambdae - 0.50*math.sqrt(max(4.0*z*z - (1.0 + z*z - k*k)**2.0, 0.0)))/math.pi


    # Edge f the occulting star lies at the origin
    if z==k:
        if z < 0.5:
            q = 2.0*k
            Kk = ellk(q)
            Ek = ellec(q)
            lambdad = 1.0/3.0 + 2.0/9.0/math.pi*(4.0*(2.0*k*k - 1.0)*Ek + (1.0 - 4.0*k*k)*Kk)
            etad = k*k/2.0*(k*k + 2.0*z*z);
            return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega

        elif z > 0.5:
            q = 0.5/k
            Kk = ellk(q)
            Ek = ellec(q)
            lambdad = 1.0/3.0 + 16.0*k/9.0/math.pi*(2.0*k*k - 1.0)*Ek -  (32.0*k**4 - 20.0*k*k + 3.0)/9.0/math.pi/k*Kk
            etad = 1.0/2.0/math.pi*(kap1 + k*k*(k*k + 2.0*z*z)*kap0 - (1.0 + 5.0*k*k + z*z)/4.0*math.sqrt((1.0 - x1)*(x2 - 1.0)))           

        else:
            lambdad = 1.0/3.0 - 4.0/math.pi/9.0
            etad = 3.0/32.0
            return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega
        return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega

    if ((z > 0.5 + abs(k - 0.5) and z < 1.0 + k) or (k > 0.5) and (z > abs(1.0 - k)) and (z < k)):
        q = math.sqrt((1.0 - x1)/4.0/z/k);
        Kk = ellk(q)
        Ek = ellec(q)
        n = 1.0/x1 - 1.0
        Pk = ellpic_bulirsch(n, q)
        lambdad = 1.0/9.0/math.pi/math.sqrt(k*z)*(((1.0 - x2)*(2.0*x2 + x1 - 3.0) - 3.0*x3*(x2 - 2.0))*Kk + 4.0*k*z*(z*z + 7.0*k*k - 4.0)*Ek - 3.0*x3/x1*Pk)
        if(z < k) : lambdad += 2.0/3.0 
        etad = 1.0/2.0/math.pi*(kap1 + k*k*(k*k + 2.0*z*z)*kap0 - (1.0 + 5.0*k*k + z*z)/4.0*math.sqrt((1.0 - x1)*(x2 - 1.0)))
        return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega               

    if (k <= 1.0) and (z <= 1.0-k):
        etad = k*k/2.0*(k*k + 2.0*z*z)
        lambdae = k*k

        q = math.sqrt((x2 - x1)/(1.0 - x1))
        Kk = ellk(q)
        Ek = ellec(q)
        n = x2/x1 - 1.0
        Pk = ellpic_bulirsch(n, q)    

        lambdad = 2.0/9.0/math.pi/math.sqrt(1.0 - x1)*((1.0 - 5.0*z*z + k*k + x3*x3)*Kk + (1.0 - x1)*(z*z + 7.0*k*k - 4.0)*Ek - 3.0*x3/x1*Pk)

        if(abs(k + z - 1.0) <= tol) :  lambdad = 2.0/3.0/math.pi*math.acos(1.0 - 2.0*k) - 4.0/9.0/math.pi*math.sqrt(k*(1.0 - k))*(3.0 + 2.0*k - 8.0*k*k)
        if(z < k) :  lambdad += 2.0/3.0

    return 1.0 - ((1.0 - c1 - 2.0*c2)*lambdae + (c1 + 2.0*c2)*lambdad + c2*etad)/omega



@numba.njit
def ellpic_bulirsch(n, k):
    kc = math.sqrt(1.-k*k)
    p = math.sqrt(n + 1.)
    m0 = 1.
    c = 1.
    d = 1./p
    e = kc

    nit = 0
    while nit < 10000:
        f = c
        c = d/p + c
        g = e/p
        d = 2.*(f*g + d)
        p = g + p
        g = m0
        m0 = kc + m0
        if abs(1.-kc/g) > 1.0e-8:
            kc = 2.*math.sqrt(e)
            e = kc*m0
        else : return 0.5*math.pi*(c*m0+d)/(m0*(m0+p))
        nit+=1      
    return 0.

@numba.njit
def ellec(k):
    # Computes polynomial approximation for the complete elliptic
    # integral of the first kind (Hasting's approximation):
    m1 = 1.0 - k*k
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639
    ee1 = 1.0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ee2 = m1*(b1 + m1*(b2 + m1*(b3 + m1*b4)))*math.log(1.0/m1)
    ellec = ee1 + ee2
    return ellec

@numba.njit
def ellk(k):
    # Computes polynomial approximation for the complete elliptic
    # integral of the second kind (Hasting's approximation):
    m1 = 1.0 - k*k
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012
    ek1 = a0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ek2 = (b0 + m1*(b1 + m1*(b2 + m1*(b3 + m1*b4))))*math.log(m1)
    ellk = ek1 - ek2
    return ellk



