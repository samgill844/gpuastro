import numba, numba.cuda 
from gpuastro.binarystar.utils import *
import math, numpy as np

__all__ = ['Flux_drop_analytical_power_2', 'Flux_drop_analytical_quadratic', 'Flux_drop_BATMAN', 'Flux_drop_analytical_uniform' ]

##########################################
# Uniform
##########################################
@numba.njit(fastmath=True)
def Flux_drop_analytical_uniform( z, k, SBR, f):
		if(z >= 1. + k) : return f		                  # no overlap
		if(z >= 1.) and ( z <= k - 1.) : return 0.0;      # total eclipse of the star
		elif (z <= 1. - k) : return f - SBR*k*k	          # planet is fully in transit		
		else:						                      # planet is crossing the limb
			kap1 = math.acos(min((1. - k*k + z*z)/2./z, 1.))
			kap0 = math.acos(min((k*k + z*z - 1.)/2./k/z, 1.))
			return f - SBR*  (k*k*kap0 + kap1 - 0.5*math.sqrt(max(4.*z*z - math.pow(1. + z*z - k*k, 2.), 0.)))/math.pi


            

##########################################
# QPOWER-2
##########################################

@numba.njit(fastmath=True)
def q1(z, p, c, a, g, I_0):
	zt = clip(abs(z), 0,1-p)
	s = 1-zt*zt
	c0 = (1-c+c*math.pow(s,g))
	c2 = 0.5*a*c*math.pow(s,(g-2))*((a-1)*zt*zt-1)
	return 1-I_0*math.pi*p*p*(c0 + 0.25*p*p*c2 - 0.125*a*c*p*p*math.pow(s,(g-1)))



@numba.njit(fastmath=True)
def q2(z, p, c, a, g, I_0, eps):
	zt = clip(abs(z), 1-p,1+p)
	d = clip((zt*zt - p*p + 1)/(2*zt),0,1)
	ra = 0.5*(zt-p+d)
	rb = 0.5*(1+d)
	sa = clip(1-ra*ra,eps,1)
	sb = clip(1-rb*rb,eps,1)
	q = clip((zt-d)/p,-1,1)
	w2 = p*p-(d-zt)*(d-zt)
	w = math.sqrt(clip(w2,eps,1))
	c0 = 1 - c + c*math.pow(sa,g)
	c1 = -a*c*ra*math.pow(sa,(g-1))
	c2 = 0.5*a*c*math.pow(sa,(g-2))*((a-1)*ra*ra-1)
	a0 = c0 + c1*(zt-ra) + c2*(zt-ra)*(zt-ra)
	a1 = c1+2*c2*(zt-ra)
	aq = math.acos(q)
	J1 =  (a0*(d-zt)-(2./3.)*a1*w2 + 0.25*c2*(d-zt)*(2.0*(d-zt)*(d-zt)-p*p))*w + (a0*p*p + 0.25*c2*math.pow(p,4))*aq 
	J2 = a*c*math.pow(sa,(g-1))*math.pow(p,4)*(0.125*aq + (1./12.)*q*(q*q-2.5)*math.sqrt(clip(1-q*q,0.0,1.0)) )
	d0 = 1 - c + c*math.pow(sb,g)
	d1 = -a*c*rb*math.pow(sb,(g-1))
	K1 = (d0-rb*d1)*math.acos(d) + ((rb*d+(2./3.)*(1-d*d))*d1 - d*d0)*math.sqrt(clip(1-d*d,0.0,1.0))
	K2 = (1/3)*c*a*math.pow(sb,(g+0.5))*(1-d)
	if J1 > 1 : J1 = 0
	return 1 - I_0*(J1 - J2 + K1 - K2)

@numba.njit(fastmath=True)
def Flux_drop_analytical_power_2(z, k, c, a, f, eps):
    '''
    Calculate the analytical flux drop por the power-2 law.

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
    I_0 = (a+2)/(math.pi*(a-c*a+2))
    g = 0.5*a

    if (z < 1-k) : return q1(z, k, c, a, g, I_0)
    elif (abs(z-1) < k) : return q2(z, k, c, a, g, I_0, eps)
    else: return 1.0

#@numba.njit
def ctoh1(c1, c2) : return 1 - c1*(1 - 2**(-c2))    

#@numba.njit
def ctoh2(c1, c2) : return c1*2**(-c2)    

#@numba.njit
def htoc1(h1, h2) : return 1 - h1 + h2

#@numba.njit
def htoc2(c1, h2) : return np.log2(c1 / h2)



##############################################################
# BATMAN
##############################################################
@numba.njit(fastmath=True)
def Flux_drop_BATMAN(z, k, limb_darkening_function, ldc,  N , fac = 5.0e-4):
    # Limits for integration
    x_in = max(z - k, 0.);					# lower bound for integration
    x_out = min(z + k, 1.0);				# upper bound for integration

    if x_in > 1. : return 1. 
    elif ((x_out - x_in) < 1e-7)  : return 1. 
    else:
        # Parameters for integration
        delta = 0.;						# variable to store the integrated intensity, \int I dA
        x = x_in;						# starting radius for integration
        dx = fac*math.acos(x); 			# initial step size    

        x += dx                         # first step

        A_i = 0. 						# initial area

        while x < x_out:
            A_f = area(z, x, k);				        # calculates area of overlapping circles
            I = limb_darkening_function(x - dx/2., ldc); 	# intensity at the midpoint
            delta += (A_f - A_i)*I;				        # increase in transit depth for this integration step
            dx = fac*math.acos(x);  				            # updating step size
            x = x + dx;					                # stepping to next element
            A_i = A_f;					                # storing area   

        dx = x_out - x + dx;  					           # calculating change in radius for last step  FIXME
        x = x_out    					                   # final radius for integration
        A_f = area(z, x, k)					           # area for last integration step
        I = limb_darkening_function(x - dx/2., ldc)           # intensity at the midpoint
        delta += (A_f - A_i)*I 					           # increase in transit depth for this integration step

        return 1.0 - delta              # flux equals 1 - \int I dA


################################################################
# QUADRATIC
################################################################
@numba.njit(fastmath=True)
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



@numba.njit(fastmath=True)
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

@numba.njit(fastmath=True)
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

@numba.njit(fastmath=True)
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