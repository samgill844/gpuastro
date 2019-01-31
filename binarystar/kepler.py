import numba, numba.cuda
import math 
from math import fmod
import numpy as np
from gpuastro.binarystar.utils import *
from gpuastro.solvers import brent
from astropy import constants

__all__ = ['getEccentricAnomaly', 'getTrueAnomaly', 'get_z', 'getProjectedPosition', 'mass_function_1',  'd_mass_function', 't_ecl_to_peri', 'astrometry', 'stellar_density']

rho_sun = constants.M_sun.value / ((4/3)*np.pi*constants.R_sun.value**3)
G_const = constants.G.value

@numba.njit(fastmath=True)
def fmod(a,b) : return (a/b - math.floor(a/b))*b

##########################
# Keplers equations
##########################
@numba.njit(fastmath=True)
def kepler(M,E,e) : return M - E + e*math.sin(E)

@numba.njit(fastmath=True)
def dkepler(E,e) : return -1 + e*math.sin(E)

#@numba.njit
def stellar_density(P, radius_1, M2=0., R1=0.1): 
	return (((3*math.pi)/(G_const*(P*24*60*60)**2))*(1/radius_1)**3  - M2/(R1**3)) / rho_sun


##########################
# Eccentric anomaly
##########################
@numba.njit(fastmath=True)
def getEccentricAnomaly(M, e, Accurate_Eccentric_Anomaly=False, tol=1e-5):
        if Accurate_Eccentric_Anomaly:  
                # Instead, we will use transdendal function evaluations
                # to calculate the Eccentric anomaly.
                m = fmod(M, (2*math.pi))
                it = 0
                e1 = fmod( m, (2*math.pi)) + e*math.sin(m) + e*e*math.sin(2.0*m)/2.0
                test = 1.0
                e0=1.
                while (test > tol):
                        it +=1
                        e0 = e1
                        e1 = e0 + (m-(e0 - e*math.sin(e0)))/(1.0 - e*math.cos(e0))
                        test = abs(e1 - e0)

                

                if (e1 < 0) : e1 = e1 + 2*math.pi
                
                return e1     

        else:
                # calculates the eccentric anomaly (see Seager Exoplanets book:  Murray & Correia eqn. 5 -- see section 3)
                if (e == 0.0) : return M

                m = fmod(M, (2*math.pi))
                #m = M % 2*math.pi
                flip = 0
                if (m > math.pi) : m = 2*math.pi - m; flip = 1

                alpha = (3*math.pi + 1.6*(math.pi-math.fabs(m))/(1+e) )/(math.pi - 6/math.pi)
                d = 3*(1 - e) + alpha*e
                r = 3*alpha*d * (d-1+e)*m + m*m*m
                q = 2*alpha*d*(1-e) - m*m
                w = math.pow((math.fabs(r) + math.sqrt(q*q*q + r*r)),(2/3))
                E = (2*r*w/(w*w + w*q + q*q) + m) / d
                f_0 = E - e*math.sin(E) - m
                f_1 = 1 - e*math.cos(E)
                f_2 = e*math.sin(E)
                f_3 = 1-f_1
                d_3 = -f_0/(f_1 - 0.5*f_0*f_2/f_1)
                d_4 = -f_0/(f_1 + 0.5*d_3*f_2 + (d_3*d_3*d_3)*f_3/6)
                E = E -f_0/(f_1 + 0.5*d_4*f_2 + d_4*d_4*f_3/6 - d_4*d_4*d_4*f_2/24)

                if (flip==1) : E =  2*math.pi - E
                return E    


###################################
# True time of periastron passage
###################################
@numba.njit(fastmath=True)
def t_ecl_to_peri(t_ecl, e, w, incl, radius_1, p_sid, t_ecl_tolerance=1e-5, Accurate_t_ecl=False):

        # Define variables used
        efac  = 1.0 - e*2
        sin2i = math.sin(incl)**2

        # Value of theta for i=90 degrees
        theta_0 = (math.pi/2) - w;             # True anomaly at superior conjunction

        if (incl != math.pi/2.) and (Accurate_t_ecl == True) :  theta_0 =  brent(get_z_, theta_0-math.pi, theta_0 + math.pi,  (e, incl, w, radius_1), t_ecl_tolerance )
        if (theta_0 == math.pi) : ee = math.pi
        else : ee =  2.0 * math.atan(math.sqrt((1.-e)/(1.0+e)) * math.tan(theta_0/2.0))

        eta = ee - e*math.sin(ee)
        delta_t = eta*p_sid/(2*math.pi)
        return t_ecl  - delta_t
        


############################
# Get the true anomaly
############################
@numba.njit(fastmath=True)
def getTrueAnomaly(time, e, w, period,t_zero, incl=90., radius_1=0.2, t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9 ):
        # Sort inclination out
        incl = math.pi*incl / 180. 

        # Calcualte the mean anomaly
        M = 2*math.pi*fmod((time -  t_ecl_to_peri(t_zero, e, w, incl, radius_1, period, t_ecl_tolerance, Accurate_t_ecl)  )/period, 1.)

        # Calculate the eccentric anomaly
        E = getEccentricAnomaly(M, e, Accurate_Eccentric_Anomaly, E_tol)

        # Now return the true anomaly
        return 2.*math.atan(math.sqrt((1.+e)/(1.-e))*math.tan(E/2.))






@numba.njit(fastmath=True)
def get_z(nu, e, incl, w, radius_1) : return (1-e*e) * math.sqrt( 1.0 - math.sin(incl)*math.sin(incl)  *  math.sin(nu + w)*math.sin(nu + w)) / (1 + e*math.sin(nu)) /radius_1

@numba.njit(fastmath=True)
def get_z_(nu, z) : return get_z(nu, z[0], z[1], z[2], z[3])        
        

@numba.njit(fastmath=True)
def getProjectedPosition(nu, w, incl) : return math.sin(nu + w)*math.sin(incl)



####################
# Astrometry
####################
@numba.jit(fastmath=True)
def Xt(M, e, Accurate_Eccentric_Anomaly=False, E_tol=1e-5) : return math.cos(getEccentricAnomaly(M, e, Accurate_Eccentric_Anomaly=False, tol=1e-5)) - e
@numba.jit(fastmath=True)
def Yt(M, e, Accurate_Eccentric_Anomaly=False, E_tol=1e-5) : return math.sqrt(1-e**2) *  math.sin(getEccentricAnomaly(M, e, Accurate_Eccentric_Anomaly=False, tol=1e-5))
@numba.jit(fastmath=True)
def A(Ohm, omega, incl) : return math.cos(Ohm)*math.cos(omega) - math.sin(Ohm)*math.sin(omega)*math.cos(incl)
@numba.jit(fastmath=True)
def B(Ohm, omega, incl) : return math.sin(Ohm)*math.cos(omega) + math.cos(Ohm)*math.sin(omega)*math.cos(incl)
@numba.jit(fastmath=True)
def F(Ohm, omega, incl) : return -math.cos(Ohm)*math.sin(omega) - math.sin(Ohm)*math.cos(omega)*math.cos(incl)
@numba.jit(fastmath=True)
def G(Ohm, omega, incl) : return -math.sin(Ohm)*math.sin(omega) + math.cos(Ohm)*math.cos(omega)*math.cos(incl)


@numba.jit(fastmath=True)
def astrometry(time, Ohm=0.1, t_zero=0.0, period=1.0, fs=0.0, fc=0.0, seps=[1,0.2],  incl=0., radius_1=0.2, Accurate_t_ecl=False, t_ecl_tolerance=1e-5, Accurate_Eccentric_Anomaly=False, E_tol=1e-5):
        # Conversion
        e = fs*fs + fc*fc
        w = math.atan2(fs, fc)
        incl = math.pi*incl/180

        # Get the mean anomaly
        astro = math.zeros((2, len(time), 2))
        for i in range(len(time)):
                M = 2*math.pi*fmod((time [i]-  t_ecl_to_peri(t_zero, e, w, incl, radius_1, period, t_ecl_tolerance, Accurate_t_ecl)  )/period, 1.)
                X = Xt(M, e, Accurate_Eccentric_Anomaly, E_tol)
                Y = Yt(M, e, Accurate_Eccentric_Anomaly, E_tol)

                for j in range(2):
                     astro[j, i, 0] = seps[j]*(A(Ohm + j*math.pi, w + j*math.pi, incl)*X +    F(Ohm + j*math.pi, w + j*math.pi, incl)*Y)
                     astro[j, i, 1] = seps[j]*(B(Ohm + j*math.pi, w + j*math.pi, incl)*X +    G(Ohm + j*math.pi, w + j*math.pi, incl)*Y)
        return astro






####################
# Mass function
##################
@numba.njit(fastmath=True)
def mass_function_1(e, P, K1):
    G = 6.67408e-11
    return ((1-e**2)**1.5)*P*86400.1*((K1*10**3)**3)/(2*math.pi*G*1.989e30) 


@numba.njit(fastmath=True)
def d_mass_function(M2, z0):
    #M1, i, e, P, K1 = z0
    #return ((M2*math.sin(i))**3 / ((M1 + M2)**2)) - mass_function_1(e, P, K1)
    return ((M2*math.sin(z0[1]))**3 / ((z0[0] + M2)**2)) - mass_function_1(z0[2], z0[3], z0[4])
