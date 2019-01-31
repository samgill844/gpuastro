import numba, numba.cuda
import math 
import numpy as np

from gpuastro.binarystar.utils import *
from gpuastro.binarystar.kepler import *
from gpuastro.binarystar.flux_drop import *
from gpuastro.binarystar.limb_darkening_laws import quadratic # to stop newer version of numba complaining
N_BATCH = 10240

if numba.cuda.is_available() : __all__ = ['lc', 'lc_gpu', 'lc_loglike']
else  : __all__ = ['lc', 'lc_loglike']



@numba.njit(fastmath=True)
def _lc(time,\
        radius_1, k, w, e, incl,\
        period, t_zero,\
        ldc_1_1, ldc_1_2, ld_law_1,\
        ld_1_func, N_BATMAN, FAC_BATMAN,\
        SBR, light_3,\
        t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol,\
        LC, N_start):

    for i in numba.prange(time.shape[0]):
        # Get true anomaly
        nu = getTrueAnomaly(time[i], e, w, period,t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol )

        # Get projected seperation0
        z = get_z(nu, e, incl, w, radius_1)

        # Initiase the flux
        l = 1.

        # At this point, we might check if the distance between
        if (z < (1.0+ k)):

            # So it's eclipsing? Lets see if its primary or secondary by
            # its projected motion!
            f = getProjectedPosition(nu, w, incl)

            if (f > 0): 
                # Calculate the flux drop for a priamry eclipse
                if ld_law_1 == 0 : l =  Flux_drop_analytical_power_2(z, k, ldc_1_1, ldc_1_2, l, 1e-5)    # Primary eclipse
                #elif ld_law_1 == 1 : l =  Flux_drop_analytical_quadratic(z, k, ldc_1[0], ldc_1[1], 1e-14)
                #elif ld_law_1 == 2 : l =  Flux_drop_BATMAN(z, k, ld_1_func, ldc_1, N_BATMAN, FAC_BATMAN)
                else: l = 1.0

                # Dont forget about the third light from the companion if SBR > 0
                if (SBR > 0) : l = l/(1. + k*k*SBR) + (1.-1.0/(1 + k*k*SBR))
            elif (SBR>0) : l =  Flux_drop_analytical_uniform(z, k, SBR, l) # Secondary eclipse

            # Now account for third light
            if (light_3 > 0.0) : l = l/(1. + light_3) + (1.-1.0/(1. + light_3))

        LC[N_start + i] = l 


@numba.njit(fastmath=True)
def __lc(time, mag, mag_err, mag_zp, mag_jitter,\
        radius_1, k, w, e, incl,\
        period, t_zero,\
        ldc_1_1, ldc_1_2, ld_law_1,\
        ld_1_func, N_BATMAN, FAC_BATMAN,\
        SBR, light_3,\
        t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol):
    L = 0.
    for i in range(time.shape[0]):
        # Get true anomaly
        nu = getTrueAnomaly(time[i], e, w, period,t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol )

        # Get projected seperation0
        z = get_z(nu, e, incl, w, radius_1)

        # Initiase the flux
        l = 1.

        # At this point, we might check if the distance between
        if (z < (1.0+ k)):

            # So it's eclipsing? Lets see if its primary or secondary by
            # its projected motion!
            f = getProjectedPosition(nu, w, incl)

            if (f > 0): 
                # Calculate the flux drop for a priamry eclipse
                if ld_law_1 == 0 : l =  Flux_drop_analytical_power_2(z, k, ldc_1_1, ldc_1_1, l, 1e-5)    # Primary eclipse
                #elif ld_law_1 == 1 : l =  Flux_drop_analytical_quadratic(z, k, ldc_1[0], ldc_1[1], 1e-14)
                #elif ld_law_1 == 2 : l =  Flux_drop_BATMAN(z, k, ld_1_func, ldc_1, N_BATMAN, FAC_BATMAN)
                else: l = 1.0

                # Dont forget about the third light from the companion if SBR > 0
                if (SBR > 0) : l = l/(1. + k*k*SBR) + (1.-1.0/(1 + k*k*SBR))
            elif (SBR>0) : l =  Flux_drop_analytical_uniform(z, k, SBR, l) # Secondary eclipse

            # Now account for third light
            if (light_3 > 0.0) : l = l/(1. + light_3) + (1.-1.0/(1. + light_3))

        wt = 1.0/(mag_err[i]**2 + mag_jitter**2)
        l = mag_zp - 2.5*math.log10(l) # convert to mag
        L -= (l - mag[i])**2*wt - math.log(wt)
    return L

@numba.njit(fastmath=True)
def lc(time,\
        radius_1=0.2, k=0.2, fs=0.0, fc=0.0,\
        incl=90., period=1.0, t_zero=0.0,\
        ldc_1_1 = 0.8, ldc_1_2 = 0.8, ld_law_1 = 0,\
        ld_1_func=-1, N_BATMAN=1000, FAC_BATMAN=5.0e-4,\
        SBR = 0.0, light_3 = 0.0,\
        t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9):
    '''
    Return the lightcurve model. 

    ld_law_1 : int
        The limb darkening law to use
        0 = Analytical power 2
        1 = Analytical quadratic
        2 = Batman integrator
            If this option is chosen, then a limb-darkening function from
            gpuastro.binarystar.limb_darkening_laws needs to be passed 
            to ld_1_func.
    '''
    # Unpack args
    w = math.atan2(fs, fc)
    e = fs*fs + fc*fc
    incl = math.pi*incl/180

    # Allocate the LC array
    LC = np.ones(len(time))
    _lc(time, radius_1, k, w, e, incl, period, t_zero, ldc_1_1, ldc_1_2, ld_law_1, ld_1_func, N_BATMAN, FAC_BATMAN,  SBR, light_3,    t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, LC, 0)
    return LC


@numba.njit(fastmath=True)
def lc_loglike(time, mag, mag_err, mag_jitter, mag_zp, radius_1=0.2, k=0.2, fs=0.0, fc=0.0, incl=90., period=1.0, t_zero=0.0, ldc_1_1=0.8, ldc_1_2=0.8, ld_law_1 = 0, ld_1_func=-1, N_BATMAN=1000, FAC_BATMAN=5.0e-4,  SBR = 0.0, light_3 = 0.0,              t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9):
    '''
    Return the lightcurve model. 

    ld_law_1 : int
        The limb darkening law to use
        0 = Analytical power 2
        1 = Analytical quadratic
        2 = Batman integrator
            If this option is chosen, then a limb-darkening function from
            gpuastro.binarystar.limb_darkening_laws needs to be passed 
            to ld_1_func.
    '''
    # Unpack args
    w = math.atan2(fs, fc)
    e = fs*fs + fc*fc
    incl = math.pi*incl/180

    # Allocate the LC array
    LC = np.ones(len(time))
    __lc(time,mag, mag_err, mag_zp, mag_jitter, radius_1, k, w, e, incl, period, t_zero, ldc_1_1, ldc_1_2, ld_law_1, ld_1_func, N_BATMAN, FAC_BATMAN,  SBR, light_3,    t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol)
    return LC




####################
# GPU functions
##################
if numba.cuda.is_available():
    @numba.cuda.jit
    def _lc_gpu(time, mag, mag_err, mag_zp, mag_jitter, radius_1, k, fs, fc, incl, period, t_zero, ldc_1, ld_law_1, ld_1_func, N_BATMAN, FAC_BATMAN,  SBR, light_3,    t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, LC):
        # Get the thread I.D.
        j = numba.cuda.grid(1)
        w = math.atan2(fs[j], fc[j])
        e = fs[j]*fs[j] + fc[j]*fc[j]
        incl_ = math.pi*incl[j]/180

        if mag_jitter[0]==-99.: 
            _lc(time, mag, mag_err, mag_zp[0], mag_jitter[0],\
                radius_1[j], k[j], w, e, incl_, period[j], t_zero[j],\
                ldc_1[j*2:j*2+2], ld_law_1,\
                ld_1_func, N_BATMAN, FAC_BATMAN,\
                SBR[j], light_3[j],    t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol,\
                LC, j*time.shape[0])
        
        else:
            _lc(time, mag, mag_err, mag_zp[j], mag_jitter[j],\
                radius_1[j], k[j], w, e, incl_, period[j], t_zero[j],\
                ldc_1[j*2:j*2+2], ld_law_1,\
                ld_1_func, N_BATMAN, FAC_BATMAN,\
                SBR[j], light_3[j],    t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol,\
                LC, j)

   


    def lc_gpu(time, mag=-99.*np.ones(2), mag_err=-99.*np.ones(2),\
        mag_jitter=-99.*np.ones(10240), mag_zp=-99.*np.ones(10240),\
        radius_1=0.2*np.ones(10240), k=0.2*np.ones(10240),\
        fs=0.0*np.ones(10240), fc=0.0*np.ones(10240), incl=90.*np.ones(10240),\
        period=1.0*np.ones(10240), t_zero=0.0*np.ones(10240),\
        ldc_1=0.8**np.ones(10240*2), ld_law_1 = 0,\
        ld_1_func=-1, N_BATMAN=1000, FAC_BATMAN=5.0e-4,\
        SBR = 0.0*np.ones(10240), light_3 = 0.0*np.ones(10240),\
        t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9,
        N_blocks=256):

        # Allocate the LC array
        if mag_jitter[0]==-99.:
            LC = np.empty(radius_1.shape[0]*time.shape[0], dtype = np.float64) # LC array
        else: 
            LC = np.zeros(radius_1.shape[0], dtype = np.float64)    # loglike array 

        # Now launch the kernel
        # int(np.ceil(radius_1.shape[0]/N_blocks)),N_blocks
        _lc_gpu[int(np.ceil(radius_1.shape[0]/N_blocks)),N_blocks](time, mag, mag_err, mag_zp, mag_jitter, radius_1, k, fs, fc, incl, period, t_zero, ldc_1, ld_law_1, ld_1_func, N_BATMAN, FAC_BATMAN,  SBR, light_3,    t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, LC)
        
        if mag_jitter[0]==-99.:
            LC = LC.reshape((radius_1.shape[0], time.shape[0]))

        
        return LC