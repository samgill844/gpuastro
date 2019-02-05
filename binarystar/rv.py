import numba, numba.cuda
import math 
import numpy as np

from gpuastro.binarystar.utils import *
from gpuastro.binarystar.kepler import *


if numba.cuda.is_available() : __all__ = ['rv', 'rv_loglike', 'rv_gpu', 'rv_loglike_gpu']
else : __all__ = ['rv', 'rv_loglike']


@numba.njit(fastmath=True)
def _rv(time, t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, RV_, N_start):

    for i in range(time.shape[0]):
        nu = getTrueAnomaly(time[i], e, w, period,t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol )
        RV_[N_start*time.shape[0] + i] = K1*math.cos(nu + w)              + V0  + dV0*(time[i] - t_zero)

@numba.cuda.jit
def d_rv(time, t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, RV_, N_start):
    # Get the time stamp
    i = numba.cuda.grid(1)

    nu = getTrueAnomaly(time[i], e, w, period,t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol )
    RV_[N_start*time.shape[0] + i] = K1*math.cos(nu + w)              + V0  + dV0*(time[i] - t_zero)



@numba.njit(fastmath=True)
def __rv(time, RV, RV_err, RV_jitter, t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol):
    L=0.0
    for i in range(time.shape[0]):
        nu = getTrueAnomaly(time[i], e, w, period,t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol )
        wt = 1./(RV_err[i]**2 + RV_jitter**2)
        L -= 0.5*(( (K1*math.cos(nu + w)              + V0  + dV0*(time[i] - t_zero)) - RV[i])**2*wt - math.log(wt))             
    return L

@numba.cuda.jit
def d__rv(time, RV, RV_err, RV_jitter, t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, L):
    # Get the time stamp
    i = numba.cuda.grid(1)
    if (i < time.shape[0]):
        nu = getTrueAnomaly(time[i], e, w, period,t_zero, incl, radius_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol )
        wt = 1./(RV_err[i]**2 + RV_jitter**2)
        numba.cuda.atomic.add(L, 0, -0.5*(( (K1*math.cos(nu + w)              + V0  + dV0*(time[i] - t_zero)) - RV[i])**2*wt - math.log(wt))   )          

@numba.njit(fastmath=True)
def rv(time, 
    t_zero=0.0, period=1.0, 
    K1=10., fs=0.0, fc=0.0, 
    V0=0.0, dV0=0.0, VS=0.0, VC=0.0, 
    radius_1=0.2, k=0.2, incl=90., ld_law_1 = 0, ldc_1 = np.array([0.8,0.8]), 
    t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9):

    # Unpack args
    w = math.atan2(fs, fc)
    e = fs*fs + fc*fc
    incl = math.pi*incl/180.

    # Allocate the RV arrays
    RV_ = np.empty(time.shape[0], dtype = np.float64)

    # Call the kernel
    _rv(time, t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, RV_, 0)
    
    return RV_


@numba.njit(fastmath=True)
def rv_loglike(time, RV=-99.*np.ones(2), RV_err=-99.*np.ones(2), RV_jitter=-99., t_zero=0.0, period=1.0, K1=10., fs=0.0, fc=0.0, V0=0.0, dV0=0.0, VS=0.0, VC=0.0, radius_1=0.2, k=0.2, incl=90., ld_law_1 = 0, ldc_1 = np.array([0.8,0.8]), t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9):
    # Unpack args
    w = math.atan2(fs, fc)
    e = fs*fs + fc*fc
    incl = math.pi*incl/180.

    # Call the kernel
    return __rv(time,RV, RV_err, RV_jitter,  t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol)

if numba.cuda.is_available():
    def rv_gpu(time, 
        t_zero=0.0, period=1.0, 
        K1=10., fs=0.0, fc=0.0, 
        V0=0.0, dV0=0.0, VS=0.0, VC=0.0, 
        radius_1=0.2, k=0.2, incl=90., ld_law_1 = 0, ldc_1 = np.array([0.8,0.8]), 
        t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9,
        threads_per_block=256):

        # Unpack args
        w = math.atan2(fs, fc)
        e = fs*fs + fc*fc
        incl = math.pi*incl/180.

        # Allocate the RV arrays
        RV_ = np.empty(time.shape[0], dtype = np.float64)

        # Calculate the blocks required
        blocks = int(np.ceil(time.shape[0] / threads_per_block))
        
        # Call the kernel
        d_rv[blocks, threads_per_block ](time, t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, RV_, 0)
        
        return RV_ 


    def rv_loglike_gpu(time, 
                        RV=-99.*np.ones(2), RV_err=-99.*np.ones(2), RV_jitter=-99., 
                        t_zero=0.0, period=1.0, K1=10., 
                        fs=0.0, fc=0.0, 
                        V0=0.0, dV0=0.0, VS=0.0, VC=0.0, 
                        radius_1=0.2, k=0.2, incl=90., ld_law_1 = 0, ldc_1 = np.array([0.8,0.8]), 
                        t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9,
                        threads_per_block=256):

        # Unpack args
        w = math.atan2(fs, fc)
        e = fs*fs + fc*fc
        incl = math.pi*incl/180.

        # Create a loglike array
        L = np.zeros(1, dtype = np.float64)

        # Calculate the blocks required
        blocks = int(np.ceil(time.shape[0] / threads_per_block))
        

        # Call the kernel
        d__rv[blocks, threads_per_block ](time,RV, RV_err, RV_jitter,  t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, L)

        return L[0]