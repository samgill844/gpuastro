import numba, numba.cuda
import math 
import numpy as np

from gpuastro.binarystar.utils import *
from gpuastro.binarystar.kepler import *


if numba.cuda.is_available() : __all__ = ['rv', 'rv_gpu']
else : __all__ = ['rv', 'rv_loglike']


@numba.njit(fastmath=True)
def _rv(time, t_zero, period, K1, w, e, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, RV_, N_start):

    for i in range(time.shape[0]):
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

@numba.njit(fastmath=True)
def rv(time, t_zero=0.0, period=1.0, K1=10., fs=0.0, fc=0.0, V0=0.0, dV0=0.0, VS=0.0, VC=0.0, radius_1=0.2, k=0.2, incl=90., ld_law_1 = 0, ldc_1 = np.array([0.8,0.8]), t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9):
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

    @numba.cuda.jit
    def _rv_gpu(time, RV, RV_err, RV_jitter, t_zero, period, K1, fs, fc, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, RV_):
        # get the model index
        j = numba.cuda.grid(1)

        # Conversion
        w = math.atan2(fs[j], fc[j])
        e = fs[j]*fs[j] + fc[j]*fc[j]
        incl_ = math.pi*incl[j]/180.

        # Call kernel
        return _rv(time, RV, RV_err, RV_jitter[j], t_zero[j], period[j], K1[j], w, e, V0[j], dV0[j], VS[j], VC[j], radius_1[j], k[j], incl_, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, RV_, j)
     

    def rv_gpu(time, RV=-99.*np.ones(2), RV_err=-99.*np.ones(2), RV_jitter=-99.*np.ones(2),\
                t_zero=0.0*np.ones(10240), period=1.0*np.ones(10240),\
                K1=10.*np.ones(10240), fs=0.0*np.ones(10240), fc=0.0*np.ones(10240),\
                V0=0.0*np.ones(10240), dV0=0.0*np.ones(10240), VS=0.0*np.ones(10240), VC=0.0*np.ones(10240),\
                radius_1=0.2*np.ones(10240), k=0.2*np.ones(10240), incl=90.*np.ones(10240),\
                ld_law_1 = 0, ldc_1 = 0.8*np.ones(2*10240), t_ecl_tolerance=1e-5, Accurate_t_ecl=False,  Accurate_Eccentric_Anomaly=False, E_tol=1e-9,
                N_blocks=256):

        # Allocate the RV arrays
        if RV_jitter[0]==-99.: RV_ = np.empty(K1.shape[0]* time.shape[0], dtype = np.float64)
        else:               RV_ = np.zeros(K1.shape[0], dtype = np.float64)

        # Launch the kernel
        _rv_gpu[int(np.ceil(K1.shape[0]/N_blocks)),N_blocks](time, RV, RV_err, RV_jitter, t_zero, period, K1, fs, fc, V0, dV0, VS, VC, radius_1, k, incl, ld_law_1, ldc_1, t_ecl_tolerance, Accurate_t_ecl,  Accurate_Eccentric_Anomaly, E_tol, RV_)

        if RV_jitter[0]==-99.: return RV_.reshape((K1.shape[0], time.shape[0]))
        else : return RV_