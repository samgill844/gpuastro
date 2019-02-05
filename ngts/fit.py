################################
#           NGTS fit           #
################################
import numpy as np 
import numba, numba.cuda 
import math 
from gpuastro.binarystar import lc_loglike, lc, lc_loglike_gpu
from gpuastro.utilities import htoc1_, htoc2__
from gpuastro.samplers import Ensemble_sampler_numba
import matplotlib.pyplot as plt 
import corner

@numba.njit
def lnprior(theta, args):
    # theta
    # 0 : t_zero
    # 1 : period
    # 2 : radius_1
    # 3 : k
    # 4 : b
    # 5 : h1
    # 6 : h2
    # 7 : zp
    # 8 : J
    #
    # args
    # 0 : time
    # 1 : mag
    # 2 : mag_er
    # 3 : t_zero_ref
    # 4 : period_ref

    if (theta[0] < args[3,0] - args[4,0]*0.1) or (theta[0] > args[3,0] + args[4,0]*0.1) : return -math.inf
    if (theta[1] < args[4,0] - 0.05) or (theta[1] > args[4,0] + 0.05) : return -math.inf
    if (theta[2] < 0.) or (theta[2] > 0.5) : return -math.inf
    if (theta[3] < 0.) or (theta[3] > 0.5) : return -math.inf
    if (theta[4] < 0.) or (theta[4] > 1) : return -math.inf
    if (theta[5] < 0.6) or (theta[5] > 0.7) : return -math.inf
    if (theta[6] < 0.4) or (theta[6] > 0.6) : return -math.inf
    if (theta[8] < 0) : return -math.inf
    return 0.


@numba.njit
def lnlike(theta, args):

    ldc_1_1 = htoc1_(theta[5], theta[6])
    ldc_1_2 = htoc2__(ldc_1_1, theta[6])

    incl = 180*math.acos(theta[2]*theta[4])/math.pi

    return   lc_loglike(time=args[0], mag=args[1], mag_err=args[2], mag_jitter = theta[7],mag_zp=theta[7],
                      t_zero = theta[0], period=theta[1], radius_1=theta[2] , k=theta[3], incl=incl, 
                      ldc_1_1=ldc_1_1, ldc_1_2=ldc_1_1 )


@numba.njit
def lnprob(theta, args):
    # Get prior
    prior = lnprior(theta, args)
    if prior==-math.inf : return prior

    # return the lnlike
    return  lnlike(theta, args)


def get_phase_model(theta):
    # theta
    # 0 : t_zero
    # 1 : period
    # 2 : radius_1
    # 3 : k
    # 4 : b
    # 5 : h1
    # 6 : h2
    # 7 : zp
    # 8 : J
    phase = np.linspace(-0.1,0.1, 10000)
    incl = 180*math.acos(theta[2]*theta[4])/math.pi
    ldc_1_1 = htoc1_(theta[5], theta[6])
    ldc_1_2 = htoc2__(ldc_1_1, theta[6])
    return phase, theta[7] - 2.5*np.log10(lc(phase, t_zero=0., period=1.0, radius_1 = theta[2], k=theta[3], incl=incl, ldc_1_1=ldc_1_1, ldc_1_2 = ldc_1_2))

def phaser(time, t_zero, period, offset=0.5):
    return ((time - t_zero + offset*period) / period)  - np.floor((time - t_zero + offset*period) / period) -offset

def fit_ngts_lighcurve(filename, T0, P, radius_1=0.2, k = 0.2, zp = 0.0, nsteps = 1000,phase_cut = 0.1, NOI='test', burn_in = -1, walkermult=4):
    # Open the filename
    hjd,  mag,  mag_err,  rel_flux, rel_flux_err = np.loadtxt(filename).T
    phase = phaser(hjd, T0, P, offset=0.5)
    mask = (phase > -phase_cut) & (phase < phase_cut)
    hjd,  mag,  mag_err,  rel_flux, rel_flux_err = np.loadtxt(filename)[mask].T

    args = np.array([hjd.tolist(),  mag.tolist(),  mag_err.tolist(), (T0*np.ones(hjd.shape[0])).tolist(), (P*np.ones(hjd.shape[0])).tolist() ]).astype(np.float64)

    theta = np.array([T0, P, radius_1, k,0.1,0.65,0.45,zp,0.01])
    print('Initial loglike : ', lnprob(theta, args))

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.scatter(phaser(hjd, T0, P, offset=0.5), mag, s=10, c='k', alpha = 0.05)

    phase, model = get_phase_model(theta)
    ax.plot(phase, model,'r')
    ax.invert_yaxis()
    ax.set_xlim(-phase_cut,phase_cut)
    ax.set_ylabel('Mag')
    ax.set_xlabel('Phase')
    fig.savefig(NOI+'_initial.png')
    plt.show()
    plt.close()

    ndim = len(theta)
    nwalkers = walkermult*ndim 
    p0 = np.array([np.random.normal(theta,1e-6) for i in range(nwalkers)])

    positions, loglike = Ensemble_sampler_numba(lnprob, args, p0, 1, a=2.0, target='cpu')
    positions, loglike = Ensemble_sampler_numba(lnprob, args, p0, nsteps, a=2.0, target='cpu')

    best_idx = np.argmax(loglike.flatten())
    best_theta = positions.reshape((4*len(theta)*nsteps, len(theta)))[best_idx]



    fig, ax = plt.subplots(figsize=(20, 5))
    ax.scatter(phaser(hjd, best_theta[0], best_theta[1], offset=0.5), mag, s=10, c='k', alpha = 0.05)

    phase, model = get_phase_model(best_theta)
    ax.plot(phase, model,'r')
    ax.invert_yaxis()
    ax.set_xlim(-phase_cut,phase_cut)
    ax.set_ylabel('Mag')
    ax.set_xlabel('Phase')
    fig.savefig(NOI+'_final.png')

    # theta
    # 0 : t_zero
    # 1 : period
    # 2 : radius_1
    # 3 : k
    # 4 : b
    # 5 : h1
    # 6 : h2
    # 7 : zp
    # 8 : J
    if burn_in==-1 : burn_in = int(nsteps/2)
    theta_names = ['T$_0$', 'P [d]', r'R$_*$/a', r'R$_2$/R$_*$', 'b', 'h$_1$', 'h$_2$', 'zp', r'$\sigma_J$']
    positions_corner = positions[burn_in:,:,:]
    positions_corner = positions_corner.reshape(positions_corner.shape[0]*positions_corner.shape[1], positions_corner.shape[2])
    fig2 = corner.corner(positions_corner, labels = theta_names)
    fig.savefig(NOI+'_corner.png')    

    plt.show()






    return positions, loglike