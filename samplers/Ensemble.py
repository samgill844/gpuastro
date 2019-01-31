import numba, numba.cuda 
import numpy as np 
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import random, math

import matplotlib.pyplot as plt



@numba.njit(parallel=True, fastmath=True)
def Parallel_Sampler_CPU_parallel(lnprob, args, cpu_loglikliehoods, cpu_positions, a=2.0):
    
    # First we need to evaluate the starting position
    for i in range(cpu_positions.shape[1]) : cpu_loglikliehoods[0, i] = lnprob(cpu_positions[0, i], args)
    
    # Iterate over the steps
    for i in range(1, cpu_positions.shape[0]):

        # Iterate over the walkers
        for j in numba.prange(cpu_positions.shape[1]):
            # Using the method of kwee and woden
            # We are going to use a walker in the other ensemble to guess a trial position
            # (https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-s.pdf)
            if (j > cpu_positions.shape[1]/2) : index = int(random.uniform(0,cpu_positions.shape[1]/2))
            else                              : index = int(random.uniform(cpu_positions.shape[1]/2, cpu_positions.shape[1]))

            # Iterate over the dimensions to create trial positions
            for k in range(cpu_positions.shape[2]):
                Z = ((a - 1.) * random.random() + 1) ** 2. / a
                #cpu_positions[i, j, k] = cpu_positions[i-1, j, k]  - Z*(cpu_positions[i-1, j, k] - cpu_positions[i-1, index, k])
                cpu_positions[i, j, k] = cpu_positions[i-1, index, k]  - Z*(cpu_positions[i-1, index, k] - cpu_positions[i-1, j, k])
                
            # Evaluate the trial solution
            cpu_loglikliehoods[i, j] = lnprob(cpu_positions[i, j], args)

            # assess trial positions
            if (cpu_loglikliehoods[i, j] < cpu_loglikliehoods[i-1, j]):
                if (random.uniform(0,1) > math.exp(cpu_loglikliehoods[i, j] - cpu_loglikliehoods[i-1, j])):
                    # Here we've got unlucky, so revert
                    cpu_loglikliehoods[i, j] =  cpu_loglikliehoods[i-1, j]
                    for k in range(cpu_positions.shape[2]) : cpu_positions[i, j, k] = cpu_positions[i-1, j, k]

@numba.njit(parallel=False, fastmath=True)
def Parallel_Sampler_CPU(lnprob, args, cpu_loglikliehoods, cpu_positions, a=2.0):
    
    # First we need to evaluate the starting position
    for i in range(cpu_positions.shape[1]) : cpu_loglikliehoods[0, i] = lnprob(cpu_positions[0, i], args)
    
    # Iterate over the steps
    for i in range(1, cpu_positions.shape[0]):

        # Iterate over the walkers
        for j in numba.prange(cpu_positions.shape[1]):
            # Using the method of kwee and woden
            # We are going to use a walker in the other ensemble to guess a trial position
            # (https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-s.pdf)
            if (j > cpu_positions.shape[1]/2) : index = int(random.uniform(0,cpu_positions.shape[1]/2))
            else                              : index = int(random.uniform(cpu_positions.shape[1]/2, cpu_positions.shape[1]))

            # Iterate over the dimensions to create trial positions
            for k in range(cpu_positions.shape[2]):
                Z = ((a - 1.) * random.random() + 1) ** 2. / a
                #cpu_positions[i, j, k] = cpu_positions[i-1, j, k]  - Z*(cpu_positions[i-1, j, k] - cpu_positions[i-1, index, k])
                cpu_positions[i, j, k] = cpu_positions[i-1, index, k]  - Z*(cpu_positions[i-1, index, k] - cpu_positions[i-1, j, k])

            # Evaluate the trial solution
            cpu_loglikliehoods[i, j] = lnprob(cpu_positions[i, j], args)

            # assess trial positions
            if (cpu_loglikliehoods[i, j] < cpu_loglikliehoods[i-1, j]):
                if (random.uniform(0,1) > math.exp(cpu_loglikliehoods[i, j] - cpu_loglikliehoods[i-1, j])):
                    # Here we've got unlucky, so revert
                    cpu_loglikliehoods[i, j] =  cpu_loglikliehoods[i-1, j]
                    for k in range(cpu_positions.shape[2]) : cpu_positions[i, j, k] = cpu_positions[i-1, j, k]


'''
@numba.njit
def lnprob(theta, args) :
    wt = 1./0.001
    return -0.5 * ( (args[0] - theta[0])**2*wt - math.log(wt)  + (args[1] - theta[1])**2*wt - math.log(wt) )

# Define the parameters
theta_names = ['par 1', 'par 2']
theta = np.array([-0.2,-0.1], dtype = np.float32)
nsteps_cpu = 2000000
ndim_cpu = len(theta)
nwalkers_cpu = 4*ndim_cpu

# Initialise the loglikeliehoods
cpu_loglikliehoods = np.empty((nsteps_cpu, nwalkers_cpu), dtype = np.float32)
cpu_positions = np.empty((nsteps_cpu, nwalkers_cpu, ndim_cpu), dtype = np.float32)
for i in range(nwalkers_cpu) : cpu_positions[0, i, :] = np.random.normal(theta, 1e-3) # start them in Guassian ball

args = np.array([-0.2,0.1])
Parallel_Sampler_CPU(lnprob, args, cpu_loglikliehoods, cpu_positions, a=2.0)


N_cpu_steps_per_second = 50

f, axs = plt.subplots(nrows=3, ncols=1, figsize = (3.54331,8))
ax1, ax2, ax3= axs
for i in range(nwalkers_cpu):
    ax1.semilogx(range(nsteps_cpu), cpu_positions[:,i,0], 'k', alpha = 0.05)
    ax2.semilogx(range(nsteps_cpu), cpu_positions[:,i,1], 'k', alpha = 0.05)

    ax1.set_ylabel(theta_names[0])
    ax2.set_ylabel(theta_names[1])

    ax3.loglog(range(nsteps_cpu), cpu_loglikliehoods[:,i], 'k', alpha = 0.07)


ax3.set_ylabel(r'$\mathcal{L}$')
ax3.set_xlabel('Step')

f.subplots_adjust(left=0.3, wspace=0.6)

f.align_ylabels(axs[:])

plt.show()#
'''
