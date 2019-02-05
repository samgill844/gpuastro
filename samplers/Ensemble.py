import numba, numba.cuda 
import numpy as np 
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import random, math

import matplotlib.pyplot as plt

__all__ = ['Ensemble_sampler_numba']

@numba.njit(parallel=True, fastmath=True)
def _Parallel_Sampler_CPU_parallel(lnprob, args, cpu_loglikliehoods, cpu_positions, a=2.0):
    
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
def _Parallel_Sampler_CPU(lnprob, args, cpu_loglikliehoods, cpu_positions, a=2.0):
    
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

def Ensemble_sampler_numba(lnprob, args, p0, nsteps, a=2.0, target='cpu', threads_per_block=256):
    # Get the number of walkers
    nwalkers = p0.shape[0]
    ndim = p0.shape[1]

    # Allocate the positions
    positions = np.empty((nsteps, nwalkers, ndim), dtype = np.float32)
    positions[0] = p0

    # Allocate the loglike array
    loglike = np.empty((nsteps, nwalkers), dtype = np.float32)


    if target=='cpu'    : _Parallel_Sampler_CPU(lnprob, args, loglike, positions, a)
    elif target=='cpup' : _Parallel_Sampler_CPU_parallel(lnprob, args, loglike, positions, a)
    elif target=='cuda' : 
        if not numba.cuda.is_available():
            print('CUDA support not available.')
        else : 
            # Calculate blocks required
            blocks = int(np.ceil(nwalkers / threads_per_block))

            # Create the RNG states
            rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)

            # Launch
            _Parallel_Sampler_GPU[blocks, threads_per_block](lnprob, args, loglike, positions, rng_states,threads_per_block,  a)
    else :
        print('I cant understand your choice.')
        print('Please chose from:')
        print('\tcpu - CPU (1 thread)')
        print('\tcpup - CPU (max threads)')
        print('\tcuda - GPU support')

    return positions, loglike



####################
# GPU sampler
##################
if numba.cuda.is_available():
    @numba.cuda.jit
    def _Parallel_Sampler_GPU(lnprob, args, gpu_loglikliehoods, gpu_positions, rng_states, threads_per_block,  a=2.0):
        # Get the thread ID (walker)
        j = numba.cuda.grid(1)

        # Define which grid the thread belongs to
        lower = int(j / threads_per_block)
        
        # First we need to evaluate the starting position for the walker
        gpu_loglikliehoods[0, j] = lnprob(gpu_positions[0, j,:], args)
        
        # Iterate over the steps
        for i in range(1, gpu_positions.shape[0]):

            # Using the method of kwee and woden
            # We are going to use a walker in the other ensemble to guess a trial position
            # (https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-s.pdf)
            # The problem is, we can only synchronise the threads of in each block,
            # so we can only select a walker from the other half of walkers in each block.
            if ((j - lower*threads_per_block) >threads_per_block/2) : index = int(lower*threads_per_block)  + int(threads_per_block*xoroshiro128p_uniform_float32(rng_states, j)/2)
            else                                                    : index = int(lower*threads_per_block)  + int(threads_per_block/2) + int(threads_per_block*xoroshiro128p_uniform_float32(rng_states, j)/2)

            # Now synchronise the walkers in each block
            numba.cuda.syncthreads()
            
            # Iterate over the dimensions to create trial positions
            for k in range(gpu_positions.shape[2]):
                Z = ((a - 1.) * xoroshiro128p_uniform_float32(rng_states, j) + 1) ** 2. / a
                #zz = ((self.a - 1.) * random.rand(Ns) + 1) ** 2. / self.a
                #if j==1 : print(Z)
                #gpu_positions[i, j, k] = gpu_positions[i-1, j, k]  - Z*(gpu_positions[i-1, j, k] - gpu_positions[i-1, index, k])
                gpu_positions[i, j, k] = gpu_positions[i-1, index, k]  - Z*(gpu_positions[i-1, index, k] - gpu_positions[i-1, j, k])

            # Evaluate the trial solution
            gpu_loglikliehoods[i, j] = lnprob(gpu_positions[i, j], args)
            #if j==0 : print(cpu_loglikliehoods[i, j] < cpu_loglikliehoods[i-1, j])

            # assess trial positions
            if (gpu_loglikliehoods[i, j] < gpu_loglikliehoods[i-1, j]):
                if (xoroshiro128p_uniform_float32(rng_states, j) > math.exp(gpu_loglikliehoods[i, j] - gpu_loglikliehoods[i-1, j])):
                    # Here we've got unlucky, so revert
                    gpu_loglikliehoods[i, j] =  gpu_loglikliehoods[i-1, j]
                    for k in range(gpu_positions.shape[2]) : gpu_positions[i, j, k] = gpu_positions[i-1, j, k]



'''
@numba.njit
def lnprob(theta, args) :
    wt = 1./0.001
    return -0.5 * ( (args[0] - theta[0])**2*wt - math.log(wt)  + (args[1] - theta[1])**2*wt - math.log(wt) )

# Define the parameters
theta_names = ['par 1', 'par 2']
theta = np.array([-0.2,-0.1], dtype = np.float32)

target = 'cpu'
nsteps = 200
ndim = len(theta)
nwalkers = 4*ndim

# Initialise the loglikeliehoods
p0 = np.empty((nwalkers, ndim), dtype = np.float32)
for i in range(nwalkers) : p0[i, :] = np.random.normal(theta, 1e-3) # start them in Guassian ball

args = np.array([-0.2,0.1])
positions, loglike = Ensemble_sampler_numba(lnprob, args, p0, nsteps, a=2.0, target=target, threads_per_block=256)

N_cpu_steps_per_second = 50

f, axs = plt.subplots(nrows=3, ncols=1, figsize = (3.54331,8))
ax1, ax2, ax3= axs
for i in range(nwalkers):
    ax1.semilogx(range(nsteps), positions[:,i,0], 'k', alpha = 0.05)
    ax2.semilogx(range(nsteps), positions[:,i,1], 'k', alpha = 0.05)

    ax1.set_ylabel(theta_names[0])
    ax2.set_ylabel(theta_names[1])

    ax3.loglog(range(nsteps), loglike[:,i], 'k', alpha = 0.07)


ax3.set_ylabel(r'$\mathcal{L}$')
ax3.set_xlabel('Step')

f.subplots_adjust(left=0.3, wspace=0.6)

f.align_ylabels(axs[:])

plt.show()#
'''
