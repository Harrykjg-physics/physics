import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from brownian_model import *

# Parameters

kB = 1.381 * (10**-23)  # Botlzmann constant in J/K

T = 300                 # Temperature of system in K
kT = kB * T
M = 1.5*(10**-10)       # suspension particle mass in kg
gamma = 8*(10**-7)      # fluid drag factor in Ns/m

npart = 1000            # number of particles
dt = 0.00002            # simulation timestep in seconds
timesteps = 50000       # total number of time steps

# MAIN PROGRAM

brownian = BrownianModel2D(M, kT, gamma, npart, dt, timesteps)

print '2D Brownian Motion simulation'
print '(%d timesteps = %f seconds)' % (timesteps, (dt*timesteps))
print 'Model Parameters:'
print 'kT =', kT, 'J \nParticle mass =', M, 'kg'
print 'Drag factor (gamma) =', gamma, 'Ns/m\n'

brownian.initialise()

MSD, time = brownian.run_simulation(plot_trajectories=3)

gradient, _, _, _, _ = stats.linregress(
    time, MSD)
Dexp = gradient / 4     # based on prediction <MSD> = 2 * dimensions * D * t
difference = abs(100 * (1 - (Dexp / (kT / gamma))))
Dpred = kT / gamma

print 'Predicted value of diffusion constant D = kT/gamma =', Dpred, 'm**2/s\n'
print 'Measured/simulation value of D =', Dexp, 'm**2/s'
print 'This is within', difference, '% of the predicted value'

