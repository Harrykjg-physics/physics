import numpy as np
import matplotlib.pyplot as plt

class BrownianModel2D:

    def __init__(self, particle_mass, kT, gamma, npart, dt, timesteps):
        self.particle_mass = particle_mass
        self.kT = kT
        self.gamma = gamma
        self.npart = npart
        self.dt = dt
        self.timesteps = timesteps

        self.noisescale = (((2.0 * self.gamma * self.kT * self.dt) ** (0.5)) / self.particle_mass)
        self.timefactor = (self.gamma * self.dt) / self.particle_mass

    def initialise(self):
        '''Model state represented as a numpy array, with
        rows representing different particles
        columns = [x, y, vx, vy] - all initialised at zero'''

        self.state = np.zeros((self.npart, 4))

    def update_timestep(self):
        '''Velocity-Verlet integrator, vectorised for efficiency'''

        x = self.state[:, 0]  # vector: x positions for all particles
        y = self.state[:, 1]  # vector: y positions for all particles
        vx = self.state[:, 2]  # vector: x velocities for all particles
        vy = self.state[:, 3]  # vector: y velocities for all particles

        # v(t+dt) = v(t) - drag(v(t)) + noise
        new_vx = vx - (vx * (self.timefactor)) + (np.random.randn(self.npart) * self.noisescale)
        self.state[:, 2] = new_vx
        new_vy = vy - (vy * (self.timefactor)) + (np.random.randn(self.npart) * self.noisescale)
        self.state[:, 3] = new_vy

        # x(t+dt) = x(t) + v(t+dt)*dt
        # With no position dependent forces, V.V. algorithm is much simpler
        self.state[:, 0] = x + (new_vx * self.dt)  # update x positions
        self.state[:, 1] = y + (new_vy * self.dt)  # update y positions

    def update_MSD(self):
        '''Calculate mean squared displacement (averaged over particles) from state'''

        x = self.state[:, 0]  # vector: x positions for all particles
        y = self.state[:, 1]  # vector: y positions for all particles

        MSD = np.sum((x ** 2) + (y ** 2)) / self.npart

        return MSD

    def run_simulation(self, plot_trajectories=0):

        self.time = np.arange(self.timesteps) * self.dt
        self.MSD = np.zeros(self.timesteps)

        if plot_trajectories > 0:
            self.trajectories = np.zeros((self.timesteps, plot_trajectories*2))

        for t in range(self.timesteps):

            self.update_timestep()
            self.MSD[t] = self.update_MSD()

            if plot_trajectories > 0:
                for p in range(plot_trajectories):
                    self.trajectories[t, 2 * p] = self.state[p, 0]
                    self.trajectories[t, (2 * p) + 1] = self.state[p, 1]

            if t % (self.timesteps/10) == 0:
                print 'running...', 100 * t / self.timesteps, '%'

        print 'Simulation complete\n'

        if plot_trajectories > 0:
            plt.rcParams["figure.figsize"] = [9, 9]
            for i in range(plot_trajectories):
                plt.plot(self.trajectories[:, (2 * i)],
                         self.trajectories[:, (2 * i) + 1],
                         lw=0.3)

            plt.axis('equal')
            plt.title('Trajectory for Brownian particle(s)')
            plt.xlabel('Displacement (m)')
            plt.ylabel('Displacement (m)')
            plt.savefig('brownian_trajectory.png')
            plt.show()

        return self.MSD, self.time

