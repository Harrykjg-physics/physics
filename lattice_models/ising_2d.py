from lattice_models import *
import matplotlib.pyplot as plt


class Ising2D:

    def __init__(self, n, m, J, kT, lattice_type):
        """Class for 2D Ising Model

        Initialisation is easiest if either kT or J is set = 1 and the other
        parameter varied, since the ratio kT:J determines system behaviour.

        Args:
            n (int): linear size (y)
            m (int): linear size (x)
            J: Coupling strength
            kT: Temperature
            lattice_type: Lattice object (recommended = SquareLattice)

        """
        self.n, self.m = n, m
        self.lattice = lattice_type(n, m)
        self.J, self.kT = J, kT
        self.initialise_system()

    def initialise_system(self):
        """Randomly initialise a system of spins (n x m array).

        Each has independent and equal probability of being up (+1) or down (-1)

        """
        initial = np.random.rand(self.n, self.m)
        boolean = initial > 0.5
        self.state = 2 * boolean - 1

    def metropolis_sim(self, n_steps):
        """Monte Carlo simulation using Metropolis algorithm

        Args:
            n_steps (int): number of Metropolis steps per simulation

        """
        for i in range(n_steps):
            self.update_single_spin()

    def update_single_spin(self):
        """ Single step of Metropolis Monte Carlo algorithm.

        Metropolis algorithm: Pick a single lattice site at random and flip
        if energetically favourable (or due to thermal fluctuation).

        """
        site = np.random.randint(self.n * self.m)
        i, j = self.lattice.single_to_double_index(site)
        spin = self.state[i, j]

        E_0 = 0
        for nind in range(self.lattice.n_neighbours):
            p, q = self.lattice.neighbour_list[i, j, nind, :]
            E_0 -= self.state[p, q] * spin * self.J
        delta_E = -2 * (E_0)

        # Flip if energetically favourable or due to thermal fluctuation
        if np.random.random() < np.exp(-delta_E / self.kT):
            self.state[i, j] = - spin

    def flip_wolff_cluster(self, i, j):
        """Given a single lattice site, build a Wolff cluster and flip it."""

        J, kT = self.J, self.kT
        cluster_spin = self.state[i, j]
        self.state[i, j] = -cluster_spin  # flip initial spin
        additions = [(i, j)]
        finished_checking = False

        while finished_checking == False:
            to_consider = additions  # Only consider potential bonds once
            additions = []  # clear at start of each round
            n_additions = 0

            # loop over cluster members not yet checked
            for cluster_member in to_consider:
                a, b = cluster_member
                # loop over neighbours
                for nind in range(4):
                    p, q = self.lattice.neighbour_list[a, b, nind, :]
                    # Decide whether neighbouring spin joins cluster
                    if ((self.state[p, q] == cluster_spin) &
                            (np.random.random() < (1 - np.exp(- (2 * J) / kT)))):
                        # flip immediately to avoid considering again
                        self.state[p, q] = - self.state[p, q]
                        additions.append((p, q))
                        n_additions += 1

            if n_additions == 0:
                finished_checking = True

    def wolff_cluster_update(self):
        """Pick a lattice site at random, construct a Wolff cluster around it,
        then flip all spins in cluster

        """
        site = np.random.randint(self.n * self.m)
        i, j = self.lattice.single_to_double_index(site)
        self.flip_wolff_cluster(i, j)

    def display_state(self, title):
        """Plot lattice as monochrome heatmap using matplotlib.pyplot

        """
        plt.imshow(self.state, cmap='hot', interpolation='nearest')
        plt.title(title)
        plt.show()

    def magnetisation_per_spin(self):
        """Calculate average magnetisation per spin, in units of the maximum.

        Returns:
            average magnetisation per spin (current state)

        """
        return np.mean(self.state)


# Initialise demo

def _get_params():
    """Read from command line (option-value pairs)"""
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n', '--lattice_size_(0)', type=int,
        default=100, help='lattice size (0)')
    parser.add_argument(
        '--m', '--lattice_size_(1)', type=int,
        default=100, help='lattice size (1)')
    parser.add_argument(
        '--J', '--coupling_strength', type=float,
        default=1.0, help='coupling_strength')
    parser.add_argument(
        '--kT', '--temperature', type=float,
        default=1.0, help='temperature (same units as J)')

    args = parser.parse_args()

    return args.n, args.m, args.J, args.kT


# Demo animation

if __name__ == '__main__':
    """Example: animated Ising2D model using metropolis algorithm"""
    from matplotlib import animation

    n, m, J, kT = _get_params()
    anim_interval = 20
    mc_steps_per_frame = 500

    ising = Ising2D(n, m, J, kT, SquareLattice)
    ising.initialise_system()

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xticks([])
    ax.set_yticks([])
    im = plt.imshow(ising.state, cmap='hot_r', animated=True)
    t = '2D Ising Model (kT, J = %d, %d); Metropolis algorithm' % (kT, J)
    fig.suptitle(t)

    steps = 0

    def update(*args):
        """Multiple metropolis steps per animation frame"""
        ising.metropolis_sim(mc_steps_per_frame)
        global steps
        steps += mc_steps_per_frame
        im.set_data(ising.state)
        fig.suptitle(t + '\n%d MC steps' % steps)
        return im,


    ani = animation.FuncAnimation(
        fig, update, interval=anim_interval, blit=False)
    plt.show()

