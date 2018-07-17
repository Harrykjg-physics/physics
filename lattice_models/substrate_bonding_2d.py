from lattice_models import *
import matplotlib.pyplot as plt
import random


class SubstrateBonding2D:

    def __init__(self, n, m, x, vdW, kT, lattice_type):
        """Simulates self-assembly of borazine molecules on a surface.

        Metropolis Monte Carlo algorithm.

        Args:
            n (int): unit cell size (dimension 0)
            m (int): unit cell size (dimension 1)
            x (int): number of molecules
            vdW: molecular bond strength (molecules bond if they share an edge)
            kT: Temperature
            lattice_type: Lattice object (recommended = HexLattice)

        """
        self.n, self.m = n, m
        self.lattice = lattice_type(n, m)
        self.x = x
        self.vdW = vdW
        self.kT = kT
        self.initialise_system()

    def initialise_system(self):
        """Initialise with self.x (randomly chosen) occupied sites."""
        state0 = np.zeros([self.n, self.m], dtype=int)
        perm = np.random.permutation(self.n * self.m)
        for i in range(self.x):
            a, b = self.lattice.single_to_double_index(perm[i])
            state0[a, b] = 1
        self.state = state0

    def indexfinder(self, arr, site_state):
        """Given an occupation array, returns a list of occupied sites.

        Args:
            arr (ndarray): lattice state
            site_state (int): site state

        Returns (list): indexes of sites in arr satisfying site==site_state

        """
        sites = []
        for i, row in enumerate(arr):
            for j, x in enumerate(row):
                if x == site_state:
                    sites.append((i, j))
        return sites

    def bond_finder(self, site1, site2):
        """Checks whether two sites share an edge"""
        i, j = site1
        bonds = 0
        for neighbour in self.lattice.neighbour_list[i, j, :, :]:
            if neighbour[0] == site2[0] and neighbour[1] == site2[1]:
                bonds += 1
        return bonds

    def new_site(self, site):
        """Checks potential lattice sites until it finds an unoccupied one"""
        perm = np.random.permutation(self.n * self.m)
        newsite = site
        for i in range(self.n * self.m):
            p, q = self.lattice.single_to_double_index(perm[i])
            if self.state[p, q] == 0:
                newsite = (p, q)
                break
        return newsite

    def count_bonds(self, occ_sites):
        """ Counts bonds in a given lattice state (relevant for energy calculation).

        Args:
            occ_sites: list of double-indexed occupied sites

        Returns: number of shared edges (bonds)

        """
        n_bonds = 0
        for i in range(len(occ_sites)):
            for j in range(i + 1, len(occ_sites)):
                n_bonds += self.bond_finder(occ_sites[i], occ_sites[j])
        return n_bonds

    def metropolis_update(self):
        """Single step of Metropolis Monte Carlo algorithm

        Pick a single occupied lattice site: move to an unoccupied site if
        energetically favourable (or due to thermal fluctuation)

        """
        kT, vdW = self.kT, self.vdW

        trial_site = random.sample(self.occupied, 1)[0]  # monomer that we might move
        fixed_sites = []
        for site in self.occupied:
            if site != trial_site:
                fixed_sites.append(site)  # other monomers

        # find a new unoccupied site to move to
        candidate = self.new_site(trial_site)
        trial_occupied = fixed_sites
        trial_occupied.append(candidate)

        E_0 = - vdW * self.bonds
        E_1 = - vdW * self.count_bonds(trial_occupied)
        delta_E = E_1 - E_0

        # flip if energetically favourable, or due to thermal fluctuation
        if np.random.random() < np.exp(-delta_E / kT):
            self.state[trial_site] = 0
            self.state[candidate] = 1
            self.bonds = self.count_bonds(trial_occupied)
            self.occupied = trial_occupied

    def run_sim(self, mc_steps, verbose=True):
        """Run Metropolis MC simulation for specified number of steps

        Args:
            mc_steps (int): number of Monte Carlo steps per sim
            verbose (bool): if True, print simulation parameters

        Returns:
             bonds_ls: list with the number of bonds at each MC step.

        """
        n, m, kT, vdW = self.n, self.m, self.kT, self.vdW

        # Initial bond count (to calculate energy)
        self.occupied = self.indexfinder(self.state, 1)
        self.bonds = self.count_bonds(self.occupied)

        bonds_ls = []
        bonds_ls.append(self.bonds)

        for t in range(mc_steps):
            self.metropolis_update()
            bonds_ls.append(self.bonds)

        if verbose == True:
            coverage = self.x / (n * m)
            print('\nSimulation complete')
            print('n, m, coverage, vdW, kT, mc_steps')
            print(n, m, coverage, vdW, kT, mc_steps)

        return bonds_ls  # number of bonds at each MC step


# Initialise demo

def _get_params():
    """Read from command line (option-value pairs)"""
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--n', '--lattice_size_(0)', type=int,
                        default=10, help='Lattice size (0)')
    parser.add_argument('--m', '--lattice_size_(1)', type=int,
                        default=10, help='Lattice size (1)')
    parser.add_argument('--vdW', '--bond_strength', type=float,
                        default=0.06, help='bond strength (eV)')
    parser.add_argument('--x', '--number_of_molecules', type=int,
                        default=3, help='number of molecules')
    args = parser.parse_args()

    return args.n, args.m, args.vdW, args.x


# Demo

if __name__ == '__main__':
    """Example: simulate, then plot average number of bonds vs temperature.
    Initialise with default values or option-value pairs from command line"""

    n, m, vdW, x = _get_params()
    mc_steps = 10000

    kT_list = np.linspace(0.012, 0.025, 21)
    avg_bonds = []
    for kT in kT_list:
        trimer_model = SubstrateBonding2D(n, m, x, vdW, kT, HexLattice)
        bonds = trimer_model.run_sim(mc_steps)
        avg_bonds.append(np.mean(bonds))

    plt.scatter(kT_list, avg_bonds, marker='x')
    plt.title('Average number of bonds vs Temperature for %d molecules' % x)
    plt.xlabel('Temperature (eV)')
    plt.ylabel('Average number of bonds')
    plt.show()
