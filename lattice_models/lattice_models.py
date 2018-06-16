import numpy as np
import random
import matplotlib.pyplot as plt


class Lattice:
    """ This class provides an interface. A subclass should contain
    a neighbours(self, i, j) function defining the neighbouring
    point (their indices) to point (i, j). This defines the lattice
    topology.

    The subclass constructor should also define the number of neighbours
    at each point"""

    def __init__(self, n, m):
        """
        Args:
            n (int): linear size in y dimension
            m (int): linear size in x dimension
        """
        self.n = n
        self.m = m
        self.generate_neighbour_list()

    def single_to_double_index(self, k):
        """
        Args:
            k (int): single index for array element

        Returns:
            i, j: corresponding double index for array element
        """
        return (k // self.m), np.mod(k, self.m)

    def double_to_single_index(self, i, j):
        return (i * self.n) + j

    def neighbours(self, i, j):
        """Must be implemented in the subclass used.
        See examples in subclasses HexLattice and SquareLattice"""
        raise NotImplementedError(
            'nearest neighbours function missing in class %s'
            % self.__class__.__name__)

    def generate_neighbour_list(self):
        """Generate an array of neighbours:
        neighbours[i, j, p, :] = a, b, where
        a, b = double indexed coordinates of pth neighbour of site (i, j)"""
        n = self.n
        m = self.m
        neighbour_list = np.zeros(
            [n, m, self.n_neighbours, 2], dtype=np.int)

        for s in range(n * m):
            i, j = self.single_to_double_index(s)
            nn = self.neighbours(i, j)
            neighbour_list[i, j, :, :] = nn
        self.neighbour_list = neighbour_list


class HexLattice(Lattice):

    def __init__(self, n, m):
        """
        Args:
            n (int): linear size in y dimension
            m (int): linear size in y dimension
        """
        self.n_neighbours = 6
        super().__init__(n, m)

    def neighbours(self, i, j):
        """Nearest neighbouring sites; this function defines a hexagonal lattice:
                X X
              X 0 X
              X X
        returns an array with entries corresponding to the six nearest neighbours
        of the given lattice site"""
        n = self.n
        m = self.m
        nn = np.zeros([6, 2], dtype=np.int)

        # order: top, top right, left, right, bottom left
        nn[0, :] = np.mod(i + 1, n), j
        nn[1, :] = np.mod(i + 1, n), np.mod(j + 1, m)
        nn[2, :] = i, np.mod(j - 1 + m, m)
        nn[3, :] = i, np.mod(j + 1, m)
        nn[4, :] = np.mod(i - 1 + n, n), np.mod(j - 1 + m, m)
        nn[5, :] = np.mod(i - 1 + n, n), j

        return nn


class SquareLattice(Lattice):
    """
    Von Neumann Neighbourhood
    """

    def __init__(self, n, m):
        """
        Args:
            n (int): linear size in y dimension
            m (int): linear size in x dimension
        """
        self.n_neighbours = 4
        super().__init__(n, m)

    def neighbours(self, i, j):
        """Nearest neighbouring sites (von Neumnann neighbourhood);
        this function defines a square lattice:
                X
              X 0 X
                X
        returns an array with entries corresponding to the four nearest
        neighbours of the given lattice site"""
        n = self.n
        m = self.m
        nn = np.zeros([4, 2], dtype=np.int)

        # order: top, left, right, bottom
        nn[0, :] = np.mod(i + 1, n), j
        nn[1, :] = i, np.mod(j - 1 + m, m)
        nn[2, :] = i, np.mod(j + 1, m)
        nn[3, :] = np.mod(i - 1 + n, n), j

        return nn


class SquareLattice2(Lattice):
    """Moore neighbourhood"""

    def __init__(self, n, m):
        """
        Args:
            n (int): linear size in y dimension
            m (int): linear size in x dimension
        """
        self.n_neighbours = 8
        super().__init__(n, m)

    def neighbours(self, i, j):
        """Nearest neighbouring sites (Moore neighbourhood);
        this function defines a square lattice:
              X X X
              X 0 X
              X X X
        returns an array with entries corresponding to the eight nearest
        neighbours of the given lattice site.
        """
        n = self.n
        m = self.m
        nn = np.zeros([8, 2], dtype=np.int)

        # order: top, left, right, bottom
        nn[0, :] = np.mod(i + 1, n), np.mod(j - 1 + m, m)
        nn[1, :] = np.mod(i + 1, n), j
        nn[2, :] = np.mod(i + 1, n), np.mod(j + 1, m)
        nn[3, :] = i, np.mod(j - 1 + m, m)
        nn[4, :] = i, np.mod(j + 1, m)
        nn[5, :] = np.mod(i - 1 + n, n), np.mod(j - 1 + m, m)
        nn[6, :] = np.mod(i - 1 + n, n), j
        nn[7, :] = np.mod(i - 1 + n, n), np.mod(j + 1, m)

        return nn


class SubstrateBonding2D:
    """Simulates van der Waals bonding (self assembly) of borazine molecules on a surface (at thermal equilibrium)
     via Metropolis Monte Carlo algorithm."""

    def __init__(self, n, m, x, vdW, kT, lattice_type):
        """
        Args:
            n (int):
            m (int):
            x (int): number of molecules
            vdW: molecular bond strength (molecules are bonded if they share an edge)
            kT: Temperature
            lattice_type: Lattice object (HexLattice or SquareLattice: recommended = HexLattice)
        """
        self.n, self.m = n, m
        self.lattice = lattice_type(n, m)
        self.x = x
        self.vdW = vdW
        self.kT = kT
        self.initialise_system()

    def initialise_system(self):
        """Initialise with self.x (randomly chosen) occupied sites"""
        state0 = np.zeros([self.n, self.m], dtype=int)
        perm = np.random.permutation(self.n * self.m)
        for i in range(self.x):
            a, b = self.lattice.single_to_double_index(perm[i])
            state0[a, b] = 1
        self.state = state0

    def indexfinder(self, arr, site_state):
        """
        Given an occupation array, returns a list of occupied sites
        (double indexed) in a given site_state

        Args:
            arr:
            site_state:

        Returns: sites in arr satisfying

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
        """

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
        """
        Pick a single occupied lattice site: move to an unoccupied site if
        energetically favourable (or due to thermal fluctuation)
        Metropolis Monte Carlo algorithm
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
        returns a list with the number of bonds at each MC step.
        If verbose==True, prints parameter list. """
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


class Ising2D:
    """Class for 2D Ising Model"""

    def __init__(self, n, m, J, kT, lattice_type):
        """
        Initialisation is easiest if either kT or J is set = 1
        and the other parameter varied, since it is the ratio kT/J
        that determines the beaviour of the system

        Args:
            n (int): linear size (y)
            m (int): linear size (x)
            J: Coupling strength
            kT: Temperature
            lattice_type: Lattice object (HexLattice or SquareLattice: recommended = SquareLattice)
        """
        self.n, self.m = n, m
        self.lattice = lattice_type(n, m)
        self.J, self.kT = J, kT
        self.initialise_system()

    def initialise_system(self):
        """Initialise a system of spins (n x m array), each with
        independent and equal probability of being up (+1) or down (-1)"""
        initial = np.random.rand(self.n, self.m)
        boolean = initial > 0.5
        self.state = 2 * boolean - 1

    def metropolis_sim(self, n_steps):
        """Monte Carlo simulation using Metropolis algorithm"""
        for i in range(n_steps):
            self.update_single_spin()

    def update_single_spin(self):
        """
        Metropolis algorithm: Pick a single lattice site at random and flip
        if energetically favourable (or due to thermal fluctuation)
        Metropolis Monte Carlo algorithm
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
        """Given a single lattice site, construct a cluster
        (list of tuples [(a, b)] representing lattice sites of cluster members)
        according to Wolff algorithm"""
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
        then flip all spins in cluster"""
        site = np.random.randint(self.n * self.m)
        i, j = self.lattice.single_to_double_index(site)
        self.flip_wolff_cluster(i, j)

    def display_state(self, title):
        """Plot lattice as monochrome heatmap using matplotlib.pyplot"""
        plt.imshow(self.state, cmap='hot', interpolation='nearest')
        plt.title(title)
        plt.show()

    def magnetisation_per_spin(self):
        """
        Returns: average magnetisation per spin (current state)
        """
        return np.mean(self.state)
