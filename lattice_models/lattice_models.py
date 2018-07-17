import numpy as np


class Lattice:

    def __init__(self, n, m):
        """ Superclass for Lattice objects.

        This class provides an interface. A subclass should contain a
        neighbours(self, i, j) function defining the neighbouring points
        (their indices) to point (i, j). This defines the lattice geometry.

        The subclass constructor should also define the number of neighbours at
        each point.

        Args:
            n (int): linear size in y dimension
            m (int): linear size in x dimension

        """
        self.n = n
        self.m = m
        self.generate_neighbour_list()

    def single_to_double_index(self, k):
        """ Convert single to double index.

        Args:
            k (int): single index for array element

        Returns:
            i, j: corresponding double index for array element

        """
        return (k // self.m), np.mod(k, self.m)

    def double_to_single_index(self, i, j):
        """ Convert double to single index

        Args:
            i, j: double index for array element

        Returns:
            k (int): single index for array element

        """
        return (i * self.n) + j

    def neighbours(self, i, j):
        """Must be implemented in the subclass used.

        See examples in subclasses HexLattice and SquareLattice.

        Raises:
            NotImplementedError: If subclass does not have such a method

        """
        raise NotImplementedError(
            'nearest neighbours function missing in class %s'
            % self.__class__.__name__)

    def generate_neighbour_list(self):
        """Generate an array of neighbours.

        self.neighbour_list[i, j, p, :] = a, b, where
        a, b = double indexed coordinates of pth neighbour of site (i, j)

        """

        n, m = self.n, self.m
        neighbour_list = np.zeros(
            [n, m, self.n_neighbours, 2], dtype=np.int)

        for s in range(n * m):
            i, j = self.single_to_double_index(s)
            nn = self.neighbours(i, j)
            neighbour_list[i, j, :, :] = nn
        self.neighbour_list = neighbour_list


class HexLattice(Lattice):

    def __init__(self, n, m):
        """ AKA Triangular lattice. Each point has siz neighbours.

        The unit tile is a rhombus.

        Args:
            n (int): linear size (0 axis)
            m (int): linear size (1 axis)

        """
        self.n_neighbours = 6
        super().__init__(n, m)

    def neighbours(self, i, j):
        """Nearest neighbouring sites.

         This function defines a hexagonal lattice:
                X X
              X 0 X
              X X

        Args:
            i, j (int): the original lattice site

        Returns:
             nn (ndarray): array with entries corresponding to the six nearest
             neighbours of the given lattice site"""
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
    def __init__(self, n, m):
        """Von Neumann Neighbourhood

        Args:
            n (int): linear size in y dimension
            m (int): linear size in x dimension

        """
        self.n_neighbours = 4
        super().__init__(n, m)

    def neighbours(self, i, j):
        """Nearest neighbouring sites (von Neumnann neighbourhood)


        This function defines a square lattice:
              X
            X 0 X
              X

        Args:
            i, j (int): the original lattice site

        Returns:
             nn (ndarray): array with entries corresponding to the four nearest
             neighbours of the given lattice site

        """
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
    def __init__(self, n, m):
        """ Moore neighbourhood.

        Args:
            n (int): linear size in y dimension
            m (int): linear size in x dimension
        """
        self.n_neighbours = 8
        super().__init__(n, m)

    def neighbours(self, i, j):
        """Nearest neighbouring sites (Moore neighbourhood).

        this function defines a square lattice:
              X X X
              X 0 X
              X X X

        Args:
            i, j (int): original lattice site

        Returns:
             nn (ndarray): array with entries corresponding to the eight nearest
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
