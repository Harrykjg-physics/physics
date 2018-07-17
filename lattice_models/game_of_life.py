from lattice_models import *
import matplotlib.pyplot as plt


class GameOfLife:
    def __init__(self, n, m, lattice_type=SquareLattice2):
        """ Conway's Game of Life

        Args:
            n (int): size y
            m (int): size x
            lattice_type: valid lattice subclass, default = SquareLattice2

        """

        self.n, self.m = n, m
        self.lattice = lattice_type(n, m)
        self.state = np.random.randint(
            0, 2, self.n * self.m).reshape([self.n, self.m])

    def evolve_one_step(self):
        """Updates game by one step according to canonical rules - from wikipedia:

        'Any live cell with fewer than two live neighbors dies, as if by under population.
        Any live cell with two or three live neighbors lives on to the next generation.
        Any live cell with more than three live neighbors dies, as if by overpopulation.
        Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.'
        https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life

        """
        new_state = np.zeros([self.n, self.m])  # default = dead

        for k in range(self.n * self.m):
            i, j = self.lattice.single_to_double_index(k)
            live_neighbours = 0

            for nind in range(self.lattice.n_neighbours):
                p, q = self.lattice.neighbour_list[i, j, nind, :]
                live_neighbours += self.state[p, q]

            if self.state[i, j] == 0 and live_neighbours == 3:
                new_state[i, j] = 1
            elif self.state[i, j] == 1 and (
                    live_neighbours == 2 or live_neighbours == 3):
                new_state[i, j] = 1

        self.state = new_state


if __name__=='__main__':
    """Animated game of life (runs indefinitely)"""
    from matplotlib import animation

    n, m, anim_interval = 20, 20, 80
    print("Conway's Game of Life")
    game = GameOfLife(n, m)

    fig = plt.figure()
    ax = plt.axes()
    ax.set_xticks([])
    ax.set_yticks([])
    im = plt.imshow(game.state, cmap='hot_r', animated=True)
    fig.suptitle('Step 0')

    steps = 0
    def update(*args):
        global steps
        steps += 1
        game.evolve_one_step()
        im.set_data(game.state)
        fig.suptitle('Step ' + str(steps))
        return im,

    ani = animation.FuncAnimation(
        fig, update, interval=anim_interval, blit=False)
    plt.show()

