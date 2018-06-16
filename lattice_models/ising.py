# Example: animated Ising2D model using metropolis algorithm

if __name__=='__main__':
    from lattice_models import *
    from matplotlib import animation

    n, m, = 100, 100
    J, kT = 1, 1
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
        ising.metropolis_sim(mc_steps_per_frame)
        global steps
        steps += mc_steps_per_frame
        im.set_data(ising.state)
        fig.suptitle(t + '\n%d MC steps' %steps)
        return im,

    ani = animation.FuncAnimation(fig, update, interval=anim_interval, blit=True)
    plt.show()

