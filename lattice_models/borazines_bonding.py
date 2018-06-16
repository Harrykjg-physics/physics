if __name__=='__main__':
    from lattice_models import *
    n, m, = 10, 10
    vdW = 0.06  # eV (borazine bonding energy)
    x = 3       # 3 molecules / trimer
    mc_steps = 10000

    kT_list = np.linspace(0.012, 0.025, 21)
    avg_bonds = []
    for kT in kT_list:
        trimer_model = SubstrateBonding2D(n, m, x, vdW, kT, HexLattice)
        bonds = trimer_model.run_sim(mc_steps)
        avg_bonds.append(np.mean(bonds))

    plt.scatter(kT_list, avg_bonds, marker='x')
    plt.title('Average number of bonds vs Temperature for borazine trimer')
    plt.xlabel('Temperature (eV)')
    plt.ylabel('Average number of bonds')
    plt.show()