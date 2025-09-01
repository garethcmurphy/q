## lax friedreichs q function

import numpy as np
import matplotlib.pyplot as plt

def qlg_step(psiL, psiR, theta):
    # Collision: unitary rotation (cosθ  sinθ; -sinθ  cosθ)
    c, s = np.cos(theta), np.sin(theta)
    Lc = c*psiL + s*psiR
    Rc = -s*psiL + c*psiR

    # Streaming (periodic): left-movers shift left, right-movers shift right
    Ls = np.roll(Lc, +1)  # moves to the left in x
    Rs = np.roll(Rc, -1)  # moves to the right in x
    return Ls, Rs

def qlg_1d(N=512, steps=400, theta=0.12):
    x = np.linspace(0, 1, N, endpoint=False)

    # Two populations: initialize a localized "density" bump
    rho0 = 0.6 * np.exp(-((x-0.3)/0.05)**2)
    psiL = 0.5 * rho0
    psiR = 0.5 * rho0

    snapshots = []
    for n in range(steps):
        psiL, psiR = qlg_step(psiL, psiR, theta)
        if n in (0, steps//3, 2*steps//3, steps-1):
            rho = psiL + psiR
            u   = (psiR - psiL) / (rho + 1e-12)
            snapshots.append((n, rho.copy(), u.copy()))
    return x, snapshots

if __name__ == "__main__":
    x, snaps = qlg_1d()
    plt.figure()
    for n, rho, _ in snaps:
        plt.plot(x, rho, label=f"step {n}")
    plt.xlabel("x"); plt.ylabel("density ~ (ψL+ψR)"); plt.legend()
    plt.title("1D Quantum-Lattice-Gas toy (classical sim)")
    plt.show()