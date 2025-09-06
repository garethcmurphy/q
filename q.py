"""qauntim lattice
 gas 1d simulation"""
import numpy as np
import matplotlib.pyplot as plt

class QuantumLatticeGas1D:
    """
    A class to simulate a 1D Quantum-Lattice-Gas (QLG) model.
    """

    def __init__(self, x, steps, theta):
        """
        Initialize the QLG simulation parameters.

        Args:
            x (np.ndarray): Spatial grid points.
            steps (int): Number of simulation steps.
            theta (float): Parameter for the QLG step function.
        """
        self.x = x
        self.steps = steps
        self.theta = theta
        self.snapshots = []

    def initialize_density(self):
        """
        Initialize the localized "density" bump and split into ψL and ψR.
        """
        rho0 = 0.6 * np.exp(-((self.x - 0.3) / 0.05) ** 2)
        self.psiL = 0.5 * rho0
        self.psiR = 0.5 * rho0

    def run_simulation(self, qlg_step):
        """
        Run the QLG simulation for the specified number of steps.

        Args:
            qlg_step (function): Function to perform a single QLG step.
        """
        for n in range(self.steps):
            self.psiL, self.psiR = qlg_step(self.psiL, self.psiR, self.theta)

            # Save snapshots at specific steps
            if n in (0, self.steps // 3, 2 * self.steps // 3, self.steps - 1):
                rho = self.psiL + self.psiR
                u = (self.psiR - self.psiL) / (rho + 1e-12)  # Avoid division by zero
                self.snapshots.append((n, rho.copy(), u.copy()))

    def plot_snapshots(self):
        """
        Plot the density snapshots at the saved steps.
        """
        plt.figure()
        for n, rho, _ in self.snapshots:
            plt.plot(self.x, rho, label=f"step {n}")
        plt.xlabel("x")
        plt.ylabel("density ~ (ψL + ψR)")
        plt.legend()
        plt.title("1D Quantum-Lattice-Gas toy (classical sim)")
        plt.show()

def qlg_step(psiL, psiR, theta):
    # Collision: unitary rotation (cosθ  sinθ; -sinθ  cosθ)
    c, s = np.cos(theta), np.sin(theta)
    Lc = c*psiL + s*psiR
    Rc = -s*psiL + c*psiR

    # Streaming (periodic): left-movers shift left, right-movers shift right
    Ls = np.roll(Lc, +1)  # moves to the left in x
    Rs = np.roll(Rc, -1)  # moves to the right in x
    return Ls, Rs

if __name__ == "__main__":
    # Define simulation parameters
    x = np.linspace(0, 1, 100)  # Spatial grid
    steps = 100  # Number of simulation steps
    theta = 0.1  # Parameter for QLG step

    # Create an instance of the QLG simulation
    qlg_sim = QuantumLatticeGas1D(x, steps, theta)

    # Initialize the density
    qlg_sim.initialize_density()

    # Run the simulation
    qlg_sim.run_simulation(qlg_step)

    # Plot the results
    qlg_sim.plot_snapshots()