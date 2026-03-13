""" Utils for LQG control model"""

import numpy as np
from scipy.linalg import solve_discrete_are
from lqg_control_model import LQGController
import matplotlib.pyplot as plt


def plot_reach(result, title="Reach"):
    """ Plots the reach trajectory and velocity profiles from the simulation results. """

    x = result["x"]
    t = np.arange(len(x))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # trajectory
    axes[0].plot(x[:, 0], x[:, 1])
    axes[0].scatter([x[0, 0]], [x[0, 1]], label="start")
    axes[0].scatter([x[-1, 0]], [x[-1, 1]], label="end")
    axes[0].set_xlabel("Outward position")
    axes[0].set_ylabel("Upward position")
    axes[0].set_title(f"{title}: trajectory")
    axes[0].legend()

    # outward velocity
    axes[1].plot(t, x[:, 2])
    axes[1].set_xlabel("Time bin")
    axes[1].set_ylabel("Outward velocity")
    axes[1].set_title(f"{title}: outward velocity")

    # upward velocity
    axes[2].plot(t, x[:, 3])
    axes[2].set_xlabel("Time bin")
    axes[2].set_ylabel("Upward velocity")
    axes[2].set_title(f"{title}: upward velocity")

    plt.tight_layout()
    plt.show()


def 