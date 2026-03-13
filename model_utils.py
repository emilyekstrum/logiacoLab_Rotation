""" Utils for LQG control model"""

import numpy as np
from scipy.linalg import solve_discrete_are
from model.lqg_control_model import LQGController
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


def compute_behavior_metrics(result, target):
    x = result["x"]
    ytilde = result["ytilde"]
    u_app = result["u_app"]

    endpoint = x[-1, :2]
    endpoint_error = np.linalg.norm(endpoint - target[:2])

    outward_vel = x[:, 2]
    upward_vel = x[:, 3]

    peak_outward_vel = outward_vel.max()
    min_outward_vel = outward_vel.min()

    # correction metric
    # after the minimum velocity, how much does velocity rebound
    min_idx = np.argmin(outward_vel)
    if min_idx < len(outward_vel) - 1:
        rebound = outward_vel[min_idx+1:].max() - outward_vel[min_idx]
        rebound_idx = min_idx + 1 + np.argmax(outward_vel[min_idx+1:])
    else:
        rebound = 0.0
        rebound_idx = min_idx

    innovation_norm = np.linalg.norm(ytilde, axis=1)
    peak_innovation = innovation_norm.max()

    return {
        "endpoint_error": endpoint_error,
        "peak_outward_vel": peak_outward_vel,
        "min_outward_vel": min_outward_vel,
        "rebound_magnitude": rebound,
        "rebound_idx": rebound_idx,
        "peak_innovation": peak_innovation,
        "cost_J": result["J"]
    }