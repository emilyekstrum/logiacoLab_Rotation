""" Utils for LQG control model"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.linalg import solve_discrete_are
from model.lqg_control_model import LQGController
import matplotlib.pyplot as plt



def stimulus_aligned_time(T, onset_idx, dt):
    """Returns time in seconds aligned so stimulus onset is 0. """

    return (np.arange(T) - onset_idx) * dt

def plot_reach(result, params, perturbation=None, title="Reach"):
    """Plot reach trajectory and velocity profiles.
    If a perturbation is used, the x-axis is aligned so stimulus = 0 and the stimulation window is shaded."""

    x = result["x"]
    T = len(x)

    # stimulus-aligned time axis
    if perturbation is not None:
        t = (np.arange(T) - perturbation.onset_idx) * params.dt
        stim_start = 0
        stim_end = perturbation.duration * params.dt
    else:
        t = np.arange(T) * params.dt
        stim_start = None
        stim_end = None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # trajectory plot
    axes[0].plot(x[:, 0], x[:, 1], color="black")
    axes[0].scatter([x[0, 0]], [x[0, 1]], label="start")
    axes[0].scatter([x[-1, 0]], [x[-1, 1]], label="end")

    axes[0].set_xlabel("Outward position")
    axes[0].set_ylabel("Upward position")
    axes[0].set_title(f"{title}: trajectory")
    axes[0].legend()

    # arm outward velocity plot
    axes[1].plot(t, x[:, 2], color="black")

    if perturbation is not None:
        axes[1].axvspan(stim_start, stim_end, color="#4A90E2", alpha=0.25)
        axes[1].axvline(0, color="red", linestyle="--", linewidth=1)

    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)

    axes[1].set_xlabel("Time from stimulus (s)")
    axes[1].set_ylabel("Outward velocity")
    axes[1].set_title(f"{title}: outward velocity")

    # arm upward velocity plot
    axes[2].plot(t, x[:, 3], color="black")

    if perturbation is not None:
        axes[2].axvspan(stim_start, stim_end, color="#4A90E2", alpha=0.25)
        axes[2].axvline(0, color="red", linestyle="--", linewidth=1)

    axes[2].axhline(0, color="gray", linestyle="--", linewidth=1)

    axes[2].set_xlabel("Time from stimulus (s)")
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