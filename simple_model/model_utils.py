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
    If a perturbation is used, the x-axis is aligned so stimulus onset is 0 and the stimulation window is shaded."""

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

    # compute acceleration from velocity (derivative)
    v_out = x[:, 2]
    v_up = x[:, 3]
    acc_out = np.diff(v_out, prepend=v_out[0]) / params.dt
    acc_up = np.diff(v_up, prepend=v_up[0]) / params.dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # trajectory plot
    axes[0, 0].plot(x[:, 0], x[:, 1], color="black")
    axes[0, 0].scatter([x[0, 0]], [x[0, 1]], label="start")
    axes[0, 0].scatter([x[-1, 0]], [x[-1, 1]], label="end")

    axes[0, 0].set_xlabel("Outward position")
    axes[0, 0].set_ylabel("Upward position")
    axes[0, 0].set_title(f"{title}: trajectory")
    axes[0, 0].legend()

    # arm outward velocity plot
    axes[0, 1].plot(t, v_out, color="black")

    if perturbation is not None:
        axes[0, 1].axvspan(stim_start, stim_end, color="#4A90E2", alpha=0.25)
        axes[0, 1].text(
            (stim_end)/2,
            axes[0, 1].get_ylim()[1]*0.9,
            "Stim",
            ha="left",
            color="#4A90E2"
        )
        axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=1)

    axes[0, 1].axhline(0, color="gray", linestyle="--", linewidth=1)

    axes[0, 1].set_xlabel("Time from stimulus (s)")
    axes[0, 1].set_ylabel("Outward velocity")
    axes[0, 1].set_title(f"{title}: outward velocity")

    # arm upward velocity plot
    axes[1, 0].plot(t, v_up, color="black")

    if perturbation is not None:
        axes[1, 0].axvspan(stim_start, stim_end, color="#4A90E2", alpha=0.25)
        axes[1, 0].text(
            (stim_end)/2,
            axes[1, 0].get_ylim()[1]*0.9,
            "Stim",
            ha="left",
            color="#4A90E2"
        )
        axes[1, 0].axvline(0, color="red", linestyle="--", linewidth=1)

    axes[1, 0].axhline(0, color="gray", linestyle="--", linewidth=1)

    axes[1, 0].set_xlabel("Time from stimulus (s)")
    axes[1, 0].set_ylabel("Upward velocity")
    axes[1, 0].set_title(f"{title}: upward velocity")

    # acceleration plot (both outward and upward)
    axes[1, 1].plot(t, acc_out, color="blue", label="Outward", alpha=0.7)
    axes[1, 1].plot(t, acc_up, color="green", label="Upward", alpha=0.7)

    if perturbation is not None:
        axes[1, 1].axvspan(stim_start, stim_end, color="#4A90E2", alpha=0.25)
        axes[1, 1].text(
            (stim_end)/2,
            axes[1, 1].get_ylim()[1]*0.9,
            "Stim",
            ha="left",
            color="#4A90E2"
        )
        axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=1)

    axes[1, 1].axhline(0, color="gray", linestyle="--", linewidth=1)

    axes[1, 1].set_xlabel("Time from stimulus (s)")
    axes[1, 1].set_ylabel("Acceleration")
    axes[1, 1].set_title(f"{title}: acceleration")
    axes[1, 1].legend()

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