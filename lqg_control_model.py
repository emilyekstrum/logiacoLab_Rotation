""" Linear Quadratic Gaussian (LQG) Control Model for cerebellar pathways.

State:
- x(t) : State vector representing the system's current state at time t.
    - P_out(t) : outward paw position at time t.
    - P_up(t) : upward paw position at time t.
    - V_out(t) : outward paw velocity at time t.
    - V_up(t) : upward paw velocity at time t.

Control Input:
- u(t) : Control input vector representing the actions taken by the system at time t; net descending motor command
    - u_out(t) : control input for outward paw movement at time t.
    - u_up(t) : control input for upward paw movement at time t.

Intenral model/estimator: Cerebellum

Controller: motor circuits / motor cortex

Parameters: 
A: State transition matrix representing the dynamics of the system.
B: Control input matrix representing how control inputs affect the state.
Q: State cost matrix representing the cost associated with being in a particular state.
R: Control cost matrix representing the cost associated with applying a particular control input.
C: Output matrix representing how the state is observed or measured.
L: Kalman gain matrix representing how the estimator updates its state estimates based on new observations.
W: Process noise covariance matrix representing the uncertainty in the system dynamics.
V: Measurement noise covariance matrix representing the uncertainty in the observations.
theta: adaptation weights representing the learning parameters for updating the internal model based on prediction errors.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.linalg import solve_discrete_are

# Define LQG parameters and model
@dataclass
class LQGParams:
    dt: float = 0.01 # 10 ms time bins
    drag: float = 0.92 #velocity retention
    control_gain: float = 0.1 # command -> acceleration gain

    # noise covariances
    W_scale: float = 1e-4 # process noise 
    V_scale: float = 5e-4 # measurement noise

    #target state: [p_out, p_up, v_out, v_up]
    target: np.array = None

    #horizon
    T: int = 120 # 1.2 second reach

    # adaptation
    eta: float = 0.04 #learning rate for updating internal model
    ff_decay: float = 0.995 # slow forgetting of feedforward weights

    def __post_init__(self):
        if self.target is None:
            #reach target at outward 1.0, upward 0.5, with zero termninal velocity
            self.target = np.array([1.0, 0.5, 0.0, 0.0], dtype=float)


# Define linear kinematic system dynamics
def build_state_space(params: LQGParams):
    dt = params.dt
    drag = params.drag
    b = params.control_gain

    # x = [p_out, p_up, v_out, v_up]
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, drag, 0],
        [0, 0, 0, drag]
    ])

    B = np.array([
        [0, 0],
        [0, 0],
        [b*dt, 0],
        [0, b*dt]
    ])
    
    # observe all satate variables initially
    C = np.eye(4)

    return A, B, C