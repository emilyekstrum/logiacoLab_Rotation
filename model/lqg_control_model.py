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
from dataclasses import dataclass
from scipy.linalg import solve_discrete_are

Array = np.ndarray

# Define LQG parameters and model
@dataclass
class LQGParams:
    """ Parameters for the LQG control model. """
    
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
        else:
            self.target = np.asarray(self.target, dtype=float)
            if self.target.shape != (4,):
                raise ValueError("Target state must be a 4-dimensional vector [p_out, p_up, v_out, v_up]")
            
@dataclass
class Perturbation:
    """ Perturbation configuration"""

    kind: str = None # None, mossy, inta_rn, inta_general
    onset_idx: int = 45 # start of perturbation in time steps, relative to reach onset
    duration: int = 5 # duration of perturbation in time steps
    pulse: np.array = None # shape (2,) for motor output pulse
    observer_bias: np.ndarray = None # shape (4,) for mossy perturbation bias on state observation
    general_noise_std: float = 0.15 # standard deviation of noise added to control input for general perturbation

    def __post_init__(self) -> None:
        valid_kinds = {None, "mossy", "inta_rn", "inta_general"}

        if self.kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {self.kind!r}")

        if self.pulse is None:
            self.pulse = np.array([-2.0, -1.0], dtype=float)
        else:
            self.pulse = np.asarray(self.pulse, dtype=float)
            if self.pulse.shape != (2,):
                raise ValueError("pulse must have shape (2,) for [u_out, u_up].")

        if self.observer_bias is None:
            self.observer_bias = np.array([0.0, 0.0, 0.8, 0.4], dtype=float)
        else:
            self.observer_bias = np.asarray(self.observer_bias, dtype=float)
            if self.observer_bias.shape != (4,):
                raise ValueError("observer_bias must have shape (4,) for state bias.")

        if self.onset_idx < 0 or self.duration < 0:
            raise ValueError("onset_idx and duration must be non-negative.")



# Define linear kinematic system dynamics
def build_state_space(params: LQGParams):
    """ Constructs the state-space representation of the system dynamics based on parameters. """

    dt = params.dt
    drag = params.drag
    b = params.control_gain

    # x = [p_out, p_up, v_out, v_up]
    # control passive limb evolution with drag, and active control input with gain b
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, drag, 0],
        [0, 0, 0, drag]
    ])

    # maps motor commands to changes in velocity, which then affect position through A
    B = np.array([
        [0, 0],
        [0, 0],
        [b*dt, 0],
        [0, b*dt]
    ])
    
    # observe all satate variables initially
    # defines what the cerebellum receives as input for state estimation and learning
    C = np.eye(4)

    return A, B, C


def dlqr(A, B, Q, R):
    """ Solves the discrete-time Linear Quadratic Regulator (LQR) problem. """

    P = solve_discrete_are(A, B, Q, R) # solve the discrete-time algebraic Riccati equation to find the optimal cost-to-go matrix P
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A) # compute the optimal state feedback gain K using the solution P
    return K, P 

def build_cost_matrices():
    """ Constructs the state and control cost matrices for the LQG problem. """

    # state cost: penalize deviation from target state, with higher weight on position errors
    # penalize position error stronlgy, velocity error less so, to encourage accurate reaching while allowing for some variability in movement speed
    Q = np.diag([120.0, 120.0, 8.0, 8.0]) # weights for [p_out, p_up, v_out, v_up]

    # control cost: penalize large control inputs to encourage efficient movements
    # penalize control inputs more strongly to encourage the system to find efficient motor commands that achieve the target state with minimal effort, while still allowing for necessary adjustments to reach the target accurately
    R = np.diag([0.3, 0.3]) # weights for [u_out, u_up]

    return Q, R

def dlqe(A, C, W, V):
    """ Solves the discrete-time Linear Quadratic Estimator (LQE) problem. """

    # solve estimator riccati equation for dual system
    P = solve_discrete_are(A.T, C.T, W, V) # solve the discrete-time algebraic Riccati equation to find the optimal estimation error covariance matrix P
    L = P @ C.T @ np.linalg.inv(C @ P @ C.T + V) # compute the optimal Kalman gain L using the solution P
    return L, P

def build_noise_matrices(params: LQGParams):
    """ Constructs the process and measurement noise covariance matrices. """

    W = params.W_scale * np.diag([1.0, 1.0, 2.0, 2.0]) # process noise is higher for velocity components, reflecting greater variability in movement execution compared to position, which is more directly influenced by control inputs and sensory feedback
    V = params.V_scale * np.diag([1.0, 1.0, 1.0, 1.0]) # measurement noise is smaller than process noise, reflecting that sensory feedback is relatively reliable compared to the inherent variability in movement execution

    return W, V

def make_time_basis(T, n_basis=8, width=10):
    """ Gaussian temporal basis funcitons across the reach
    returns Phi with shape [T, n_basis]"""

    centers = np.linspace(10, T-20, n_basis) # centers of the Gaussian basis functions, spaced evenly across the reach duration, starting slightly after the reach onset (10 time steps) and ending slightly before the reach offset (20 time steps before T) to ensure coverage of the entire movement period while avoiding edge effects where basis functions might be less effective
    t = np.aarange(T)[:, None] # time vector from 0 to T-1, reshaped to be a column vector for broadcasting with centers
    Phi = np.exp(-0.5 * ((t-centers[None, :])/width) ** 2) # compute the Gaussian basis functions by calculating the exponential of the negative squared distance between each time point and each center, normalized by the width parameter to control the spread of the basis functions. 
    
    # matrix Phi where each column corresponds to a Gaussian basis function centered at a specific time point, and each row corresponds to a time point in the reach duration.
    Phi /= Phi.max(axis=0, keepdims=True)

    return Phi


def initial_feedforward_weights(n_basis=8, n_u = 2):
    # initialize feedforward weights to zero
    # initially there is no learned feedforward command and the system relies entirely on feedback control to achieve the target state.
    return np.zeros((n_basis, n_u)) 


# simulate a reach with LQG control and cerebellar adaptation
def simulate_reach(
        params: LQGParams,
        K, L, Q, R, W, V, 
        Phi, ff_weights,
        perturbation: Perturbation = None,
        x0=None,
        rng=None):
    
    """ Simulates a single reach using the LQG control model with cerebellar adaptation. """

    if rng is None: 
        rng = np.random.default_rng()
    if x0 is None:
        x0 = np.zeros(4) # start at rest with paw at origin

    A, B, C = build_state_space(params)
    T = params.T

    x = np.zeros((t, 4)) # state trajectory
    xhat_pred = zp.zeros((T, 4)) # predicted state trajectory from internal model before observing current state
    xhat = np.zeros((T, 4)) # estimated state trajectory from internal model
    y = np.zeros((T, 4)) # observed state trajectory with measurement noise
    yhat = np.zeros((T, 4)) # predicted observations from internal model
    ytilde = np.zeros((T, 4)) # observation prediction error trajectory
    u_nom = np.zeros((T, 2)) # nominal control input trajectory from feedback controller
    u_app = np.zeros((T, 2)) # applied control input trajectory including feedforward command and perturbations
    u_ff = Phi @ ff_weights # feedforward control command trajectory from cerebellar adaptation

    x[0] = x0 # initialize true state trajectory with initial state
    xhat[0] = x0 # initialize state estimate to true initial state
    y[0] = C @ x[0] + rng.multivariate_normal(np.zeros(4), V) # initial observation with measurement noise
    yhat[0] = C @ xhat[0] # initial predicted observation from internal model
    ytilde[0] = y[0] - yhat[0] # initial observation prediction error

    target = params.target

    for t in range(1, T):
        # prediction step
        xhat_pred[t] = A @ xhat[t-1] + B @ u_app[t-1] # predict next state based on previous state estimate and applied control input

        # mossy fiber perturbation enters observer/internal model
        if perturbation is not None and perturbation.kind == "mossy":
            if perturbation.oneset_idx <= t < perturbation.onset_idx + perturbation.duration:
                xhat_pred[t] += perturbation.observer_bias # add bias to predicted state to simulate altered sensory input from mossy fiber perturbation

        # predicted observation
        yhat[t] = C @ xhat_pred[t] 

        # real observation
        obs_noise = rng.multivariate_normal(np.zeros(4), V) # sample measurement noise for current time step
        y[t] = C @ x[t] + obs_noise # true observation with measurement noise

        # prediction error
        ytilde[t] = y[t] - yhat[t] # compute observation prediction

        #control around target state
        state_error = xhat[t] - target # compute error between current state estimate and target state
        u_nom[t] = -K @ state_error + u_ff[t] # compute nominal control input from feedback controller

        # applied control perturbed at output
        u_app[t] = u_nom[t].copy()

        if perturbation is not None and perturbation.kind in ['inta_rn', 'inta_general']:
            if perturbation.onset_idx <= t < perturbation.onset_idx + perturbation.duration:
                u_app[t] += perturbation.pulse # add perturbation pulse to control input to simulate intra-cerebellar or general motor output perturbation

            if perturbation.kind == 'inta_general':
                # make less selective
                u_app[t] += rng.normal(0, 0.5, size=2) # add some random noise to the control input to simulate a more general perturbation that affects multiple aspects of motor output

        # process noise
        proc_noise = rng.multivariate_normal(np.zeros(4), W) # sample process noise for current time step

        # plant update
        x[t] = A @ x[t-1] + B @ u_app[t] + proc_noise # update true state based on previous state, applied control input, and process noise


    # trial cost
    J = 0.0
    for t in range(T):
        state_error = x[t] - target
        J += state_error.T @ Q @ state_error + u_app[t].T @ R @ u_app[t] # accumulate cost based on state error and control effort

    return {
        'x': x, # true state trajectory
        'xhat' : xhat, # estimated state trajectory from internal model
        'xhat_pred' : xhat_pred, # predicted state trajectory from internal model before observing current state
        'y': y, # observed state trajectory with measurement noise
        'yhat': yhat, # predicted observations from internal model
        'ytilde': ytilde, # observation prediction error trajectory
        'u_nom': u_nom, # nominal control input trajectory from feedback controller
        'u_app': u_app, # applied control input trajectory including feedforward command and perturbations
        'u_ff': u_ff, # feedforward control command trajectory from cerebellar adaptation
        'J': J # total cost for the trial
    }
