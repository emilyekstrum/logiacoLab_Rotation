""" Linear Quadratic Gaussian (LQG) Control Model for cerebellar pathways using general kinematics.

x(t) = A x(t-1) + B u(t-1) + w(t-1)  # state update with process noise
y(t) = C x(t) + v(t)  # observation with measurement noise

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

Plant: arm dynamics

Parameters: 
A: State transition matrix representing the dynamics of the system. 
    - array (4, 4): 4D state space.
B: Control input matrix representing how control inputs affect the state. 
    - array (4, 2): 4D state space, 2D control input.
Q: State cost matrix representing the cost associated with being in a particular state.
    - array (4, 4): 4D state space.
R: Control cost matrix representing the cost associated with applying a particular control input.
    - array (2, 2): 2D control input.
C: Output matrix representing how the state is observed or measured.
    - array (4, 4): 4D state space, 4D observation (initially observing all state variables).
L: Kalman gain matrix representing how the estimator updates its state estimates based on new observations.
    - array (4, 4): 4D state space, 4D observation.
W: Process noise covariance matrix representing the uncertainty in the system dynamics.
    - array (4, 4): 4D state space.
V: Measurement noise covariance matrix representing the uncertainty in the observations.
    - array (4, 4): 4D observation.

Perturbations:
- Mossy fiber perturbation: bias added to the predicted state in the internal model, simulating altered sensory input to the cerebellum.
    - xhat_pred(t) = A xhat(t-1) + B u_app(t-1) + observer_bias (during perturbation window)
        - observer_bias = velocity bias added (increased outward and upward velocity during perturbation)

- intA/RN perturbation: pulse added to the motor output, simulating a brief disruption of motor commands.
     - u_app(t) = u_nom(t) + u_ff(t) + pulse (during perturbation window)
        - pulse = strong braking force added to both outward and upward control inputs during perturbation window

- intA general perturbation: pulse plus noise added to the motor output, simulating a more general disruption of motor commands that includes variability.
    - u_app(t) = u_nom(t) + u_ff(t) + pulse + noise (during perturbation window)
        - noise = random variability added to control inputs during perturbation window, in addition to the pulse, to simulate a more general disruption of motor commands that includes variability
"""

import numpy as np
from dataclasses import dataclass
from scipy.linalg import solve_discrete_are
from typing import Optional

Array = np.ndarray

# Define LQG parameters and model
@dataclass
class LQGParams:
    """ Parameters for the LQG control model. """
    
    dt: float = 0.01 # 10 ms time bins
    drag: float = 0.92 #velocity retention
    control_gain: float = 1.0 # command -> acceleration gain (increased from 0.1 to allow reaching the target)

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
            #reach target at outward 2.0, upward 2.0, with zero termninal velocity
            self.target = np.array([2.0, 2.0, 0.0, 0.0], dtype=float)
        else:
            self.target = np.asarray(self.target, dtype=float)
            if self.target.shape != (4,):
                raise ValueError("Target state must be a 4-dimensional vector [p_out, p_up, v_out, v_up]")
            
@dataclass
class Perturbation:
    """ Perturbation configuration
    
    - mossy fiber perturbation: bias added to the predicted state in the internal model, simulating altered sensory input to the cerebellum
    - intA/RN perturbation: pulse added to the motor output, simulating a brief disruption of motor commands
    - intA general perturbation: pulse plus noise added to the motor output, simulating a more general disruption of motor commands that includes variability
    """

    kind: str = None # None, mossy, inta_rn, inta_general
    onset_idx: int = 45 # start of perturbation in time steps, relative to reach onset
    duration: int = 15 # duration of perturbation in time steps (150ms for visible position effects)
    pulse: np.array = None # shape (2,) for motor output pulse
    observer_bias: np.ndarray = None # shape (4,) for mossy perturbation bias on state observation
    general_noise_std: float = 0.5 # standard deviation of noise added to control input for general perturbation

    def __post_init__(self) -> None:
        valid_kinds = {None, "mossy", "inta_rn", "inta_general"}

        if self.kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {self.kind!r}")

        if self.pulse is None:
            self.pulse = np.array([-15.0, -10.0], dtype=float) # strong braking force to create visible behavioral effects
        else:
            self.pulse = np.asarray(self.pulse, dtype=float)
            if self.pulse.shape != (2,):
                raise ValueError("pulse must have shape (2,) for [u_out, u_up].")

        if self.observer_bias is None:
            self.observer_bias = np.array([0.0, 0.0, 3.0, 2.5], dtype=float) # strong velocity bias for visible prediction errors
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
    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, drag, 0.0],
            [0.0, 0.0, 0.0, drag],
        ],
        dtype=float,
    )

    # maps motor commands to changes in velocity, which then affect position through A
    B = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [b * dt, 0.0],
            [0.0, b * dt],
        ],
        dtype=float,
    )
    
    # observe all satate variables initially
    # defines what the cerebellum receives as input for state estimation and learning
    C = np.eye(4, dtype=float)

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
    # penalize control inputs more strongly to encourage the system to find efficient motor commands that achieve the target state with minimal effort, 
    # while still allowing for necessary adjustments to reach the target accurately
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

    # process noise is higher for velocity components, reflecting greater variability in movement execution compared to position, 
    # which is more directly influenced by control inputs and sensory feedback
    W = params.W_scale * np.diag([1.0, 1.0, 2.0, 2.0]) 

    # measurement noise is smaller than process noise, reflecting that sensory feedback is relatively reliable 
    # compared to the inherent variability in movement execution
    V = params.V_scale * np.diag([1.0, 1.0, 1.0, 1.0]) 

    return W, V

def make_time_basis(T, n_basis=8, width=10):
    """Create Gaussian temporal basis functions with shape [T, n_basis]."""
    if T <= 0:
        raise ValueError("T must be positive.")
    if n_basis <= 0:
        raise ValueError("n_basis must be positive.")
    if width <= 0:
        raise ValueError("width must be positive.")

    # centers start at 10 to avoid being too close to reach onset, and end at least 20 time steps before reach end to 
    # allow for learning effects to influence the reach trajectory
    centers = np.linspace(10, max(10, T - 20), n_basis) 
    t = np.arange(T)[:, None] # time vector of shape [T, 1] for broadcasting with centers of shape [n_basis]

    # compute Gaussian basis functions for each time step and basis function center, 
    # resulting in a [T, n_basis] matrix where each column is a Gaussian function centered at a different time point in the reach
    Phi = np.exp(-0.5 * ((t - centers[None, :]) / width) ** 2) 

    # normalize each basis function to have a maximum value of 1, preventing numerical issues with very small values and ensuring 
    # that the feedforward command can scale appropriately based on the learned weights without being dominated by the shape of the basis functions
    Phi /= np.maximum(Phi.max(axis=0, keepdims=True), 1e-12) 
    
    return Phi


def initial_feedforward_weights(n_basis=8, n_u = 2):
    # initialize feedforward weights to zero
    # initially there is no learned feedforward command and the system relies entirely on feedback control to achieve the target state.
    return np.zeros((n_basis, n_u), dtype=float) 

def update_feedforward_weights(
    ff_weights: Array,
    Phi: Array,
    ytilde: Array,
    B: Array,
    params: LQGParams) -> Array:
    """Update feedforward weights based on prediction errors using cerebellar-like learning.
    
    Learning rule: weights are adjusted to minimize future prediction errors by learning
    a feedforward command that compensates for predictable perturbations.
    
    Returns:
        Updated feedforward weights [n_basis, n_u]
    """
    
    # get velocity components of prediction error (indices 2-3)
    # use velocity errors as they reflect motor command errors most directly
    velocity_errors = ytilde[:, 2:4]  # [T, 2]
    
    # compute weight update: learn to produce motor commands that would have
    # reduced the velocity prediction errors we observed
    # delta_w = -eta * Phi.T @ velocity_errors
    # The negative sign means we learn to counteract the errors
    delta_w = -params.eta * (Phi.T @ velocity_errors)  # [n_basis, 2]
    
    # apply weight decay (forgetting) and add new learning
    ff_weights_new = params.ff_decay * ff_weights + delta_w
    
    return ff_weights_new

def train_with_adaptation(
    controller: 'LQGController',
    Phi: Array,
    n_trials: int = 100,
    perturbation: Optional[Perturbation] = None,
    perturbation_trials: Optional[tuple] = None,
    rng: Optional[np.random.Generator] = None) -> dict:
    """
    Train the controller over multiple trials with cerebellar adaptation.
    
    Args:
        controller: LQGController instance
        Phi: Temporal basis functions
        n_trials: Number of training trials
        perturbation: Perturbation to apply (if None, no perturbation)
        perturbation_trials: Tuple (start, end) indicating which trials have perturbation.
                           If None, perturbation applied to all trials (if perturbation is not None)
        rng: Random number generator
    
    Returns:
        Dictionary containing training history:
            - 'ff_weights_history': Weight evolution [n_trials+1, n_basis, n_u]
            - 'cost_history': Cost per trial [n_trials]
            - 'endpoint_error_history': Endpoint error per trial [n_trials]
            - 'trials': List of trial results for selected trials
    """
    
    if rng is None:
        rng = np.random.default_rng()
    
    # initialize weights
    ff_weights = controller.make_default_ff_weights(n_basis=Phi.shape[1])
    
    # storage for history
    n_basis = Phi.shape[1]
    ff_weights_history = np.zeros((n_trials + 1, n_basis, 2))
    ff_weights_history[0] = ff_weights.copy()
    
    cost_history = np.zeros(n_trials)
    endpoint_error_history = np.zeros(n_trials)
    
    # store full trial data for selected trials
    save_trial_indices = [0, n_trials//4, n_trials//2, 3*n_trials//4, n_trials-1]
    trials_data = {}
    
    # determine perturbation schedule
    if perturbation_trials is None and perturbation is not None:
        # apply perturbation to all trials
        pert_start, pert_end = 0, n_trials
    elif perturbation_trials is not None:
        pert_start, pert_end = perturbation_trials
    else:
        pert_start, pert_end = 0, 0  # no perturbation
    
    for trial in range(n_trials):
        # determine if this trial has perturbation
        if perturbation is not None and pert_start <= trial < pert_end:
            trial_pert = perturbation
        else:
            trial_pert = None
        
        # simulate reach
        result = controller.simulate_reach(
            Phi=Phi,
            ff_weights=ff_weights,
            perturbation=trial_pert,
            rng=rng
        )
        
        # record metrics
        cost_history[trial] = result['J']
        endpoint_error = np.linalg.norm(result['x'][-1, :2] - controller.params.target[:2])
        endpoint_error_history[trial] = endpoint_error
        
        # save full trial data for selected trials
        if trial in save_trial_indices:
            trials_data[trial] = result
        
        # update weights based on prediction errors
        ff_weights = update_feedforward_weights(
            ff_weights=ff_weights,
            Phi=Phi,
            ytilde=result['ytilde'],
            B=controller.B,
            params=controller.params
        )
        
        ff_weights_history[trial + 1] = ff_weights.copy()
    
    return {
        'ff_weights_history': ff_weights_history,
        'cost_history': cost_history,
        'endpoint_error_history': endpoint_error_history,
        'trials': trials_data,
        'final_ff_weights': ff_weights
    }

@dataclass
class LQGController:
    """ LQG Controller class encapsulating the parameters and methods for simulating reaches with cerebellar adaptation. """

    def __init__ (self, params: LQGParams):
        self.params = params if params is not None else LQGParams()

        self.A, self.B, self.C = build_state_space(params)
        self.Q, self.R = build_cost_matrices()
        self.W, self.V = build_noise_matrices(params)

        self.K, self.P_lqr = dlqr(self.A, self.B, self.Q, self.R)
        self.L, self.P_kf = dlqe(self.A, self.C, self.W, self.V)


    # simulate a reach with LQG control and cerebellar adaptation
    def simulate_reach(
        self,
        Phi: Array,
        ff_weights: Array,
        perturbation: Optional[Perturbation] = None,
        x0: Optional[Array] = None,
        rng: Optional[np.random.Generator] = None,):
        
        """ Instance wrapper around simulate_reach """

        return simulate_reach(
        params=self.params,
        K=self.K,
        L=self.L,
        Q=self.Q,
        R=self.R,
        W=self.W,
        V=self.V,
        Phi=Phi,
        ff_weights=ff_weights,
        perturbation=perturbation,
        x0=x0,
        rng=rng,
        )
    
    def make_default_basis(self, n_basis: int = 8, width: float = 10.0) -> Array:
        """Convenience helper to create default temporal basis functions."""
        return make_time_basis(self.params.T, n_basis=n_basis, width=width)

    def make_default_ff_weights(self, n_basis: int = 8) -> Array:
        """Convenience helper to create zero-initialized feedforward weights."""
        return initial_feedforward_weights(n_basis=n_basis, n_u=2)


def simulate_reach(
    params: LQGParams,
    K: Array,
    L: Array,
    Q: Array,
    R: Array,
    W: Array,
    V: Array,
    Phi: Array,
    ff_weights: Array,
    perturbation: Optional[Perturbation] = None,
    x0: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None): 
        """ simulates a reach trajectory under LQG control with cerebellar adaptation and optional perturbations. 
        Returns the state trajectory, control inputs, and cost for the trial. """

        if rng is None: 
            rng = np.random.default_rng()
        if x0 is None:
            x0 = np.zeros(4) # start at rest with paw at origin

        A, B, C = build_state_space(params)
        T = params.T

        x = np.zeros((T, 4)) # state trajectory
        xhat_pred = np.zeros((T, 4)) # predicted state trajectory from internal model before observing current state
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

        # compute initial control (will be applied at t=1)
        state_error = xhat[0] - target
        u_nom[0] = -K @ state_error + u_ff[0]
        u_app[0] = u_nom[0].copy()

        for t in range(1, T):
            # apply motor output perturbations to the control that will affect THIS timestep
            # ensures perturbation affects x[t] when onset_idx <= t < onset_idx + duration
            u_to_apply = u_app[t-1].copy()
            
            if perturbation is not None and perturbation.kind in ['inta_rn', 'inta_general']:
                if perturbation.onset_idx <= t < perturbation.onset_idx + perturbation.duration:
                    u_to_apply += perturbation.pulse
                    
                    # add noise only during perturbation window for inta_general
                    if perturbation.kind == 'inta_general':
                        u_to_apply += rng.normal(0, perturbation.general_noise_std, size=2)
            
            # plant update - uses the potentially perturbed control
            proc_noise = rng.multivariate_normal(np.zeros(4), W)
            x[t] = A @ x[t-1] + B @ u_to_apply + proc_noise

            # observation of current state
            obs_noise = rng.multivariate_normal(np.zeros(4), V)
            y[t] = C @ x[t] + obs_noise

            # estimator prediction
            # uses the UNPERTURBED control from internal model's perspective
            xhat_pred[t] = A @ xhat[t-1] + B @ u_app[t-1]

            # Mossy fiber perturbation enters observer/internal model
            if perturbation is not None and perturbation.kind == "mossy":
                if perturbation.onset_idx <= t < perturbation.onset_idx + perturbation.duration:
                    xhat_pred[t] += perturbation.observer_bias

            # predicted observation
            yhat[t] = C @ xhat_pred[t]

            # prediction error
            ytilde[t] = y[t] - yhat[t]
            
            # correction step (Kalman update)
            xhat[t] = xhat_pred[t] + L @ ytilde[t]

            # compute control for NEXT timestep based on current estimate
            state_error = xhat[t] - target
            u_nom[t] = -K @ state_error + u_ff[t]
            u_app[t] = u_nom[t].copy()


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

__all__ = [
    "LQGParams",
    "Perturbation",
    "LQGController",
    "build_state_space",
    "build_cost_matrices",
    "build_noise_matrices",
    "dlqr",
    "dlqe",
    "make_time_basis",
    "initial_feedforward_weights",
    "simulate_reach",
    "update_feedforward_weights",
    "train_with_adaptation",
]
