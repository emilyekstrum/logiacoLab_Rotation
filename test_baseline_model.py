""" Tests the baseline LQG control model with a simple 2D point mass system. """

import numpy as np
from scipy.linalg import solve_discrete_are
from lqg_control_model import LQGController
import model_utils

params = LQGController.Params()

A, B, C = LQGController.build_state_space(params)
Q, R = LQGController.build_cost_matrices()
W, V = LQGController.build_noise_matrices(params)

K, _ = LQGController.dlqr(A, B, Q, R)
L, _ = LQGController.dlqe(A, C, W, V)

Phi = LQGController.make_time_basis(params.T, n_basis=8, width=10)
ff_weights = LQGController.initial_feedforward_weights(n_basis=Phi.shape[1], n_u=2)

baseline = LQGController.simulate_reach(
    params=params,
    K=K, L=L,
    Q=Q, R=R,
    W=W, V=V,
    ff_weights=ff_weights,
    Phi=Phi,
    perturbation=None,
    rng=np.random.default_rng(0)
)

# simulate a reach without perturbation
LQGController.plot_reach(baseline, title="Baseline LQG Control Reach")


# simulate a reach with a mossy fiber perturbation
mossy_pert = LQGController.Perturbation(
    kind='mossy',
    onset_idx=45,
    duration=5,
    observer_bias=np.array([0.0, 0.0, 0.9, 0.5])
)

res_mossy = LQGController.simulate_reach(
    params=params,
    K=K, L=L,
    Q=Q, R=R,
    W=W, V=V,
    ff_weights=ff_weights,
    Phi=Phi,
    perturbation=mossy_pert,
    rng=np.random.default_rng(3)
)

model_utils.plot_reach(res_mossy, title="Mossy fiber perturbation")

# simulate a reach with a inta general perturbation
inta_general_pert = LQGController.Perturbation(
    kind='inta_general',
    onset_idx=45,
    duration=5,
    pulse=np.array([-2.2, -1.0])
)

res_inta_general = LQGController.simulate_reach(
    params=params,
    K=K, L=L,
    Q=Q, R=R,
    W=W, V=V,
    ff_weights=ff_weights,
    Phi=Phi,
    perturbation=inta_general_pert,
    rng=np.random.default_rng(2)
)

model_utils.plot_reach(res_inta_general, title="General IntA perturbation")

# simulate a reach with inta rn perturbation
inta_rn_pert = LQGController.Perturbation(
    kind='inta_rn',
    onset_idx=45,
    duration=5,
    pulse=np.array([-2.2, -1.0])
)

res_inta_rn = LQGController.simulate_reach(
    params=params,
    K=K, L=L,
    Q=Q, R=R,
    W=W, V=V,
    ff_weights=ff_weights,
    Phi=Phi,
    perturbation=inta_rn_pert,
    rng=np.random.default_rng(1)
)

model_utils.plot_reach(res_inta_rn, title="IntA→RN perturbation")


# behavior summary metrics
print("Baseline:", model_utils.compute_behavior_metrics(baseline, params.target))
print("IntA→RN:",  model_utils.compute_behavior_metrics(res_inta_rn, params.target))
print("General IntA:",  model_utils.compute_behavior_metrics(res_inta_general, params.target))
print("Mossy:",  model_utils.compute_behavior_metrics(res_mossy, params.target))