""" Tests the baseline LQG control model with a simple 2D point mass system. """

import numpy as np
from scipy.linalg import solve_discrete_are
from lqg_control_model import LQGController

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