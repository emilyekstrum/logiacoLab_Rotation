""" Tests the baseline LQG control model with a simple 2D point mass system. """

import sys
from pathlib import Path

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.linalg import solve_discrete_are
from model.lqg_control_model import LQGController, LQGParams, Perturbation
import model_utils

params = LQGParams()
controller = LQGController(params)

Phi = controller.make_default_basis(n_basis=8, width=10.0)
ff_weights = controller.make_default_ff_weights(n_basis=8)

baseline = controller.simulate_reach(
    Phi=Phi,
    ff_weights=ff_weights,
    perturbation=None,
    rng=np.random.default_rng(0)
)

# plot baseline reach
model_utils.plot_reach(baseline, title="Baseline LQG Control Reach")

# # simulate a reach with a mossy fiber perturbation
# mossy_pert = Perturbation(
#     kind='mossy',
#     onset_idx=45,
#     duration=5,
#     observer_bias=np.array([0.0, 0.0, 0.9, 0.5])
# )

# res_mossy = controller.simulate_reach(
#     Phi=Phi,
#     ff_weights=ff_weights,
#     perturbation=mossy_pert,
#     rng=np.random.default_rng(3)
# )

# model_utils.plot_reach(res_mossy, title="Mossy fiber perturbation")

# # simulate a reach with a inta general perturbation
# inta_general_pert = Perturbation(
#     kind='inta_general',
#     onset_idx=45,
#     duration=5,
#     pulse=np.array([-2.2, -1.0])
# )

# res_inta_general = controller.simulate_reach(
#     Phi=Phi,
#     ff_weights=ff_weights,
#     perturbation=inta_general_pert,
#     rng=np.random.default_rng(2)
# )

# model_utils.plot_reach(res_inta_general, title="General IntA perturbation")

# # simulate a reach with inta rn perturbation
# inta_rn_pert = Perturbation(
#     kind='inta_rn',
#     onset_idx=45,
#     duration=5,
#     pulse=np.array([-2.2, -1.0])
# )

# res_inta_rn = controller.simulate_reach(
#     Phi=Phi,
#     ff_weights=ff_weights,
#     perturbation=inta_rn_pert,
#     rng=np.random.default_rng(1)
# )

# model_utils.plot_reach(res_inta_rn, title="IntA→RN perturbation")


# behavior summary metrics
print("Baseline:", model_utils.compute_behavior_metrics(baseline, params.target))
# print("IntA→RN:",  model_utils.compute_behavior_metrics(res_inta_rn, params.target))
# print("General IntA:",  model_utils.compute_behavior_metrics(res_inta_general, params.target))
# print("Mossy:",  model_utils.compute_behavior_metrics(res_mossy, params.target))
