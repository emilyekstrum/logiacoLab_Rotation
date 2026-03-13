""" Tests the LQG control model with an intA/RN perturbation. """

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.lqg_control_model import (
    LQGController,
    LQGParams,
    Perturbation,
) 

from model_utils import plot_reach, compute_behavior_metrics

params = LQGParams()
controller = LQGController(params)

Phi = controller.make_default_basis(n_basis=8, width=10.0)
ff_weights = controller.make_default_ff_weights(n_basis=8)

pert = Perturbation(kind="inta_rn", onset_idx=45, duration=5)

result = controller.simulate_reach(
    Phi=Phi,
    ff_weights=ff_weights,
    perturbation=pert,
)

print(result["x"].shape)
print(result["J"])

plot_reach(result, title="IntA/RN perturbation")
compute_behavior_metrics(result, target=params.target)