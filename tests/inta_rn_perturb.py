from model.lqg_control_model import (
    LQGController,
    LQGParams,
    Perturbation,
)

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