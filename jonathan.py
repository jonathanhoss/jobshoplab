from pathlib import Path

import numpy as np
import torch
from jssp_core.instances import FT06_INSTANCE, parse_instance
from jssp_gnn.train_gnn_matrix_form import GnnMatrixSolver

from jobshoplab import JobShopLabEnv, load_config

gnn_solver = GnnMatrixSolver(Path("./tmp/models/13-27-57/checkpoints/best_model.pt"))
gnn_solver.load_config_from_yaml(Path("./tmp/models/13-27-57/checkpoints/config.yaml"))
gnn_solver_instance = parse_instance(FT06_INSTANCE)
gnn_solver_env = gnn_solver._create_env(gnn_solver_instance)
gnn_solver._load_model(gnn_solver_env)

# Load a pre-defined configuration
config = load_config(config_path=Path("./data/config/custom_config.yaml"))

# Create the environment
env = JobShopLabEnv(config=config)


def obs_processor(obs, solver_td):
    obs_solver_td = solver_td.get("next", solver_td)
    return {
        "edge_index": obs["edge_index"],
        "node_feats": obs["node_feats"],
        "mask": obs_solver_td["mask"],
    }


print("Starting simulation...")
# Run with random actions until done
done = False
action = 0
obs, info = env.reset()
solver_td = gnn_solver_env.reset()

steps = 0
while not done and steps <= 100:
    # Get action from GNN solver
    solver_action = gnn_solver.get_action(obs_processor(obs, solver_td))
    action = int(solver_action) + 1  # Adjust offset

    # Check if action is valid before executing
    success, obs, reward, terminated, truncated, info = env.simulate_step(action)

    if success:
        # Execute the valid action
        obs, reward, truncated, terminated, info = env.step(action)

        # Update the solver environment

        solver_td["action"] = torch.tensor([solver_action], dtype=torch.int32)
        solver_td = gnn_solver_env.step(solver_td)
    else:
        # Invalid action: take no-op instead
        obs, reward, truncated, terminated, info = env.step(0)
        print(f"Invalid action at step {steps}, took no-op")

    done = truncated or terminated
    steps += 1
    print(f"Step {steps}: Reward={reward}, Done={done}")
# Visualize the final schedule
env.render()
