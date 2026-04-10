import torch
import numpy as np
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from scripts.train_dqn import main as main_dqn
from scripts.eval_dqn import main as main_eval_dqn
from exercises.ex2_dqn_config import DQN_PARAMETERS
from itertools import product


# "lr": 2e-4,            # TODO
# "epsilon": 0.35,       # TODO
# "target_update": 100,   # TODO
# "hidden_dim": 96,     # TODO


split = 5
lr_range = np.linspace(1e-4, 7e-4, split)
epsilon_range = np.linspace(0.10, 0.4, split)
target_update_range = np.linspace(30, 150, split)
hidden_dim_range = np.linspace(96, 196, split)

# Create all combinations of parameters
param_combinations = list(product(
    lr_range,
    epsilon_range,
    target_update_range,
    hidden_dim_range
))

print(f"Total combinations: {len(param_combinations)}")


def run(args):
    torch.set_num_threads(1)
    i, lr, epsilon, target_update, hidden_dim = args
    print(f"Combination {i+1}/{len(param_combinations)}: lr={lr:.6f}, epsilon={epsilon:.4f}, target_update={int(target_update)}, hidden_dim={int(hidden_dim)}")
    DQN_PARAMETERS_copy = DQN_PARAMETERS.copy()
    DQN_PARAMETERS_copy["lr"] = float(lr)
    DQN_PARAMETERS_copy["epsilon"] = float(epsilon)
    DQN_PARAMETERS_copy["target_update"] = int(target_update)
    DQN_PARAMETERS_copy["hidden_dim"] = int(hidden_dim)
    success = main_eval_dqn(main_dqn(DQN_PARAMETERS_copy))
    return [success, lr, epsilon, target_update, hidden_dim]


if __name__ == "__main__":
    args_list = [(i, lr, epsilon, target_update, hidden_dim)
                 for i, (lr, epsilon, target_update, hidden_dim) in enumerate(param_combinations)]

    n_workers = min(cpu_count(), len(param_combinations), 8)
    print(f"Running with {n_workers} processes...")

    with Pool(processes=n_workers) as pool:
        results = pool.map(run, args_list)

    success_and_params = np.array(results)

    # Sort by success (first column) in descending order
    sorted_indices = np.argsort(success_and_params[:, 0])[::-1]
    sorted_results = success_and_params[sorted_indices]

    # Save to text file
    output_path = ROOT_DIR / "dqn_parameter_search_results.txt"
    with open(output_path, "w") as f:
        f.write("Success | LR | Epsilon | Target Update | Hidden Dim\n")
        f.write("-" * 60 + "\n")
        for row in sorted_results:
            f.write(f"{row[0]:.4f} | {row[1]:.6f} | {row[2]:.4f} | {int(row[3])} | {int(row[4])}\n")

    print(f"\nResults saved to {output_path}")
    print("\nTop 5 parameter combinations:")
    print("Success | LR | Epsilon | Target Update | Hidden Dim")
    print("-" * 60)
    for row in sorted_results[:5]:
        print(f"{row[0]:.4f} | {row[1]:.6f} | {row[2]:.4f} | {int(row[3])} | {int(row[4])}")
