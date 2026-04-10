"""
Hyperparameters for Exercise 2 (DQN).

You are encouraged to tune:
- lr
- epsilon
- target_update
- hidden_dim

Please keep the remaining parameters unchanged unless explicitly stated.
"""

DQN_PARAMETERS = {
    # TODO: Tune the following hyperparameters
    # Replace the default values with your own choices.
    "lr": 7e-4,            # TODO
    "epsilon": 0.4,       # TODO
    "target_update": 30,   # TODO
    "hidden_dim": 146,     # TODO
    
    # Fixed parameters
    "gamma": 0.99,
    "num_episodes": 500,
    "buffer_size": 10000,
    "minimal_size": 500,
    "batch_size": 64,
    "seed": 0,
}