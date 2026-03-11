import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from __init__ import *
from utils import *
from env.so100_tracking_env import SO100TrackEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on SO100 tracking")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel environments; set 1 for single process")
    parser.add_argument("--max_iterations", type=int, default=500,
                        help="Number of PPO update iterations")
    parser.add_argument("--save_checkpt_freq", type=int, default=50,
                        help="Checkpoint every N update iterations")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda)")
    parser.add_argument("--load_run", type=str, default=None,
                        help="Run ID to resume from (e.g. '1')")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load (e.g. '500')")
    return parser.parse_args()

def make_env():
    def _init():
        env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
        env = Monitor(env, info_keywords=("ee_tracking_error",))
        return env
    return _init

if __name__ == "__main__":
    args = parse_args()

    if args.num_envs > 1:
        # Wrap in a vectorized environment for parallel simulation
        start_method = "spawn"
        envs = SubprocVecEnv([make_env() for _ in range(args.num_envs)], start_method=start_method)
        envs = VecMonitor(envs)
        print(f"Successfully launched {args.num_envs} environments!")
        vec_env = envs
    else:
        # Create a single environment for debug with rendering
        vec_env = SO100TrackEnv(xml_path=XML_PATH, render_mode="human")

    if args.load_run is not None and args.checkpoint is not None:
        policy_path = EXP_DIR / f"so100_tracking_{args.load_run}" / f"model_{args.checkpoint}.zip"
        print(f"Resuming from {policy_path}...")
        model = PPO.load(policy_path, env=vec_env, device=args.device,
                         tensorboard_log=EXP_DIR)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            device=args.device,
            verbose=1,
            tensorboard_log=EXP_DIR,
            gamma=0.99,
            ent_coef=0.001,
            vf_coef=1.0
        )

    update_checkpoint_callback = UpdateCheckpointCallback(
        save_path=None,
        name_prefix="model",
        save_freq_updates=args.save_checkpt_freq,
        verbose=0,
    )

    # Train for a target number of PPO update steps (each update = n_steps * n_envs transitions)
    total_update_steps = args.max_iterations
    rollout_batch_size = model.n_steps * model.n_envs
    total_timesteps = total_update_steps * rollout_batch_size
    print(f"Training for {total_update_steps} iterations (~{total_timesteps} steps, rollout size {rollout_batch_size})")

    envs_ref = model.get_env()
    try:
        model.learn(total_timesteps=total_timesteps,
                    log_interval=1,
                    tb_log_name=EXP_NAME,
                    callback=[EpisodeLoggingCallback(),
                              update_checkpoint_callback,
                              KLAdaptiveLRCallback(target_kl=0.05, init_lr=1e-3, min_lr=1e-5, max_lr=1e-3)]
                    )
    finally:
        # Ensure environments cleanly close GL contexts
        if envs_ref is not None:
            envs_ref.close()

    # Save the final model
    log_dir = model.logger.get_dir()
    if log_dir is None:
        raise ValueError("Logger directory is not set; cannot save final model")
    final_update_step = update_checkpoint_callback.update_counter
    final_model_path = Path(log_dir) / f"model_{final_update_step}"
    model.save(str(final_model_path))
    print(f"Saved models to {log_dir}")