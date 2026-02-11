import argparse
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl
import os

from traffic_rl.callbacks import TrafficWandbCallback, ValidationCallback, run_baseline
from traffic_rl.rewards import reward_minimize_queue, reward_vidali_waiting_time, reward_minimize_max_queue
from traffic_rl.observations import GridObservationFunction

os.environ["LIBSUMO_AS_TRACI"] = "1"

# Algorithm configurations
# Tuned for fair comparison across algorithms
# Episode: 3600s, delta_time: 5s = 720 steps/episode
ALGORITHM_CONFIGS = {
    "dqn": {
        "class": DQN,
        "hyperparams": {
            "learning_rate": 0.0005,        # Standard for DQN
            "buffer_size": 50000,            # ~70 episodes of experience
            "learning_starts": 1000,         # ~1.4 episodes before learning
            "batch_size": 64,                # Standard batch size
            "target_update_interval": 500,   # ~0.7 episodes
            "train_freq": 4,                 # Update every 4 steps
            "gradient_steps": 1,             # 1 gradient step per update
            "exploration_fraction": 0.3,     # Explore for 30% of training
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "gamma": 0.99,                   # Discount factor
        }
    },
    "ppo": {
        "class": PPO,
        "hyperparams": {
            "learning_rate": 0.0003,         # Standard for PPO
            "n_steps": 2048,                 # ~2.8 episodes of data per update
            "batch_size": 64,                # Minibatch size for updates
            "n_epochs": 10,                  # 10 passes through collected data
            "gamma": 0.99,                   # Discount factor
            "gae_lambda": 0.95,              # GAE parameter for advantage estimation
            "clip_range": 0.2,               # PPO clipping parameter
            "ent_coef": 0.01,                # Entropy bonus for exploration
            "vf_coef": 0.5,                  # Value function coefficient
            "max_grad_norm": 0.5,            # Gradient clipping
        }
    },
    "a2c": {
        "class": A2C,
        "hyperparams": {
            "learning_rate": 0.0005,         # Matched to DQN for comparison
            "n_steps": 256,                  # ~0.36 episodes between updates (balanced)
            "gamma": 0.99,                   # Discount factor
            "gae_lambda": 0.95,              # GAE parameter (match PPO)
            "ent_coef": 0.01,                # Entropy bonus (match PPO)
            "vf_coef": 0.5,                  # Value function coefficient
            "max_grad_norm": 0.5,            # Gradient clipping
            "normalize_advantage": True,     # Stabilizes training
        }
    }
}


def main(args):
    NET_FILE = "scenarios/cross_dynamic/cross.net.xml"
    ROUTE_FILE = "scenarios/cross_dynamic/train.rou.xml"
    EVAL_ROUTE_FILE = "scenarios/cross_dynamic/eval.rou.xml"
    
    # Validate algorithm choice
    if args.algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(f"Algorithm must be one of {list(ALGORITHM_CONFIGS.keys())}")
    
    algo_config = ALGORITHM_CONFIGS[args.algorithm]
    
    # 0. Compute Baseline First
    print("Computing baseline metrics...")
    baseline_metrics = run_baseline(NET_FILE, EVAL_ROUTE_FILE, args.eval_episode_seconds)
    print(f"Training episode length: {args.episode_seconds}s ({args.episode_seconds/3600:.2f}h)")
    print(f"Evaluation episode length: {args.eval_episode_seconds}s ({args.eval_episode_seconds/3600:.2f}h)")
    
    # 1. Setup Training Environment
    def make_env():
        return gym.make('sumo-rl-v0',
                       net_file=NET_FILE,
                       route_file=ROUTE_FILE,
                       out_csv_name=f"outputs/{args.algorithm}_queue_run",
                       use_gui=args.gui,
                       num_seconds=args.episode_seconds,
                       add_system_info=True,
                       reward_fn=reward_minimize_max_queue,
                       observation_class=GridObservationFunction)
    
    env = DummyVecEnv([make_env])
    
    # Setup Evaluation Environment (Separate instance)
    eval_env = gym.make('sumo-rl-v0',
                        net_file=NET_FILE,
                        route_file=EVAL_ROUTE_FILE,
                        use_gui=False,
                        num_seconds=args.eval_episode_seconds,
                        add_system_info=True,
                        reward_fn=reward_minimize_max_queue,
                        observation_class=GridObservationFunction,
                        sumo_seed='42')
    
    # Optional: Apply normalization
    if args.normalize:
        print("Applying observation normalization...")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # 2. Create the agent
    AlgorithmClass = algo_config["class"]
    hyperparams = algo_config["hyperparams"].copy()
    
    # Allow command-line override of learning rate
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    
    model = AlgorithmClass(
        "MlpPolicy",
        env,
        verbose=1,
        **hyperparams
    )
    
    print(f"Starting {args.algorithm.upper()} training...")
    
    # 3. Initialize Wandb
    wandb.init(
        entity="fds-final-project",
        project="rl-traffic-light",
        name=f"{args.algorithm}-cross-{args.run_name}" if args.run_name else f"{args.algorithm}-cross-queue",
        config={
            "algorithm": args.algorithm,
            "net_file": NET_FILE,
            "route_file": ROUTE_FILE,
            "eval_route_file": EVAL_ROUTE_FILE,
            "episode_seconds": args.episode_seconds,
            "eval_episode_seconds": args.eval_episode_seconds,
            "total_timesteps": args.total_timesteps,
            "normalize": args.normalize,
            "baseline_metrics": baseline_metrics,
            **hyperparams
        }
    )
    
    # Calculate eval_freq based on delta_time (default is 5s)
    STEPS_PER_EPISODE = args.episode_seconds // 5
    
    # 4. Setup callbacks
    callbacks = [
        TrafficWandbCallback(),
        ValidationCallback(eval_env, baseline_metrics, eval_freq=STEPS_PER_EPISODE)
    ]
    
    # 5. Train
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    finally:
        # Save model
        model_path = f"{args.algorithm}_model"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save normalization stats if used
        if args.normalize:
            env.save(f"{args.algorithm}_vec_normalize.pkl")
        
        # Cleanup
        eval_env.close()
        env.close()
        wandb.finish()
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train traffic light controller with various RL algorithms from Stable Baselines3"
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        required=True,
        choices=["dqn", "ppo", "a2c"],
        help="RL algorithm to use"
    )
    
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=16200,
        help="Duration of each training episode in seconds (default: 16200s = 4.5 hours)"
    )
    
    parser.add_argument(
        "--eval-episode-seconds",
        type=int,
        default=5400,
        help="Duration of each evaluation episode in seconds (default: 5400s = 1.5 hours)"
    )
    
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,
        help="Total training timesteps"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides default for selected algorithm)"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply observation normalization (VecNormalize)"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI visualization"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (for Wandb)"
    )
    
    args = parser.parse_args()
    main(args)
