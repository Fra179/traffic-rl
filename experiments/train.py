import argparse
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import sumo_rl
import os
from gymnasium import spaces

from traffic_rl.callbacks import TrafficWandbCallback, ValidationCallback, run_baseline
from traffic_rl.rewards import reward_minimize_queue, reward_vidali_waiting_time, reward_minimize_max_queue
from traffic_rl.observations import GridObservationFunction

os.environ["LIBSUMO_AS_TRACI"] = "1"


class PettingZooToGymWrapper(gym.Env):
    """
    Wrapper to convert PettingZoo parallel environment to Gym environment.
    Uses parameter sharing: all agents share the same policy.
    """
    
    def __init__(self, pz_env):
        super().__init__()
        self.pz_env = pz_env
        self.agents = []
        
        # Get a sample agent to determine observation and action spaces
        self.pz_env.reset()
        if self.pz_env.agents:
            sample_agent = self.pz_env.agents[0]
            self.observation_space = self.pz_env.observation_space(sample_agent)
            self.action_space = self.pz_env.action_space(sample_agent)
        
        self.episode_rewards = {}
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the first agent's observation."""
        observations, infos = self.pz_env.reset(seed=seed, options=options)
        self.agents = list(self.pz_env.agents)
        self.episode_rewards = {agent: 0 for agent in self.agents}
        
        if self.agents:
            return observations[self.agents[0]], infos
        return None, infos
    
    def step(self, action):
        """
        Execute action for all agents (parameter sharing).
        """
        if not self.agents:
            return None, 0, True, True, {}
        
        # Use same action for all agents (parameter sharing)
        actions = {agent: action for agent in self.agents}
        
        # Step the PettingZoo environment
        observations, rewards, terminations, truncations, infos = self.pz_env.step(actions)
        
        # Calculate aggregate reward
        total_reward = sum(rewards.values()) if rewards else 0
        avg_reward = total_reward / len(self.agents) if self.agents else 0
        
        # Check if episode is done
        done = all(terminations.values()) if terminations else True
        truncated = all(truncations.values()) if truncations else False
        
        # Return first agent's observation
        next_obs = observations.get(self.agents[0]) if not done else None
        
        info = infos.get(self.agents[0], {}) if infos else {}
        info['total_reward'] = total_reward
        info['avg_reward'] = avg_reward
        
        return next_obs, avg_reward, done, truncated, info
    
    def render(self):
        """Render the environment."""
        return self.pz_env.render()
    
    def close(self):
        """Close the environment."""
        self.pz_env.close()


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
    # Setup file paths
    if args.scenario_dir:
        # Custom scenario directory provided
        BASE_DIR = args.scenario_dir
        NET_FILE = f"{BASE_DIR}/{args.net_file}" if args.net_file else f"{BASE_DIR}/net.xml"
        ROUTE_FILE = f"{BASE_DIR}/{args.train_route_file}" if args.train_route_file else f"{BASE_DIR}/train.rou.xml"
        EVAL_ROUTE_FILE = f"{BASE_DIR}/{args.eval_route_file}" if args.eval_route_file else f"{BASE_DIR}/eval.rou.xml"
    else:
        # Default to cross_dynamic scenario
        NET_FILE = "scenarios/cross_dynamic/cross.net.xml"
        ROUTE_FILE = "scenarios/cross_dynamic/train.rou.xml"
        EVAL_ROUTE_FILE = "scenarios/cross_dynamic/eval.rou.xml"
    
    # Validate algorithm choice
    if args.algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(f"Algorithm must be one of {list(ALGORITHM_CONFIGS.keys())}")
    
    algo_config = ALGORITHM_CONFIGS[args.algorithm]
    
    # Auto-detect episode duration from route files if not specified
    episode_seconds = args.episode_seconds
    eval_episode_seconds = args.eval_episode_seconds
    
    if args.auto_duration:
        print("Auto-detecting episode duration from route files...")
        try:
            import xml.etree.ElementTree as ET
            
            # Get training duration
            train_tree = ET.parse(ROUTE_FILE)
            train_root = train_tree.getroot()
            train_children = list(train_root)
            train_comment = train_children[0] if len(train_children) > 0 else None
            if train_comment is not None and hasattr(train_comment, 'text') and train_comment.text and 'Total Duration:' in train_comment.text:
                episode_seconds = int(train_comment.text.split('Total Duration:')[1].split('s')[0].strip())
                print(f"  Detected training duration: {episode_seconds}s ({episode_seconds/3600:.2f}h)")
            
            # Get eval duration
            eval_tree = ET.parse(EVAL_ROUTE_FILE)
            eval_root = eval_tree.getroot()
            eval_children = list(eval_root)
            eval_comment = eval_children[0] if len(eval_children) > 0 else None
            if eval_comment is not None and hasattr(eval_comment, 'text') and eval_comment.text and 'Total Duration:' in eval_comment.text:
                eval_episode_seconds = int(eval_comment.text.split('Total Duration:')[1].split('s')[0].strip())
                print(f"  Detected eval duration: {eval_episode_seconds}s ({eval_episode_seconds/3600:.2f}h)")
        except Exception as e:
            print(f"  Warning: Could not auto-detect duration ({e}), using provided values")
    
    # 0. Compute Baseline First (skip for multi-agent as it requires single-agent env)
    baseline_metrics = {}
    if not args.multiagent:
        print("Computing baseline metrics...")
        baseline_metrics = run_baseline(NET_FILE, EVAL_ROUTE_FILE, eval_episode_seconds)
    else:
        print("Skipping baseline computation for multi-agent mode (not supported yet)")
    
    print(f"Training episode length: {episode_seconds}s ({episode_seconds/3600:.2f}h)")
    print(f"Evaluation episode length: {eval_episode_seconds}s ({eval_episode_seconds/3600:.2f}h)")
    
    # 1. Setup Training Environment (Multi-agent or Single-agent)
    def make_env():
        if args.multiagent:
            # Use PettingZoo parallel environment for multi-agent
            pz_env = sumo_rl.parallel_env(
                net_file=NET_FILE,
                route_file=ROUTE_FILE,
                out_csv_name=f"outputs/{args.output_prefix}_{args.algorithm}_queue_run" if args.output_prefix else None,
                use_gui=args.gui,
                num_seconds=episode_seconds,
                add_system_info=True,
                reward_fn=reward_minimize_max_queue,
                observation_class=GridObservationFunction
            )
            # Wrap PettingZoo env to make it compatible with SB3
            return PettingZooToGymWrapper(pz_env)
        else:
            # Use standard Gym environment for single-agent
            return gym.make('sumo-rl-v0',
                           net_file=NET_FILE,
                           route_file=ROUTE_FILE,
                           out_csv_name=f"outputs/{args.output_prefix}_{args.algorithm}_queue_run" if args.output_prefix else None,
                           use_gui=args.gui,
                           num_seconds=episode_seconds,
                           add_system_info=True,
                           reward_fn=reward_minimize_max_queue,
                           observation_class=GridObservationFunction)
    
    # Create vectorized environment with support for parallel environments
    if args.n_envs > 1:
        print(f"Creating {args.n_envs} parallel training environments with SubprocVecEnv...")
        env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    else:
        print("Creating single training environment with DummyVecEnv...")
        env = DummyVecEnv([make_env])
    
    # Setup Evaluation Environment (Single separate instance, no GUI)
    print("Creating evaluation environment...")
    if args.multiagent:
        pz_eval_env = sumo_rl.parallel_env(
            net_file=NET_FILE,
            route_file=EVAL_ROUTE_FILE,
            use_gui=False,
            num_seconds=eval_episode_seconds,
            add_system_info=True,
            reward_fn=reward_minimize_max_queue,
            observation_class=GridObservationFunction,
            sumo_seed='42'
        )
        eval_env = PettingZooToGymWrapper(pz_eval_env)
    else:
        eval_env = gym.make('sumo-rl-v0',
                            net_file=NET_FILE,
                            route_file=EVAL_ROUTE_FILE,
                            use_gui=False,
                            num_seconds=eval_episode_seconds,
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
    
    mode_str = "MULTI-AGENT" if args.multiagent else "SINGLE-AGENT"
    print(f"\nStarting {args.algorithm.upper()} training in {mode_str} mode...")
    print(f"  Algorithm: {args.algorithm.upper()}")
    print(f"  Number of parallel environments: {args.n_envs}")
    print(f"  Total timesteps: {args.total_timesteps}")
    
    # 3. Initialize Wandb
    run_name = f"{args.algorithm}-{args.run_name}" if args.run_name else f"{args.algorithm}-queue"
    if args.multiagent:
        run_name = f"ma-{run_name}"  # Prefix with 'ma' for multi-agent
    
    wandb.init(
        entity="fds-final-project",
        project="rl-traffic-light",
        name=run_name,
        config={
            "algorithm": args.algorithm,
            "multiagent": args.multiagent,
            "parameter_sharing": args.multiagent,  # Currently always True for multi-agent
            "n_envs": args.n_envs,
            "scenario_dir": args.scenario_dir if args.scenario_dir else "scenarios/cross_dynamic",
            "net_file": NET_FILE,
            "route_file": ROUTE_FILE,
            "eval_route_file": EVAL_ROUTE_FILE,
            "episode_seconds": episode_seconds,
            "eval_episode_seconds": eval_episode_seconds,
            "total_timesteps": args.total_timesteps,
            "normalize": args.normalize,
            "baseline_metrics": baseline_metrics,
            **hyperparams
        }
    )
    
    # Calculate eval_freq based on delta_time (default is 5s)
    STEPS_PER_EPISODE = episode_seconds // 5
    
    # 4. Setup callbacks
    # Setup path for best model
    model_name = f"{args.output_prefix}_{args.algorithm}_model" if args.output_prefix else f"{args.algorithm}_model"
    if args.multiagent:
        model_name = f"{model_name}_multiagent"
    best_model_path = f"weights/{model_name}_best"
    os.makedirs("weights", exist_ok=True)
    
    callbacks = [
        TrafficWandbCallback(),
        ValidationCallback(eval_env, baseline_metrics, eval_freq=STEPS_PER_EPISODE),
        EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=f"logs/{model_name}",
            eval_freq=STEPS_PER_EPISODE,
            deterministic=True,
            render=False,
            verbose=1,
            n_eval_episodes=1  # One full episode per eval (already long duration)
        )
    ]
    
    # 5. Train
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
    finally:
        # Save final model (for comparison with best)
        final_model_path = f"weights/{model_name}_final"
        model.save(final_model_path)
        print(f"\nTraining complete!")
        print(f"  Final model saved to {final_model_path}")
        print(f"  Best model saved to {best_model_path}/best_model.zip")
        
        # Save normalization stats if used
        if args.normalize:
            norm_name = f"{args.output_prefix}_{args.algorithm}_vec_normalize.pkl" if args.output_prefix else f"{args.algorithm}_vec_normalize.pkl"
            if args.multiagent:
                norm_name = f"multiagent_{norm_name}"
            env.save(f"weights/{norm_name}")
            print(f"  Normalization stats saved to weights/{norm_name}")
        
        # Cleanup
        eval_env.close()
        env.close()
        wandb.finish()
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train traffic light controller with various RL algorithms (single or multi-agent)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Multi-agent configuration
    parser.add_argument(
        "--multiagent",
        action="store_true",
        help="Enable multi-agent training using PettingZoo parallel environment with parameter sharing"
    )
    
    # Parallel training configuration
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel training environments (use SubprocVecEnv if > 1)"
    )
    
    # Scenario configuration
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Path to scenario directory"
    )
    
    parser.add_argument(
        "--net-file",
        type=str,
        default=None,
        help="Network file name relative to scenario-dir"
    )
    
    parser.add_argument(
        "--train-route-file",
        type=str,
        default=None,
        help="Training route file name relative to scenario-dir"
    )
    
    parser.add_argument(
        "--eval-route-file",
        type=str,
        default=None,
        help="Evaluation route file name relative to scenario-dir"
    )
    
    parser.add_argument(
        "--auto-duration",
        action="store_true",
        help="Auto-detect episode duration from route file comments"
    )
    
    # Algorithm configuration
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        required=True,
        choices=["dqn", "ppo", "a2c"],
        help="RL algorithm to use"
    )
    
    # Episode configuration
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=16200,
        help="Duration of each training episode in seconds (ignored if --auto-duration)"
    )
    
    parser.add_argument(
        "--eval-episode-seconds",
        type=int,
        default=5400,
        help="Duration of each evaluation episode in seconds (ignored if --auto-duration)"
    )
    
    # Training configuration
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
    
    # Output configuration
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output CSV files"
    )
    
    # Visualization
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI visualization"
    )
    
    # Wandb configuration
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (for Wandb)"
    )
    
    args = parser.parse_args()
    main(args)
