import argparse
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl
import os
import pandas as pd
from pathlib import Path

from traffic_rl.callbacks import run_baseline
from traffic_rl.rewards import reward_minimize_queue
from traffic_rl.observations import GridObservationFunction

os.environ["LIBSUMO_AS_TRACI"] = "1"


def evaluate_model(model, env, n_episodes=10, render=False):
    """
    Evaluate a trained model on the environment.
    
    Args:
        model: Trained RL model
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment (GUI)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_metrics = {
        'system_mean_waiting_time': [],
        'system_total_stopped': [],
        'system_mean_speed': []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Extract system-level metrics from final info
        if 'system_mean_waiting_time' in info:
            episode_metrics['system_mean_waiting_time'].append(info['system_mean_waiting_time'])
        if 'system_total_stopped' in info:
            episode_metrics['system_total_stopped'].append(info['system_total_stopped'])
        if 'system_mean_speed' in info:
            episode_metrics['system_mean_speed'].append(info['system_mean_speed'])
        
        print(f"Episode {episode + 1}/{n_episodes} - Reward: {episode_reward:.2f}, Length: {episode_length}")
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_waiting_time': np.mean(episode_metrics['system_mean_waiting_time']) if episode_metrics['system_mean_waiting_time'] else None,
        'mean_total_stopped': np.mean(episode_metrics['system_total_stopped']) if episode_metrics['system_total_stopped'] else None,
        'mean_speed': np.mean(episode_metrics['system_mean_speed']) if episode_metrics['system_mean_speed'] else None,
    }
    
    return results, episode_rewards, episode_metrics


def main(args):
    NET_FILE = "scenarios/cross_dynamic/cross_turn.net.xml"
    ROUTE_FILE = "scenarios/cross_dynamic/eval.rou.xml"
    
    # Check if model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    print(f"Loading model from: {args.model_path}")
    print(f"Algorithm: {args.algorithm.upper()}")
    
    # Load the model
    if args.algorithm == "dqn":
        model = DQN.load(args.model_path)
    elif args.algorithm == "ppo":
        model = PPO.load(args.model_path)
    elif args.algorithm == "a2c":
        model = A2C.load(args.model_path)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Create evaluation environment
    env = gym.make('sumo-rl-v0',
                   net_file=NET_FILE,
                   route_file=ROUTE_FILE,
                   use_gui=args.gui,
                   num_seconds=args.episode_seconds,
                   add_system_info=True,
                   reward_fn=reward_minimize_queue,
                   observation_class=GridObservationFunction,
                   sumo_seed=args.seed if args.seed else 'random')
    
    # Load VecNormalize stats if they exist
    if args.normalize:
        normalize_path = Path(args.normalize)
        if normalize_path.exists():
            print(f"Loading VecNormalize stats from: {args.normalize}")
            # Wrap environment for normalization
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(args.normalize, env)
            env.training = False  # Don't update stats during evaluation
            env.norm_reward = False
        else:
            print(f"Warning: VecNormalize file not found: {args.normalize}")
    
    # Compute baseline if requested
    baseline_metrics = None
    if args.compare_baseline:
        print("\nComputing baseline metrics...")
        baseline_metrics = run_baseline(NET_FILE, ROUTE_FILE, args.episode_seconds)
        print(f"Baseline - Mean Waiting Time: {baseline_metrics['mean_waiting_time']:.2f}s")
        print(f"Baseline - Mean Stopped Vehicles: {baseline_metrics['mean_stopped']:.2f}")
        print(f"Baseline - Mean Speed: {baseline_metrics['mean_speed']:.2f} m/s\n")
    
    # Initialize Wandb if requested
    if args.use_wandb:
        wandb.init(
            entity="fds-final-project",
            project="rl-traffic-light",
            name=f"eval-{args.algorithm}-{args.run_name}" if args.run_name else f"eval-{args.algorithm}",
            config={
                "algorithm": args.algorithm,
                "model_path": str(args.model_path),
                "n_eval_episodes": args.n_episodes,
                "episode_seconds": args.episode_seconds,
                "seed": args.seed,
            }
        )
    
    # Evaluate the model
    print(f"\nEvaluating model for {args.n_episodes} episodes...")
    results, episode_rewards, episode_metrics = evaluate_model(
        model, env, n_episodes=args.n_episodes, render=args.gui
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} steps")
    
    if results['mean_waiting_time'] is not None:
        print(f"\nTraffic Metrics:")
        print(f"  Mean Waiting Time: {results['mean_waiting_time']:.2f}s")
        print(f"  Mean Total Stopped: {results['mean_total_stopped']:.2f}")
        print(f"  Mean Speed: {results['mean_speed']:.2f} m/s")
        
        # Compare with baseline
        if baseline_metrics:
            print(f"\nComparison with Baseline:")
            waiting_improvement = ((baseline_metrics['mean_waiting_time'] - results['mean_waiting_time']) 
                                   / baseline_metrics['mean_waiting_time'] * 100)
            stopped_improvement = ((baseline_metrics['mean_stopped'] - results['mean_total_stopped']) 
                                   / baseline_metrics['mean_stopped'] * 100)
            speed_improvement = ((results['mean_speed'] - baseline_metrics['mean_speed']) 
                                / baseline_metrics['mean_speed'] * 100)
            
            print(f"  Waiting Time: {waiting_improvement:+.1f}%")
            print(f"  Stopped Vehicles: {stopped_improvement:+.1f}%")
            print(f"  Speed: {speed_improvement:+.1f}%")
    
    print("="*60 + "\n")
    
    # Log to Wandb
    if args.use_wandb:
        wandb_log = {
            "eval/mean_reward": results['mean_reward'],
            "eval/std_reward": results['std_reward'],
            "eval/mean_episode_length": results['mean_episode_length'],
        }
        
        if results['mean_waiting_time'] is not None:
            wandb_log.update({
                "eval/mean_waiting_time": results['mean_waiting_time'],
                "eval/mean_total_stopped": results['mean_total_stopped'],
                "eval/mean_speed": results['mean_speed'],
            })
        
        if baseline_metrics:
            wandb_log.update({
                "eval/baseline_waiting_time": baseline_metrics['mean_waiting_time'],
                "eval/baseline_stopped": baseline_metrics['mean_stopped'],
                "eval/baseline_speed": baseline_metrics['mean_speed'],
                "eval/waiting_improvement_pct": waiting_improvement,
                "eval/stopped_improvement_pct": stopped_improvement,
                "eval/speed_improvement_pct": speed_improvement,
            })
        
        wandb.log(wandb_log)
    
    # Save results to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        df = pd.DataFrame({
            'episode': range(1, args.n_episodes + 1),
            'reward': episode_rewards,
            'waiting_time': episode_metrics['system_mean_waiting_time'],
            'total_stopped': episode_metrics['system_total_stopped'],
            'mean_speed': episode_metrics['system_mean_speed'],
        })
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    # Cleanup
    env.close()
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained RL traffic light controller models"
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the pretrained model (.zip file)"
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        required=True,
        choices=["dqn", "ppo", "a2c"],
        help="Algorithm used to train the model"
    )
    
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=3600,
        help="Duration of each episode in seconds"
    )
    
    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats file (.pkl)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for evaluation (for reproducibility)"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI visualization"
    )
    
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare results with baseline (fixed time traffic light)"
    )
    
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log evaluation results to Weights & Biases"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this evaluation run (for Wandb)"
    )
    
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save evaluation results as CSV"
    )
    
    args = parser.parse_args()
    main(args)
