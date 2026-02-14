#!/usr/bin/env python3
"""
Evaluate multiple independently trained agents on a multi-agent environment.
Each agent uses its own trained policy (heterogeneous multi-agent).
"""
import argparse
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN, PPO, A2C
from pathlib import Path
import sumo_rl
import os

from traffic_rl.callbacks import run_baseline
from traffic_rl.rewards import reward_minimize_queue, reward_minimize_max_queue
from traffic_rl.observations import GridObservationFunction

os.environ["LIBSUMO_AS_TRACI"] = "1"


# Mapping from intersection letters to traffic light IDs in the full network
INTERSECTION_TO_TL_ID = {
    'A': 'cluster_29784567_310818818',
    'B': 'cluster_12614600_1860618754_1860618762_1860618778_1860618795_1860618797_1860618805_1860618824_1860618825_29270276_29270294_29270295_3649758678_3651880147_3657149480_3657636381',
    'C': 'cluster_28150269_29271707_4377814009_4377814010_4377814011_4377814012_4377814013_4377814014_4377814015_4377814016_4377814017',
    'D': 'cluster_12614653_1882193216_1882193222_29271706_4377814000_4377814001_4377814002_4377814003_4377814004_4377814005_4377814006_4377814007',
    'E': 'cluster_29271704_29271705_6911007220_6911007221_6911007222_6911007223_6911007224_6911007225_6911007226_6911007227_6911007228_6911007229',
    'F': 'cluster_1860618835_1860618840_1860618845_1860618847_29784556_29784557',
    'G': 'cluster_12614600_1860618754_1860618762_1860618778_1860618795_1860618797_1860618805_1860618824_1860618825_29270276_29270294_29270295_3649758678_3651880147_3657149480_3657636381',  # Same as B
    'H': 'cluster_29784655_5960120416_5960120417_5960120418_5960120419_5960120420_5960120421_5960120422_5960120423',
    'I': '283089493',
    'J': 'cluster_29271680_494154248_5960119458_5960119459_5960119460_5960119461_5960119462_5960119463_5960119464',
}


class HeterogeneousMultiAgentWrapper(gym.Env):
    """
    Wrapper for heterogeneous multi-agent evaluation.
    Each agent uses its own trained policy (no parameter sharing).
    """
    
    def __init__(self, pz_env, agent_models_map):
        """
        Args:
            pz_env: PettingZoo parallel environment
            agent_models_map: Dict mapping agent_id -> trained model
        """
        super().__init__()
        self.pz_env = pz_env
        self.agent_models_map = agent_models_map
        self.agents = []
        self.current_observations = {}
        
        # Get a sample agent to determine observation and action spaces
        self.pz_env.reset()
        if self.pz_env.agents:
            sample_agent = self.pz_env.agents[0]
            self.observation_space = self.pz_env.observation_space(sample_agent)
            self.action_space = self.pz_env.action_space(sample_agent)
        
        # Track which agents have loaded models
        available_agents = set(agent_models_map.keys())
        print(f"[HeterogeneousMultiAgentWrapper] Available trained agents: {available_agents}")
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observations."""
        observations, infos = self.pz_env.reset(seed=seed, options=options)
        self.agents = list(self.pz_env.agents)
        self.current_observations = observations
        
        # Check which agents have trained models
        agents_with_models = [a for a in self.agents if a in self.agent_models_map]
        agents_without_models = [a for a in self.agents if a not in self.agent_models_map]
        
        print(f"[Reset] Total agents: {len(self.agents)}")
        print(f"[Reset] Agents with trained models: {len(agents_with_models)}")
        if agents_without_models:
            print(f"[Reset] WARNING: Agents without trained models: {agents_without_models[:5]}...")
        
        # Return first agent's observation for compatibility
        if self.agents:
            return observations[self.agents[0]], infos
        return None, infos
    
    def step(self, action=None):
        """
        Execute actions for all agents using their respective policies.
        The 'action' parameter is ignored - we compute actions from models.
        """
        if not self.agents:
            return None, 0, True, True, {}
        
        # Use observations from previous step/reset
        observations = self.current_observations
        
        # Compute actions for each agent using its trained model
        actions = {}
        for agent in self.agents:
            if agent in self.agent_models_map:
                obs = observations.get(agent)
                if obs is not None:
                    action, _ = self.agent_models_map[agent].predict(obs, deterministic=True)
                    actions[agent] = action
                else:
                    # No observation - use default action
                    actions[agent] = 0
            else:
                # No trained model - use default action (e.g., always green on phase 0)
                actions[agent] = 0
        
        # Step the PettingZoo environment
        observations, rewards, terminations, truncations, infos = self.pz_env.step(actions)
        
        # Store observations for next step
        self.current_observations = observations
        
        # Calculate aggregate reward
        total_reward = sum(rewards.values()) if rewards else 0
        avg_reward = total_reward / len(self.agents) if self.agents else 0
        
        # Check if episode is done
        done = all(terminations.values()) if terminations else True
        truncated = all(truncations.values()) if truncations else False
        
        # Return first agent's observation
        next_obs = observations.get(self.agents[0]) if not done else None
        
        # Aggregate info
        info = {}
        if infos:
            # Try to get system-wide metrics from any agent's info
            sample_info = next(iter(infos.values()))
            if 'system_mean_waiting_time' in sample_info:
                info['system_mean_waiting_time'] = sample_info['system_mean_waiting_time']
            if 'system_total_stopped' in sample_info:
                info['system_total_stopped'] = sample_info['system_total_stopped']
            if 'system_mean_speed' in sample_info:
                info['system_mean_speed'] = sample_info['system_mean_speed']
        
        info['total_reward'] = total_reward
        info['avg_reward'] = avg_reward
        
        return next_obs, avg_reward, done, truncated, info
    
    def render(self):
        """Render the environment."""
        return self.pz_env.render()
    
    def close(self):
        """Close the environment."""
        self.pz_env.close()


def load_models(intersections, algorithm, model_dir="weights", prefix="berlin"):
    """
    Load trained models for specified intersections.
    
    Args:
        intersections: List of intersection letters (e.g., ['A', 'B', 'C'])
        algorithm: RL algorithm used ('dqn', 'ppo', 'a2c')
        model_dir: Directory containing model weights
        prefix: Prefix for model file names
        
    Returns:
        Dict mapping traffic_light_id -> loaded model
    """
    model_class = {"dqn": DQN, "ppo": PPO, "a2c": A2C}[algorithm.lower()]
    agent_models_map = {}
    
    for intersection in intersections:
        tl_id = INTERSECTION_TO_TL_ID[intersection]
        
        # Try to find the best model first, then final
        model_path_best = Path(model_dir) / f"{prefix}_{intersection}_{algorithm}_{algorithm}_model_best" / "best_model.zip"
        model_path_final = Path(model_dir) / f"{prefix}_{intersection}_{algorithm}_{algorithm}_model_final.zip"
        
        model_path = None
        if model_path_best.exists():
            model_path = model_path_best
            print(f"[{intersection}] Loading best model: {model_path}")
        elif model_path_final.exists():
            model_path = model_path_final
            print(f"[{intersection}] Loading final model: {model_path}")
        else:
            print(f"[{intersection}] WARNING: No model found for {intersection}")
            print(f"  Searched for: {model_path_best}")
            print(f"  Searched for: {model_path_final}")
            continue
        
        try:
            model = model_class.load(model_path)
            agent_models_map[tl_id] = model
            print(f"[{intersection}] Successfully loaded model for TL ID: {tl_id[:50]}...")
        except Exception as e:
            print(f"[{intersection}] ERROR loading model: {e}")
    
    return agent_models_map


def evaluate_heterogeneous_agents(env, n_episodes=10):
    """
    Evaluate heterogeneous agents on the environment.
    
    Args:
        env: HeterogeneousMultiAgentWrapper environment
        n_episodes: Number of episodes to evaluate
        
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
            # Note: action is computed inside step() using agent policies
            obs, reward, done, truncated, info = env.step()
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
        if 'system_mean_waiting_time' in info:
            print(f"  Waiting Time: {info['system_mean_waiting_time']:.2f}s, "
                  f"Stopped: {info['system_total_stopped']:.1f}, "
                  f"Speed: {info['system_mean_speed']:.2f} m/s")
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_waiting_time': np.mean(episode_metrics['system_mean_waiting_time']) if episode_metrics['system_mean_waiting_time'] else None,
        'std_waiting_time': np.std(episode_metrics['system_mean_waiting_time']) if episode_metrics['system_mean_waiting_time'] else None,
        'mean_total_stopped': np.mean(episode_metrics['system_total_stopped']) if episode_metrics['system_total_stopped'] else None,
        'std_total_stopped': np.std(episode_metrics['system_total_stopped']) if episode_metrics['system_total_stopped'] else None,
        'mean_speed': np.mean(episode_metrics['system_mean_speed']) if episode_metrics['system_mean_speed'] else None,
        'std_speed': np.std(episode_metrics['system_mean_speed']) if episode_metrics['system_mean_speed'] else None,
    }
    
    return results, episode_rewards, episode_metrics


def main(args):
    # File paths
    NET_FILE = args.net_file
    ROUTE_FILE = args.route_file
    
    print("="*80)
    print("HETEROGENEOUS MULTI-AGENT EVALUATION")
    print("="*80)
    print(f"Network: {NET_FILE}")
    print(f"Routes: {ROUTE_FILE}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Intersections: {', '.join(args.intersections)}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Episode duration: {args.episode_seconds}s")
    print("="*80 + "\n")
    
    # Load trained models for each intersection
    print("Loading trained models...")
    agent_models_map = load_models(
        args.intersections,
        args.algorithm,
        model_dir=args.model_dir,
        prefix=args.model_prefix
    )
    
    if not agent_models_map:
        print("\nERROR: No models were loaded. Cannot proceed with evaluation.")
        print("Make sure the models exist in the specified directory.")
        return
    
    print(f"\nOK: Successfully loaded {len(agent_models_map)} models")
    print()
    
    # Create PettingZoo parallel environment
    print("Creating multi-agent environment...")
    pz_env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=args.gui,
        num_seconds=args.episode_seconds,
        add_system_info=True,
        reward_fn=reward_minimize_max_queue,
        observation_class=GridObservationFunction,
        sumo_seed=args.seed if args.seed else 'random'
    )
    
    # Wrap environment for heterogeneous multi-agent evaluation
    env = HeterogeneousMultiAgentWrapper(pz_env, agent_models_map)
    print()
    
    # Compute baseline if requested
    baseline_metrics = None
    if args.compare_baseline:
        print("Computing baseline metrics...")
        print("Running baseline with fixed-time traffic lights...")
        try:
            baseline_metrics = run_baseline(NET_FILE, ROUTE_FILE, args.episode_seconds)
            print(f"OK: Baseline - Waiting Time: {baseline_metrics['mean_waiting_time']:.2f}s")
            print(f"OK: Baseline - Stopped Vehicles: {baseline_metrics['mean_stopped']:.2f}")
            print(f"OK: Baseline - Speed: {baseline_metrics['mean_speed']:.2f} m/s")
        except Exception as e:
            print(f"WARNING: Could not compute baseline: {e}")
            baseline_metrics = None
        print()
    
    # Initialize Wandb if requested
    if args.use_wandb:
        wandb.init(
            entity="fds-final-project",
            project="rl-traffic-light",
            name=args.run_name if args.run_name else f"eval-multiagent-{args.algorithm}",
            config={
                "algorithm": args.algorithm,
                "intersections": args.intersections,
                "n_agents": len(agent_models_map),
                "n_eval_episodes": args.n_episodes,
                "episode_seconds": args.episode_seconds,
                "net_file": NET_FILE,
                "route_file": ROUTE_FILE,
                "seed": args.seed,
            }
        )
    
    # Evaluate the heterogeneous agents
    print(f"\nEvaluating heterogeneous agents for {args.n_episodes} episodes...")
    print("="*80)
    results, episode_rewards, episode_metrics = evaluate_heterogeneous_agents(
        env, n_episodes=args.n_episodes
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} steps")
    
    if results['mean_waiting_time'] is not None:
        print(f"\nTraffic Metrics:")
        print(f"  Mean Waiting Time: {results['mean_waiting_time']:.2f} ± {results['std_waiting_time']:.2f} s")
        print(f"  Mean Total Stopped: {results['mean_total_stopped']:.2f} ± {results['std_total_stopped']:.2f}")
        print(f"  Mean Speed: {results['mean_speed']:.2f} ± {results['std_speed']:.2f} m/s")
        
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
    
    print("="*80 + "\n")
    
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
                "eval/std_waiting_time": results['std_waiting_time'],
                "eval/mean_total_stopped": results['mean_total_stopped'],
                "eval/std_total_stopped": results['std_total_stopped'],
                "eval/mean_speed": results['mean_speed'],
                "eval/std_speed": results['std_speed'],
            })
        
        wandb.log(wandb_log)
        wandb.finish()
    
    # Save results to file if requested
    if args.output_file:
        import json
        output_data = {
            'config': {
                'algorithm': args.algorithm,
                'intersections': args.intersections,
                'n_episodes': args.n_episodes,
                'episode_seconds': args.episode_seconds,
            },
            'results': results,
            'episode_rewards': episode_rewards,
            'episode_metrics': {k: v for k, v in episode_metrics.items()}
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output_file}")
    
    env.close()
    print("Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate heterogeneous multi-agent system with independently trained agents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment configuration
    parser.add_argument(
        "--net-file",
        type=str,
        default="scenarios/berlin-small/berlin-small-static.net.xml",
        help="SUMO network file for evaluation"
    )
    
    parser.add_argument(
        "--route-file",
        type=str,
        default="scenarios/berlin-small/berlin-small-static-eval.rou.xml",
        help="SUMO route file for evaluation"
    )
    
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=3600,
        help="Duration of each episode in seconds"
    )
    
    # Model configuration
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        required=True,
        choices=["dqn", "ppo", "a2c"],
        help="RL algorithm used for training"
    )
    
    parser.add_argument(
        "--intersections",
        type=str,
        nargs="+",
        default=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        help="List of intersection letters to load models for (A-J)"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="weights",
        help="Directory containing trained model weights"
    )
    
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="berlin",
        help="Prefix for model file names"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compute baseline metrics for comparison"
    )
    
    # Visualization and logging
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI visualization"
    )
    
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log evaluation metrics to Wandb"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this evaluation run (for Wandb)"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Save evaluation results to JSON file"
    )
    
    args = parser.parse_args()
    main(args)
