import os
import sys
from stable_baselines3.common.vec_env import SubprocVecEnv

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import sumo_rl
from gymnasium import spaces

from traffic_rl.rewards import reward_minimize_max_queue

os.environ["LIBSUMO_AS_TRACI"] = "1"

# Training configuration constants
LEARNING_RATE = 2e-5
N_STEPS = 2048
BATCH_SIZE = 512
N_EPOCHS = 10
NUM_SECONDS = 7200
N_ENVS = 8
TOTAL_TIMESTEPS = 100000
GAMMA = 0.95
GAE_LAMBDA = 0.9
CLIP_RANGE = 0.4
ENT_COEF = 0.1
VF_COEF = 0.25


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


class WandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Log step-level metrics (check dict membership correctly)
        if 'infos' in self.locals and self.locals['infos'] is not None:
            infos = self.locals['infos']
            if len(infos) > 0:
                info = infos[0]
                
                # Extract metrics from info dict
                metrics_to_log = {
                    'step': self.step_count,
                }
                
                # Add reward if available
                if 'rewards' in self.locals:
                    rewards = self.locals['rewards']
                    if len(rewards) > 0:
                        metrics_to_log['reward'] = float(rewards[0])
                
                # Add environment metrics if available
                for key in ['waiting_time', 'total_vehicles', 'queue_length', 
                           'mean_speed', 'arrived_vehicles_cumulative', 'switches_cumulative']:
                    if key in info:
                        metrics_to_log[key] = info[key]
                
                # Only log if we have metrics beyond just the step count
                if len(metrics_to_log) > 1:
                    wandb.log(metrics_to_log)
                    
                # Print progress periodically
                if self.step_count % 100 == 0:
                    print(f"Step {self.step_count}: reward={metrics_to_log.get('reward', 0):.2f}")
        
        # Log episode summary when episodes complete
        if len(self.model.ep_info_buffer) > 0:
            # Check if this is a new episode
            ep_info = self.model.ep_info_buffer[-1]
            
            # Simple tracking - log if we haven't seen this many episodes yet
            current_ep_count = len(self.model.ep_info_buffer)
            if current_ep_count > self.episode_count:
                self.episode_count = current_ep_count
                
                episode_metrics = {
                    'episode': self.episode_count,
                    'episode_reward': ep_info.get('r', 0),
                    'episode_length': ep_info.get('l', 0),
                }
                
                if 'total_reward' in ep_info:
                    episode_metrics['episode_total_reward'] = ep_info['total_reward']
                if 'avg_reward' in ep_info:
                    episode_metrics['episode_avg_reward'] = ep_info['avg_reward']
                
                wandb.log(episode_metrics)
                print(f"Episode {self.episode_count} completed: reward={ep_info.get('r', 0):.2f}, length={ep_info.get('l', 0)}")
        
        return True


def make_env():
    pz_env = sumo_rl.parallel_env(
        net_file="scenarios/grid4x4/grid4x4.net.xml",
        route_file="scenarios/grid4x4/train_generated.rou.xml",
        out_csv_name=None,  # Disable CSV for performance
        use_gui=False,
        num_seconds=NUM_SECONDS,
        reward_fn=reward_minimize_max_queue,
    )
    return PettingZooToGymWrapper(pz_env)

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        entity="fds-final-project",
        project="rl-traffic-light",
        name="4x4grid-multiagent-ppo",
        config={
            "algorithm": "PPO",
            "env": "4x4grid",
            "reward_fn": "minimize_max_queue",
            "learning_rate": LEARNING_RATE,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "n_epochs": N_EPOCHS,
            "num_seconds": NUM_SECONDS,
            "n_envs": N_ENVS,
            "total_timesteps": TOTAL_TIMESTEPS,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
        }
    )
    
    # Create parallel environments
    vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        verbose=1,
        tensorboard_log="./logs/4x4grid_ppo/",
        # device="cpu"  # Force CPU usage
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=WandbCallback(),
        progress_bar=True
    )
    
    # Save model
    model.save("models/4x4grid_ppo")
    print("Training complete! Model saved to models/4x4grid_ppo")
    
    # Close wandb
    wandb.finish()