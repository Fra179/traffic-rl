import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import sumo_rl
from gymnasium import spaces

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


class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_count += 1
            if self.episode_count % 10 == 0:
                print(f"Episode {self.episode_count}: reward={ep_info.get('r', 0):.2f}, length={ep_info.get('l', 0)}")
        return True


if __name__ == "__main__":
    # Create PettingZoo environment
    pz_env = sumo_rl.parallel_env(
        net_file="scenarios/grid4x4/grid4x4.net.xml",
        route_file="scenarios/grid4x4/train_generated.rou.xml",
        out_csv_name="outputs/4x4grid/ppo",
        use_gui=False,
        num_seconds=80000,
    )
    
    # Wrap as single-agent Gym environment
    gym_env = PettingZooToGymWrapper(pz_env)
    vec_env = DummyVecEnv([lambda: gym_env])
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=2e-5,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.9,
        clip_range=0.4,
        ent_coef=0.1,
        vf_coef=0.25,
        verbose=1,
        tensorboard_log="./logs/4x4grid_ppo/",
        device="cpu"  # Force CPU usage
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=100000,
        callback=ProgressCallback(),
        progress_bar=True
    )
    
    # Save model
    model.save("models/4x4grid_ppo")
    print("Training complete! Model saved to models/4x4grid_ppo")