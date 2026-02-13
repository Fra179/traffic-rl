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
from traffic_rl.callbacks import run_baseline

os.environ["LIBSUMO_AS_TRACI"] = "1"

# Scenario configuration
NET_FILE = "scenarios/berlin-small/berlin-small.net.xml"
ROUTE_FILE = "scenarios/berlin-small/berlin-small.rou.xml"
EVAL_ROUTE_FILE = "scenarios/berlin-small/berlin-small.rou.xml"

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
LOG_FREQ = 100  # Log training metrics every N steps
VAL_FREQ = 7200  # Run validation every N steps (roughly per episode)


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
    def __init__(self, eval_env=None, baseline_metrics=None, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.step_count = 0
        self.cumulative_arrived = 0
        self.cumulative_switches = 0
        self.last_green_phases = {}
        self.eval_env = eval_env
        self.baseline_metrics = baseline_metrics
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Log step-level metrics every LOG_FREQ steps
        if self.step_count % LOG_FREQ == 0:
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
                    
                    # Get SUMO environment metrics
                    wait_time = info.get('system_total_waiting_time', 0)
                    queue_len = info.get('system_total_stopped', 0)
                    mean_speed = info.get('system_mean_speed', 0)
                    
                    metrics_to_log['waiting_time'] = wait_time
                    metrics_to_log['queue_length'] = queue_len
                    metrics_to_log['mean_speed'] = mean_speed
                    
                    # Try to get vehicle counts and switches from environment
                    try:
                        sumo_envs = self.training_env.envs
                        if len(sumo_envs) > 0:
                            env_unwrapped = sumo_envs[0].unwrapped
                            # Navigate through wrappers
                            while hasattr(env_unwrapped, 'pz_env'):
                                env_unwrapped = env_unwrapped.pz_env
                            
                            if hasattr(env_unwrapped, 'env') and hasattr(env_unwrapped.env, 'sumo'):
                                sumo_conn = env_unwrapped.env.sumo
                                total_vehicles = sumo_conn.vehicle.getIDCount()
                                arrived_now = sumo_conn.simulation.getArrivedNumber()
                                self.cumulative_arrived += arrived_now
                                
                                metrics_to_log['total_vehicles'] = total_vehicles
                                metrics_to_log['arrived_vehicles_cumulative'] = self.cumulative_arrived
                                
                                # Track phase switches
                                ts_dict = env_unwrapped.env.traffic_signals
                                if not self.last_green_phases:
                                    self.last_green_phases = {ts: ts_obj.green_phase for ts, ts_obj in ts_dict.items()}
                                
                                for ts_id, ts_obj in ts_dict.items():
                                    curr_green = ts_obj.green_phase
                                    if curr_green != self.last_green_phases.get(ts_id, -1):
                                        self.cumulative_switches += 1
                                        self.last_green_phases[ts_id] = curr_green
                                
                                metrics_to_log['switches_cumulative'] = self.cumulative_switches
                    except Exception as e:
                        pass
                    
                    # Only log if we have meaningful metrics
                    if len(metrics_to_log) > 1:
                        wandb.log(metrics_to_log)
                        print(f"Step {self.step_count}: reward={metrics_to_log.get('reward', 0):.2f}, "
                              f"queue={metrics_to_log.get('queue_length', 0):.1f}")
        
        # Run validation periodically
        if self.eval_env is not None and self.step_count > 0 and self.step_count % VAL_FREQ == 0:
            self._run_validation()
        
        # Log episode summary when episodes complete
        if len(self.model.ep_info_buffer) > 0:
            current_ep_count = len(self.model.ep_info_buffer)
            if current_ep_count > self.episode_count:
                self.episode_count = current_ep_count
                ep_info = self.model.ep_info_buffer[-1]
                
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
    
    def _run_validation(self):
        """Run validation episode with deterministic policy"""
        print(f"Running Validation at step {self.step_count}...")
        
        obs, info = self.eval_env.reset()
        done = False
        truncated = False
        
        wait_times = []
        queues = []
        speeds = []
        total_arrived = 0
        total_switches = 0
        
        # Initialize phase tracking
        try:
            env_unwrapped = self.eval_env.unwrapped
            while hasattr(env_unwrapped, 'pz_env'):
                env_unwrapped = env_unwrapped.pz_env
            
            if hasattr(env_unwrapped, 'env') and hasattr(env_unwrapped.env, 'traffic_signals'):
                ts_dict = env_unwrapped.env.traffic_signals
                last_green_phases = {ts: ts_obj.green_phase for ts, ts_obj in ts_dict.items()}
            else:
                last_green_phases = {}
        except:
            last_green_phases = {}
        
        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.eval_env.step(action)
            
            wait_times.append(info.get('system_total_waiting_time', 0))
            queues.append(info.get('system_total_stopped', 0))
            speeds.append(info.get('system_mean_speed', 0))
            
            # Track arrived and switches
            try:
                env_unwrapped = self.eval_env.unwrapped
                while hasattr(env_unwrapped, 'pz_env'):
                    env_unwrapped = env_unwrapped.pz_env
                
                if hasattr(env_unwrapped, 'env'):
                    total_arrived += env_unwrapped.env.sumo.simulation.getArrivedNumber()
                    
                    ts_dict = env_unwrapped.env.traffic_signals
                    for ts_id, ts_obj in ts_dict.items():
                        curr = ts_obj.green_phase
                        if curr != last_green_phases.get(ts_id, curr):
                            total_switches += 1
                            last_green_phases[ts_id] = curr
            except:
                pass
        
        mean_wait = np.mean(wait_times) if wait_times else 0
        mean_queue = np.mean(queues) if queues else 0
        mean_speed = np.mean(speeds) if speeds else 0
        
        # Log validation metrics
        log_dict = {
            "validation/step": self.step_count,
            "validation/mean_waiting_time": mean_wait,
            "validation/mean_queue_length": mean_queue,
            "validation/mean_speed": mean_speed,
            "validation/total_arrived": total_arrived,
            "validation/total_switches": total_switches,
        }
        
        # Add baseline comparison if available
        if self.baseline_metrics:
            log_dict.update({
                "baseline/mean_waiting_time": self.baseline_metrics['mean_waiting_time'],
                "baseline/mean_queue_length": self.baseline_metrics['mean_queue_length'],
                "baseline/mean_speed": self.baseline_metrics['mean_speed'],
                "baseline/total_arrived": self.baseline_metrics['total_arrived'],
            })
            print(f"Validation Complete. Arrived: {total_arrived} vs Baseline: {self.baseline_metrics['total_arrived']}")
        else:
            print(f"Validation Complete. Arrived: {total_arrived}, Speed: {mean_speed:.2f}")
        
        wandb.log(log_dict)


def make_env():
    pz_env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        out_csv_name=None,  # Disable CSV for performance
        use_gui=False,
        num_seconds=NUM_SECONDS,
        reward_fn=reward_minimize_max_queue,
        add_system_info=True,  # Enable system metrics
    )
    return PettingZooToGymWrapper(pz_env)

def make_eval_env():
    """Create evaluation environment (single instance, fixed seed)"""
    pz_env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=EVAL_ROUTE_FILE,  # Use eval routes if available
        out_csv_name=None,
        use_gui=False,
        num_seconds=NUM_SECONDS,
        reward_fn=reward_minimize_max_queue,
        add_system_info=True,
        sumo_seed=42,  # Fixed seed for reproducible evaluation
    )
    return PettingZooToGymWrapper(pz_env)

if __name__ == "__main__":
    # Try to compute baseline metrics (may fail for multi-agent scenarios)
    baseline_metrics = None
    try:
        print("Computing baseline metrics...")
        baseline_metrics = run_baseline(
            NET_FILE,
            ROUTE_FILE,
            NUM_SECONDS
        )
        print(f"Baseline: arrived={baseline_metrics['total_arrived']}, "
              f"speed={baseline_metrics['mean_speed']:.2f}")
    except Exception as e:
        print(f"Could not compute baseline (multi-agent scenario): {e}")
        print("Continuing without baseline comparison...")
    
    # Initialize wandb
    wandb.init(
        entity="fds-final-project",
        project="rl-traffic-light",
        name="4x4grid-multiagent-ppo",
        config={
            "algorithm": "PPO",
            "env": "4x4grid",
            "net_file": NET_FILE,
            "route_file": ROUTE_FILE,
            "eval_route_file": EVAL_ROUTE_FILE,
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
            "log_freq": LOG_FREQ,
            "val_freq": VAL_FREQ,
        }
    )
    
    # Add baseline to config if available
    if baseline_metrics:
        wandb.config.update({"baseline_metrics": baseline_metrics})
    
    # Create parallel training environments
    vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
    
    # Create single evaluation environment
    try:
        eval_env = make_eval_env()
        print("Evaluation environment created successfully")
    except FileNotFoundError:
        # Fallback to training routes if eval routes don't exist
        print("Eval routes not found, using training routes for validation")
        eval_env = make_env()

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
        callback=WandbCallback(eval_env=eval_env, baseline_metrics=baseline_metrics),
        progress_bar=True
    )
    
    # Cleanup
    eval_env.close()
    vec_env.close()
    
    # Save model
    model.save("models/4x4grid_ppo")
    print("Training complete! Model saved to models/4x4grid_ppo")
    
    # Close wandb
    wandb.finish()