import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

def reward_minimize_queue(ts):
    # Directly penalize the number of stopped cars
    return -float(ts.get_total_queued())

class TrafficWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrafficWandbCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]
        reward = self.locals['rewards'][0] # Normalized Reward
        
        wait_time = infos.get('system_total_waiting_time', 0)
        queue_len = infos.get('system_total_stopped', 0)
        mean_speed = infos.get('system_mean_speed', 0)
        
        try:
            sumo_conn = self.training_env.envs[0].unwrapped.sumo
            total_vehicles = sumo_conn.vehicle.getIDCount()
        except:
            total_vehicles = 0

        wandb.log({
            "step": self.num_timesteps,
            "reward": reward,
            "waiting_time": wait_time,
            "queue_length": queue_len,
            "mean_speed": mean_speed,
            "total_vehicles": total_vehicles,
        })
        return True

class ValidationCallback(BaseCallback):
    def __init__(self, eval_env, baseline_metrics, eval_freq=720, verbose=0):
        super(ValidationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.baseline_metrics = baseline_metrics
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0:
            print(f"Running Validation at step {self.num_timesteps}...")
            
            # Run one episode with current model
            obs, info = self.eval_env.reset()
            done = False
            truncated = False
            
            # Collectors
            wait_times = []
            queues = []
            speeds = []
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                wait_times.append(info.get('system_total_waiting_time', 0))
                queues.append(info.get('system_total_stopped', 0))
                speeds.append(info.get('system_mean_speed', 0))

            mean_wait = np.mean(wait_times)
            mean_queue = np.mean(queues)
            mean_speed = np.mean(speeds)

            # Log comparison
            wandb.log({
                "validation/step": self.num_timesteps,
                "validation/mean_waiting_time": mean_wait,
                "validation/mean_queue_length": mean_queue,
                "validation/mean_speed": mean_speed,
                
                "baseline/mean_waiting_time": self.baseline_metrics['mean_waiting_time'],
                "baseline/mean_queue_length": self.baseline_metrics['mean_queue_length'],
                "baseline/mean_speed": self.baseline_metrics['mean_speed'],
            })
            
            print(f"Validation Complete. Speed: {mean_speed:.2f} vs Baseline: {self.baseline_metrics['mean_speed']:.2f}")
            
            # Important: Reset env for next time (though reset() does it)
            # self.eval_env.reset() # Not strictly needed as we reset at start of block
            
        return True

def run_baseline(net_file, route_file, num_seconds):
    print("Computing Fixed-TS Baseline...")
    import gymnasium as gym # Ensure clean env
    
    # Use fixed_ts=True for baseline
    env = gym.make('sumo-rl-v0',
                   net_file=net_file,
                   route_file=route_file,
                   num_seconds=num_seconds,
                   use_gui=False,
                   fixed_ts=True,
                   sumo_seed='42') # Fixed seed for baseline to be consistent
    
    obs, info = env.reset()
    done = False
    truncated = False
    
    wait_times = []
    queues = []
    speeds = []
    
    while not (done or truncated):
        action = env.action_space.sample() # Ignored
        obs, reward, done, truncated, info = env.step(action)
        
        wait_times.append(info.get('system_total_waiting_time', 0))
        queues.append(info.get('system_total_stopped', 0))
        speeds.append(info.get('system_mean_speed', 0))
        
    env.close()
    
    metrics = {
        "mean_waiting_time": np.mean(wait_times),
        "mean_queue_length": np.mean(queues),
        "mean_speed": np.mean(speeds)
    }
    print(f"Baseline Computed: {metrics}")
    return metrics
