"""Weights & Biases callbacks for traffic light training"""

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class TrafficWandbCallback(BaseCallback):
    """
    Callback for logging training metrics to Weights & Biases.
    
    Logs:
    - Reward
    - Waiting time
    - Queue length
    - Mean speed
    - Total vehicles
    - Cumulative arrived vehicles
    - Cumulative phase switches
    """
    
    def __init__(self, verbose=0):
        super(TrafficWandbCallback, self).__init__(verbose)
        self.cumulative_arrived = 0
        self.cumulative_switches = 0
        self.last_green_phases = {}

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]
        reward = self.locals['rewards'][0]  # Normalized Reward
        
        wait_time = infos.get('system_total_waiting_time', 0)
        queue_len = infos.get('system_total_stopped', 0)
        mean_speed = infos.get('system_mean_speed', 0)
        
        arrived_now = 0
        total_vehicles = 0
        
        try:
            sumo_envs = self.training_env.envs
            if len(sumo_envs) > 0:
                env_unwrapped = sumo_envs[0].unwrapped
                sumo_conn = env_unwrapped.sumo
                total_vehicles = sumo_conn.vehicle.getIDCount()
                arrived_now = sumo_conn.simulation.getArrivedNumber()
                
                # Count switches
                ts_dict = env_unwrapped.traffic_signals
                # Initialize on first seen
                if not self.last_green_phases:
                    self.last_green_phases = {ts: ts_obj.green_phase for ts, ts_obj in ts_dict.items()}
                
                for ts_id, ts_obj in ts_dict.items():
                    curr_green = ts_obj.green_phase
                    if curr_green != self.last_green_phases.get(ts_id, -1):
                        self.cumulative_switches += 1
                        self.last_green_phases[ts_id] = curr_green
        except Exception as e:
            pass

        self.cumulative_arrived += arrived_now

        log_dict = {
            "step": self.num_timesteps,
            "reward": reward,
            "waiting_time": wait_time,
            "queue_length": queue_len,
            "mean_speed": mean_speed,
            "total_vehicles": total_vehicles,
            "arrived_vehicles_cumulative": self.cumulative_arrived,
            "switches_cumulative": self.cumulative_switches
        }
        
        wandb.log(log_dict)
        return True


class ValidationCallback(BaseCallback):
    """
    Callback for periodic validation during training.
    
    Runs deterministic evaluation episodes and compares against baseline.
    
    Args:
        eval_env: Separate evaluation environment
        baseline_metrics: Dictionary of baseline metrics
        eval_freq: How often to run validation (in steps)
        verbose: Verbosity level
    """
    
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
            total_arrived = 0
            total_switches = 0
            
            # Init phase tracking
            try:
                ts_dict = self.eval_env.unwrapped.traffic_signals
                last_green_phases = {ts: ts_obj.green_phase for ts, ts_obj in ts_dict.items()}
            except:
                last_green_phases = {}

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                
                wait_times.append(info.get('system_total_waiting_time', 0))
                queues.append(info.get('system_total_stopped', 0))
                speeds.append(info.get('system_mean_speed', 0))
                
                try:
                    total_arrived += self.eval_env.unwrapped.sumo.simulation.getArrivedNumber()
                    
                    # Track switches
                    ts_dict = self.eval_env.unwrapped.traffic_signals
                    for ts_id, ts_obj in ts_dict.items():
                        curr = ts_obj.green_phase
                        if curr != last_green_phases.get(ts_id, curr):
                            total_switches += 1
                            last_green_phases[ts_id] = curr
                except:
                    pass

            mean_wait = np.mean(wait_times)
            mean_queue = np.mean(queues)
            mean_speed = np.mean(speeds)

            # Log comparison
            wandb.log({
                "validation/step": self.num_timesteps,
                "validation/mean_waiting_time": mean_wait,
                "validation/mean_queue_length": mean_queue,
                "validation/mean_speed": mean_speed,
                "validation/total_arrived": total_arrived,
                "validation/total_switches": total_switches,
                
                "baseline/mean_waiting_time": self.baseline_metrics['mean_waiting_time'],
                "baseline/mean_queue_length": self.baseline_metrics['mean_queue_length'],
                "baseline/mean_speed": self.baseline_metrics['mean_speed'],
                "baseline/total_arrived": self.baseline_metrics['total_arrived'],
                "baseline/total_switches": self.baseline_metrics.get('total_switches', 0),
            })
            
            print(f"Validation Complete. Arrived: {total_arrived} vs Baseline: {self.baseline_metrics['total_arrived']}")
            
        return True
