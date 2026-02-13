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
                
                # Handle multi-agent (PettingZooToGymWrapper) vs single-agent
                if hasattr(env_unwrapped, 'pz_env'):
                    # Multi-agent: access through PettingZoo parallel env
                    pz_env = env_unwrapped.pz_env
                    sumo_conn = pz_env.env.sumo if hasattr(pz_env, 'env') else pz_env.sumo
                    ts_dict = pz_env.env.ts_ids if hasattr(pz_env, 'env') else pz_env.ts_ids
                    ts_dict = {ts_id: pz_env.env._traffic_signals[ts_id] if hasattr(pz_env, 'env') else pz_env._traffic_signals[ts_id] for ts_id in ts_dict}
                else:
                    # Single-agent: direct access
                    sumo_conn = env_unwrapped.sumo
                    ts_dict = env_unwrapped.traffic_signals
                
                total_vehicles = sumo_conn.vehicle.getIDCount()
                arrived_now = sumo_conn.simulation.getArrivedNumber()
                
                # Count switches
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
    Saves best model based on throughput (total arrived vehicles).
    
    Args:
        eval_env: Separate evaluation environment
        baseline_metrics: Dictionary of baseline metrics
        eval_freq: How often to run validation (in steps)
        best_model_save_path: Path to save best model (based on throughput)
        verbose: Verbosity level
    """
    
    def __init__(self, eval_env, baseline_metrics, eval_freq=720, best_model_save_path=None, verbose=0):
        super(ValidationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.baseline_metrics = baseline_metrics
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.best_throughput = -np.inf

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.eval_freq == 0:
            print(f"\nRunning Validation at step {self.num_timesteps}...")
            
            # Run one episode with current model
            obs, info = self.eval_env.reset()
            done = False
            truncated = False
            
            # Collectors
            wait_times = []
            queues = []
            speeds = []
            total_switches = 0
            
            # Init phase tracking
            try:
                env_unwrapped = self.eval_env.unwrapped
                # Handle multi-agent (PettingZooToGymWrapper) vs single-agent
                if hasattr(env_unwrapped, 'pz_env'):
                    pz_env = env_unwrapped.pz_env
                    ts_ids = pz_env.env.ts_ids if hasattr(pz_env, 'env') else pz_env.ts_ids
                    ts_dict = {ts_id: pz_env.env._traffic_signals[ts_id] if hasattr(pz_env, 'env') else pz_env._traffic_signals[ts_id] for ts_id in ts_ids}
                else:
                    ts_dict = env_unwrapped.traffic_signals
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
                    env_unwrapped = self.eval_env.unwrapped
                    # Handle multi-agent (PettingZooToGymWrapper) vs single-agent
                    if hasattr(env_unwrapped, 'pz_env'):
                        pz_env = env_unwrapped.pz_env
                        ts_ids = pz_env.env.ts_ids if hasattr(pz_env, 'env') else pz_env.ts_ids
                        ts_dict = {ts_id: pz_env.env._traffic_signals[ts_id] if hasattr(pz_env, 'env') else pz_env._traffic_signals[ts_id] for ts_id in ts_ids}
                    else:
                        ts_dict = env_unwrapped.traffic_signals
                    
                    # Track switches
                    for ts_id, ts_obj in ts_dict.items():
                        curr = ts_obj.green_phase
                        if curr != last_green_phases.get(ts_id, curr):
                            total_switches += 1
                            last_green_phases[ts_id] = curr
                except:
                    pass
            
            # Get total arrived vehicles from SUMO after episode completes
            try:
                env_unwrapped = self.eval_env.unwrapped
                if hasattr(env_unwrapped, 'pz_env'):
                    pz_env = env_unwrapped.pz_env
                    sumo_conn = pz_env.env.sumo if hasattr(pz_env, 'env') else pz_env.sumo
                else:
                    sumo_conn = env_unwrapped.sumo
                
                # Get total arrived from SUMO statistics (cumulative for the episode)
                total_arrived = sumo_conn.simulation.getArrivedNumber()
                # Also try departed as a sanity check
                total_departed = sumo_conn.simulation.getDepartedNumber()
            except Exception as e:
                print(f"  Warning: Could not get arrival statistics: {e}")
                total_arrived = 0
                total_departed = 0

            mean_wait = np.mean(wait_times)
            mean_queue = np.mean(queues)
            mean_speed = np.mean(speeds)

            # Log comparison
            log_dict = {
                "validation/step": self.num_timesteps,
                "validation/mean_waiting_time": mean_wait,
                "validation/mean_queue_length": mean_queue,
                "validation/mean_speed": mean_speed,
                "validation/total_arrived": total_arrived,
                "validation/total_departed": total_departed,
                "validation/total_switches": total_switches,
            }
            
            # Add baseline metrics if available
            if self.baseline_metrics:
                log_dict.update({
                    "baseline/mean_waiting_time": self.baseline_metrics['mean_waiting_time'],
                    "baseline/mean_queue_length": self.baseline_metrics['mean_queue_length'],
                    "baseline/mean_speed": self.baseline_metrics['mean_speed'],
                    "baseline/total_arrived": self.baseline_metrics['total_arrived'],
                    "baseline/total_switches": self.baseline_metrics.get('total_switches', 0),
                })
            
            wandb.log(log_dict)
            
            # Save best model based on throughput (total arrived vehicles)
            if total_arrived > self.best_throughput:
                prev_best = self.best_throughput
                self.best_throughput = total_arrived
                if self.best_model_save_path is not None:
                    import os
                    os.makedirs(self.best_model_save_path, exist_ok=True)
                    self.model.save(f"{self.best_model_save_path}/best_model")
                    if prev_best == -np.inf:
                        print(f"  New best model saved! Throughput: {total_arrived}")
                    else:
                        print(f"  New best model saved! Throughput: {total_arrived} (previous: {int(prev_best)})")
                    wandb.log({"validation/best_throughput": self.best_throughput})
            
            if self.baseline_metrics:
                print(f"Validation Complete. Arrived: {total_arrived} vs Baseline: {self.baseline_metrics['total_arrived']} (Best: {self.best_throughput})")
            else:
                print(f"Validation Complete. Arrived: {total_arrived} (Best: {self.best_throughput})")
            
        return True
