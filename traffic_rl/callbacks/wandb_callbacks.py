"""Weights & Biases callbacks for traffic light training"""

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import traci


def _arrived_from_sumo_conn(sumo_conn):
    """Prefer ID-list based arrivals; fallback to numeric counter."""
    try:
        arrived_ids = sumo_conn.simulation.getArrivedIDList()
        if arrived_ids is not None:
            return max(0, int(len(arrived_ids)))
    except Exception:
        pass
    try:
        return max(0, int(sumo_conn.simulation.getArrivedNumber()))
    except Exception:
        return 0


def _arrived_from_global_traci():
    try:
        arrived_ids = traci.simulation.getArrivedIDList()
        if arrived_ids is not None:
            return max(0, int(len(arrived_ids)))
    except Exception:
        pass
    try:
        return max(0, int(traci.simulation.getArrivedNumber()))
    except Exception:
        return 0


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
        infos_all = self.locals.get("infos", [])
        rewards_all = self.locals.get("rewards", [])
        if len(infos_all) == 0:
            return True

        # Aggregate across vectorized envs for stable logging with n_envs > 1.
        reward = float(np.mean(rewards_all)) if len(rewards_all) > 0 else 0.0
        wait_time = float(np.mean([info.get("system_total_waiting_time", 0) for info in infos_all]))
        queue_len = float(np.mean([info.get("system_total_stopped", 0) for info in infos_all]))
        mean_speed = float(np.mean([info.get("system_mean_speed", 0) for info in infos_all]))
        
        arrived_now = 0
        total_vehicles = 0

        # Prefer metrics coming through info (subprocess-safe for vectorized envs).
        info_total_vehicles = int(sum(info.get("system_total_vehicles", 0) for info in infos_all))
        info_arrived_now = int(sum(info.get("system_arrived_now", 0) for info in infos_all))
        if info_total_vehicles > 0:
            total_vehicles = info_total_vehicles
        if info_arrived_now > 0:
            arrived_now = info_arrived_now
        
        try:
            # Works for DummyVecEnv and SubprocVecEnv when wrappers expose this method.
            metrics_list = self.training_env.env_method("get_sumo_metrics")
            if metrics_list:
                metrics_total_vehicles = int(sum(m.get("total_vehicles", 0) for m in metrics_list))
                metrics_arrived_now = int(sum(m.get("arrived_now", 0) for m in metrics_list))
                if metrics_total_vehicles > 0:
                    total_vehicles = metrics_total_vehicles
                if metrics_arrived_now > 0:
                    arrived_now = metrics_arrived_now
                self.cumulative_switches = int(sum(m.get("cumulative_switches", 0) for m in metrics_list))
        except Exception:
            # Fallback for single-agent envs without get_sumo_metrics.
            try:
                sumo_envs = self.training_env.envs
                if len(sumo_envs) > 0:
                    env_unwrapped = sumo_envs[0].unwrapped
                    sumo_conn = env_unwrapped.sumo
                    total_vehicles = sumo_conn.vehicle.getIDCount()
                    arrived_now = _arrived_from_sumo_conn(sumo_conn)
            except Exception:
                pass

        if arrived_now <= 0:
            arrived_now = _arrived_from_global_traci()

        if total_vehicles == 0 and len(infos_all) > 0:
            # Some env variants expose this directly in info; prefer non-zero if available.
            total_vehicles = int(max(info.get("system_total_vehicles", 0) for info in infos_all))

        if total_vehicles < 0:
            total_vehicles = 0
        if arrived_now < 0:
            arrived_now = 0

        if np.isnan(reward):
            reward = 0.0

        if np.isnan(wait_time):
            wait_time = 0.0
        if np.isnan(queue_len):
            queue_len = 0.0
        if np.isnan(mean_speed):
            mean_speed = 0.0

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
            total_arrived = 0
            total_departed = 0
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
                        sumo_conn = pz_env.env.sumo if hasattr(pz_env, 'env') else pz_env.sumo
                    else:
                        ts_dict = env_unwrapped.traffic_signals
                        sumo_conn = env_unwrapped.sumo
                    
                    # Track switches
                    for ts_id, ts_obj in ts_dict.items():
                        curr = ts_obj.green_phase
                        if curr != last_green_phases.get(ts_id, curr):
                            total_switches += 1
                            last_green_phases[ts_id] = curr

                    # Accumulate throughput over the full episode.
                    # SUMO getters are per-step counters, not episode totals.
                    total_departed += int(sumo_conn.simulation.getDepartedNumber())
                    info_arrived = info.get("system_arrived_now")
                    if info_arrived is not None:
                        total_arrived += int(info_arrived)
                    else:
                        total_arrived += _arrived_from_sumo_conn(sumo_conn)
                except:
                    total_arrived += _arrived_from_global_traci()

            mean_wait = float(np.mean(wait_times)) if wait_times else 0.0
            mean_queue = float(np.mean(queues)) if queues else 0.0
            mean_speed = float(np.mean(speeds)) if speeds else 0.0

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
