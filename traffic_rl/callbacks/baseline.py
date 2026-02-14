"""Baseline computation utilities"""

import numpy as np
import gymnasium as gym
import sumo_rl
import tempfile
from pathlib import Path
from traffic_rl.utils import read_summary_arrived


def run_baseline(net_file, route_file, num_seconds, delta_time=5):
    """
    Run a fixed-timing baseline simulation to establish performance metrics.
    
    Args:
        net_file: Path to SUMO network file
        route_file: Path to SUMO route file
        num_seconds: Duration of simulation
        delta_time: SUMO-RL control step in seconds
        
    Returns:
        dict: Baseline metrics including waiting time, queue length, speed, etc.
    """
    print("Computing Fixed-TS Baseline...")
    
    tmp_summary = tempfile.NamedTemporaryFile(prefix="baseline_summary_", suffix=".xml", delete=False)
    tmp_summary_path = tmp_summary.name
    tmp_summary.close()

    # Use fixed_ts=True for baseline
    env = gym.make('sumo-rl-v0',
                   net_file=net_file,
                   route_file=route_file,
                   num_seconds=num_seconds,
                   delta_time=delta_time,
                   use_gui=False,
                   fixed_ts=True,
                   sumo_seed='42',  # Fixed seed for consistency
                   additional_sumo_cmd=f"--summary-output {tmp_summary_path}")
    
    obs, info = env.reset()
    done = False
    truncated = False
    
    wait_times = []
    queues = []
    speeds = []
    total_arrived = 0
    total_switches = 0
    
    # Init phase tracking
    try:
        ts_dict = env.unwrapped.traffic_signals
        last_green_phases = {ts: ts_obj.green_phase for ts, ts_obj in ts_dict.items()}
    except:
        last_green_phases = {}
    
    while not (done or truncated):
        action = env.action_space.sample()  # Ignored in fixed_ts mode
        obs, reward, done, truncated, info = env.step(action)
        
        wait_times.append(info.get('system_total_waiting_time', 0))
        queues.append(info.get('system_total_stopped', 0))
        speeds.append(info.get('system_mean_speed', 0))
        
        try:
            total_arrived += env.unwrapped.sumo.simulation.getArrivedNumber()
            
            # Track switches
            ts_dict = env.unwrapped.traffic_signals
            for ts_id, ts_obj in ts_dict.items():
                curr = ts_obj.green_phase
                if curr != last_green_phases.get(ts_id, curr):
                    total_switches += 1
                    last_green_phases[ts_id] = curr
        except:
            pass
    
    env.close()

    # Prefer robust cumulative arrived count from SUMO summary output.
    parsed_arrived = read_summary_arrived(tmp_summary_path)
    if parsed_arrived is not None:
        total_arrived = parsed_arrived
    try:
        Path(tmp_summary_path).unlink(missing_ok=True)
    except Exception:
        pass
    
    metrics = {
        "mean_waiting_time": np.mean(wait_times),
        "mean_queue_length": np.mean(queues),
        "mean_speed": np.mean(speeds),
        "total_arrived": total_arrived,
        "total_switches": total_switches
    }
    
    print(f"Baseline Computed: {metrics}")
    return metrics

if __name__ == "__main__":
    # Example usage
    NET_FILE = "scenarios/cross_dynamic/cross.net.xml"
    ROUTE_FILE = "scenarios/cross_dynamic/eval.rou.xml"
    EPISODE_SECONDS = 5400  # 1 hour
    
    baseline_metrics = run_baseline(NET_FILE, ROUTE_FILE, EPISODE_SECONDS)
    print(f"Baseline Metrics: {baseline_metrics}")
