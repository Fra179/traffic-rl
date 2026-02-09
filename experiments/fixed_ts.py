import gymnasium as gym
import sumo_rl
import os
import sys

def run_fixed_ts(net_file, route_file, out_csv_name, episode_seconds=3600, use_gui=False, sumo_seed="random"):
    """
    Runs a simulation with fixed traffic signal policies (using the logic in .rou.xml).
    This serves as a baseline to compare learned policies against.
    
    Args:
        net_file: Path to .net.xml
        route_file: Path to .rou.xml
        out_csv_name: Base name for output CSV stats
        episode_seconds: Duration of simulation
        use_gui: Whether to show SUMO GUI
        sumo_seed: Seed for traffic generation
    """
    print(f"Starting Fixed-TS Baseline Simulation...")
    print(f"Net: {net_file}")
    print(f"Route: {route_file}")
    
    # Check if files exist to avoid obscure SUMO errors
    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Net file not found: {net_file}")
    if not os.path.exists(route_file):
        raise FileNotFoundError(f"Route file not found: {route_file}")

    # fixed_ts=True tells SUMO-RL to ignore step() actions and follow the defined program
    env = gym.make('sumo-rl-v0',
                   net_file=net_file,
                   route_file=route_file,
                   out_csv_name=out_csv_name,
                   use_gui=use_gui,
                   num_seconds=episode_seconds,
                   add_system_info=True,
                   add_per_agent_info=True,
                   fixed_ts=True,
                   sumo_seed=sumo_seed)

    obs, info = env.reset()
    done = False
    truncated = False
    step = 0
    
    while not (done or truncated):
        # Actions are ignored in fixed_ts=True mode, but we must step the env
        action = env.action_space.sample()  
        obs, reward, done, truncated, info = env.step(action)
        step += 1
        
    # Explicitly save metrics for the completed episode
    # env.unwrapped accesses the base SumoEnvironment
    if hasattr(env.unwrapped, 'save_csv'):
        env.unwrapped.save_csv(out_csv_name, 1)

    env.close()
    print(f"Simulation finished after {step} steps.")
    print(f"Metrics should be saved to something like {out_csv_name}_runX.csv")

if __name__ == "__main__":
    # Determine paths based on CWD
    # We assume 'scenarios' is in the project root.
    
    cwd = os.getcwd()
    if os.path.basename(cwd) == "experiments":
        # If running from experiments folder
        base_path = ".."
    else:
        # If running from root
        base_path = "."

    NET_FILE = os.path.join(base_path, "scenarios/cross/cross.net.xml")
    ROUTE_FILE = os.path.join(base_path, "scenarios/cross/cross.rou.xml")
    OUT_DIR = os.path.join(base_path, "outputs")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    OUTPUT_NAME = os.path.join(OUT_DIR, "fixed_ts_baseline")
    
    # Use same settings as cross_dqn.py for fair comparison
    EPISODE_SECONDS = 3600
    
    run_fixed_ts(
        net_file=NET_FILE, 
        route_file=ROUTE_FILE, 
        out_csv_name=OUTPUT_NAME, 
        episode_seconds=EPISODE_SECONDS,
        use_gui=True
    )
