import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl
import os
from utils import TrafficWandbCallback, ValidationCallback, run_baseline, reward_minimize_queue

os.environ["LIBSUMO_AS_TRACI"] = "1"


# --- REWARD FUNCTION ---
# Imported from utils.py

def reward_vidali_waiting_time(ts):
    """
    Reward = Negative sum of the accumulated waiting time of all vehicles.
    Matches the 'Deep-QLearning-Agent' repository logic.
    """
    # 1. Get dictionary of wait times {lane_id: wait_time}
    wait_times = ts.get_accumulated_waiting_time_per_lane()
    
    # 2. Sum them all up and negate
    total_wait = sum(wait_times)
    
    return -float(total_wait) 

if __name__ == "__main__":

    NET_FILE = "scenarios/cross/cross.net.xml"  
    ROUTE_FILE = "scenarios/cross/cross.rou.xml"
    EPISODE_SECONDS = 3600
    TOTAL_SECONDS = 50000
    
    # 0. Compute Baseline First
    baseline_metrics = run_baseline(NET_FILE, ROUTE_FILE, EPISODE_SECONDS)

    # 1. Setup Environment
    # We use 'SumoEnvironment-v0' directly or via gym.make with careful args
    env = gym.make('sumo-rl-v0',
                   net_file=NET_FILE,
                   route_file=ROUTE_FILE,
                   out_csv_name="outputs/ppo_run",
                   use_gui=False,          # Set False to speed up training
                   num_seconds=EPISODE_SECONDS,
                   add_system_info=True,
                   reward_fn=reward_minimize_queue)  # CRITICAL for stats extraction!

    env = DummyVecEnv([lambda: env])

    # Setup Evaluation Environment (Separate instance)
    eval_env = gym.make('sumo-rl-v0',
                   net_file=NET_FILE,
                   route_file=ROUTE_FILE,
                   use_gui=False,          
                   num_seconds=EPISODE_SECONDS, 
                   add_system_info=True,
                   reward_fn=reward_minimize_queue,
                   sumo_seed='42') 

    # 3. Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99
    )

    print("Starting PPO training...")
    
    wandb.init(
        entity="fds-final-project",
        project="rl-traffic-light",
        name="ppo-cross-queue",
        config={
            "net_file": NET_FILE,
            "route_file": ROUTE_FILE,
            "episode_seconds": EPISODE_SECONDS,
            "total_timesteps": TOTAL_SECONDS,
            "learning_rate": 0.0003,
            "baseline_metrics": baseline_metrics
        }
    )
    
     # Calculate eval_freq based on assumption of 5s delta_time
    # If delta_time is different, adjust this. default is 5.
    STEPS_PER_EPISODE = EPISODE_SECONDS // 5 
    
    # Combine callbacks
    callbacks = [
        TrafficWandbCallback(),
        ValidationCallback(eval_env, baseline_metrics, eval_freq=STEPS_PER_EPISODE)
    ]
    
    # 4. Train
    try:
        model.learn(total_timesteps=TOTAL_SECONDS, callback=callbacks)
    finally:
        model.save("ppo_queue_model")
        eval_env.close() 
        wandb.finish()
    
    print("Done.")
