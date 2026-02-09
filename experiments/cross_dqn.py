import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN
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
    TRAFFIC_BUFFER = 1.05
    TOTAL_SECONDS = 50000
    
    # 0. Compute Baseline First
    # We use the same parameters as training, but with fixed_ts=True
    baseline_metrics = run_baseline(NET_FILE, ROUTE_FILE, EPISODE_SECONDS)

    # 1. Setup Training Environment
    env = gym.make('sumo-rl-v0',
                   net_file=NET_FILE,
                   route_file=ROUTE_FILE,
                   out_csv_name="outputs/dqn_queue_run",
                   use_gui=False,          
                   num_seconds=EPISODE_SECONDS, 
                   add_system_info=True,
                   reward_fn=reward_minimize_queue) 

    env = DummyVecEnv([lambda: env])
    
    # Setup Evaluation Environment (Separate instance)
    # We use a fixed seed '42' to be consistent with baseline for fair comparison
    eval_env = gym.make('sumo-rl-v0',
                   net_file=NET_FILE,
                   route_file=ROUTE_FILE,
                   use_gui=False,          
                   num_seconds=EPISODE_SECONDS, 
                   add_system_info=True,
                   reward_fn=reward_minimize_queue,
                   sumo_seed='42') 
    
    # Normalization is crucial here! 
    # Queue length can be -50 or -100. We need to scale this down for DQN.
    # env = VecNormalize(env, norm_obs=True)

    # 2. DQN Agent
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,
        buffer_size=50000,         # Memory size
        learning_starts=1000,      
        batch_size=64,             
        target_update_interval=500,
        train_freq=4,              
        gradient_steps=1,          
        exploration_fraction=0.2,  
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05
    )

    print("Starting DQN (Minimize Queue)...")
    
    wandb.init(
        entity="fds-final-project",
        project="rl-traffic-light",
        name="dqn-cross-queue",
        config={
            "net_file": NET_FILE,
            "route_file": ROUTE_FILE,
            "episode_seconds": EPISODE_SECONDS,
            "total_timesteps": TOTAL_SECONDS,
            "learning_rate": 0.0005,
            "batch_size": 64,
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

    try:
        model.learn(total_timesteps=TOTAL_SECONDS, callback=callbacks)
    finally:
        model.save("dqn_queue_model")
        env.save("dqn_queue_vec_normalize.pkl")
        eval_env.close() # Close eval env
        wandb.finish()

    print("Done.")