import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl

# --- WAYLAND/DISPLAY FIX ---
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('TkAgg')

# --- METRICS CALLBACK ---
class LivePlotCallback(BaseCallback):
    def __init__(self, alpha=0.05, verbose=0):
        super(LivePlotCallback, self).__init__(verbose)
        self.alpha = alpha
        
        self.steps = []
        self.rewards, self.ema_rewards = [], []
        self.waits, self.ema_waits = [], []
        self.queues, self.ema_queues = [], []
        self.speeds, self.ema_speeds = [], []      
        self.total_cars, self.ema_total_cars = [], [] 
        
        plt.ion()
        self.fig, self.axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
        self.fig.suptitle(f'DQN - Minimize Queue Length | EMA={self.alpha}')
        
        labels = ['Reward (Queue)', 'Wait (s)', 'Queue Length', 'Speed (m/s)', 'Total Cars']
        self.lines_raw = []
        self.lines_ema = []
        
        for i, ax in enumerate(self.axs):
            ax.set_ylabel(labels[i])
            l_raw, = ax.plot([], [], color='gray', alpha=0.2, linewidth=1)
            color = ['blue', 'red', 'orange', 'green', 'purple'][i]
            l_ema, = ax.plot([], [], color=color, alpha=1.0, linewidth=1.5, label='EMA')
            self.lines_raw.append(l_raw)
            self.lines_ema.append(l_ema)
            if i == 0: ax.legend(loc='upper left')

        self.axs[-1].set_xlabel('Steps')

    def _calc_ema(self, new_val, history_list):
        if not history_list: return new_val
        return (self.alpha * new_val) + ((1 - self.alpha) * history_list[-1])

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]
        dones = self.locals['dones'][0]
        reward = self.locals['rewards'][0] # Normalized Reward
        
        wait_time = infos.get('system_total_waiting_time', 0)
        queue_len = infos.get('system_total_stopped', 0)
        mean_speed = infos.get('system_mean_speed', 0)
        
        try:
            sumo_conn = self.training_env.envs[0].unwrapped.sumo
            total_vehicles = sumo_conn.vehicle.getIDCount()
        except:
            total_vehicles = 0

        ema_reward = self._calc_ema(reward, self.ema_rewards)
        ema_wait = self._calc_ema(wait_time, self.ema_waits)
        ema_queue = self._calc_ema(queue_len, self.ema_queues)
        ema_speed = self._calc_ema(mean_speed, self.ema_speeds)
        ema_total = self._calc_ema(total_vehicles, self.ema_total_cars)

        self.steps.append(self.num_timesteps)
        self.rewards.append(reward); self.ema_rewards.append(ema_reward)
        self.waits.append(wait_time); self.ema_waits.append(ema_wait)
        self.queues.append(queue_len); self.ema_queues.append(ema_queue)
        self.speeds.append(mean_speed); self.ema_speeds.append(ema_speed)
        self.total_cars.append(total_vehicles); self.ema_total_cars.append(ema_total)

        if dones:
            for ax in self.axs:
                ax.axvline(x=self.num_timesteps, color='k', linestyle='--', alpha=0.3)

        if self.num_timesteps % 50 == 0:
            self._update_plot()
        return True

    def _update_plot(self):
        data_pairs = [
            (self.rewards, self.ema_rewards),
            (self.waits, self.ema_waits),
            (self.queues, self.ema_queues),
            (self.speeds, self.ema_speeds),
            (self.total_cars, self.ema_total_cars)
        ]
        for i, (raw, ema) in enumerate(data_pairs):
            self.lines_raw[i].set_data(self.steps, raw)
            self.lines_ema[i].set_data(self.steps, ema)
            self.axs[i].relim()
            self.axs[i].autoscale_view()
        plt.draw()
        plt.pause(0.001)

# --- REWARD FUNCTION ---
def reward_minimize_queue(ts):
    # Directly penalize the number of stopped cars
    return -float(ts.get_total_queued())

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
    

    # 1. Setup Environment
    env = gym.make('sumo-rl-v0',
                   net_file=NET_FILE,
                   route_file=ROUTE_FILE,
                   out_csv_name="outputs/dqn_queue_run",
                   use_gui=False,          
                   num_seconds=EPISODE_SECONDS, 
                   add_system_info=True,
                   reward_fn=reward_minimize_queue) # <--- USING NEW REWARD

    env = DummyVecEnv([lambda: env])
    
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
    
    try:
        model.learn(total_timesteps=50000, callback=LivePlotCallback(alpha=0.05))
    finally:
        model.save("dqn_queue_model")
        env.save("dqn_queue_vec_normalize.pkl")

    print("Done.")
    plt.ioff()
    plt.show()