import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import sumo_rl
import os

class LivePlotCallback(BaseCallback):
    """
    Plots metrics with EMA smoothing and Vertical Lines for Episode Resets.
    """
    def __init__(self, alpha=0.05, verbose=0):
        super(LivePlotCallback, self).__init__(verbose)
        self.alpha = alpha
        
        # Data Buffers
        self.steps = []
        self.rewards, self.ema_rewards = [], []
        self.waits, self.ema_waits = [], []
        self.queues, self.ema_queues = [], []
        
        # Setup the plot
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        self.fig.suptitle(f'Live Metrics (EMA={self.alpha}) | Vertical Line = Reset')
        
        self.axs[0].set_ylabel('Reward')
        self.axs[1].set_ylabel('System Wait (s)')
        self.axs[2].set_ylabel('Queue Length')
        self.axs[2].set_xlabel('Steps')

        # Initialize Lines
        self.line_reward_raw, = self.axs[0].plot([], [], 'b-', alpha=0.15, label='Raw')
        self.line_reward_ema, = self.axs[0].plot([], [], 'b-', alpha=1.0, linewidth=1.5, label='EMA')
        
        self.line_wait_raw, = self.axs[1].plot([], [], 'r-', alpha=0.15)
        self.line_wait_ema, = self.axs[1].plot([], [], 'r-', alpha=1.0, linewidth=1.5)
        
        self.line_queue_raw, = self.axs[2].plot([], [], 'g-', alpha=0.15)
        self.line_queue_ema, = self.axs[2].plot([], [], 'g-', alpha=1.0, linewidth=1.5)

        self.axs[0].legend(loc='upper left')

    def _calc_ema(self, new_val, history_list):
        if not history_list:
            return new_val
        return (self.alpha * new_val) + ((1 - self.alpha) * history_list[-1])

    def _on_step(self) -> bool:
        infos = self.locals['infos'][0]
        dones = self.locals['dones'][0] # Check if episode finished
        
        wait_time = infos.get('system_total_waiting_time', 0)
        queue_len = infos.get('system_total_stopped', 0)
        reward = self.locals['rewards'][0]

        # Calculate EMAs
        ema_reward = self._calc_ema(reward, self.ema_rewards)
        ema_wait = self._calc_ema(wait_time, self.ema_waits)
        ema_queue = self._calc_ema(queue_len, self.ema_queues)

        self.steps.append(self.num_timesteps)
        self.rewards.append(reward)
        self.ema_rewards.append(ema_reward)
        self.waits.append(wait_time)
        self.ema_waits.append(ema_wait)
        self.queues.append(queue_len)
        self.ema_queues.append(ema_queue)

        # Draw a vertical line if the environment reset (Episode End)
        if dones:
            for ax in self.axs:
                ax.axvline(x=self.num_timesteps, color='k', linestyle='--', alpha=0.3)

        if self.num_timesteps % 50 == 0:
            self._update_plot()

        return True

    def _update_plot(self):
        self.line_reward_raw.set_data(self.steps, self.rewards)
        self.line_reward_ema.set_data(self.steps, self.ema_rewards)
        
        self.line_wait_raw.set_data(self.steps, self.waits)
        self.line_wait_ema.set_data(self.steps, self.ema_waits)
        
        self.line_queue_raw.set_data(self.steps, self.queues)
        self.line_queue_ema.set_data(self.steps, self.ema_queues)
        
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
            
        plt.draw()
        plt.pause(0.001)

if __name__ == "__main__":

    os.environ["LIBSUMO_AS_TRACI"] = "1"
    # Configuration
    NET_FILE = "scenarios/cross/cross.net.xml"  # Make sure this exists!
    ROUTE_FILE = "scenarios/cross/cross.rou.xml"
    TOTAL_SECONDS = 10000
    
    # 1. Generate Traffic

    # 2. Setup Environment
    # We use 'SumoEnvironment-v0' directly or via gym.make with careful args
    env = gym.make('sumo-rl-v0',
                   net_file=NET_FILE,
                   route_file=ROUTE_FILE,
                   out_csv_name="outputs/ppo_run",
                   use_gui=False,          # Set False to speed up training
                   num_seconds=TOTAL_SECONDS,
                   add_system_info=True)  # CRITICAL for stats extraction!

    env = DummyVecEnv([lambda: env])

    # 3. Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.99
    )

    # 4. Train
    print("Starting PPO training with live plotting...")
    try:
        model.learn(total_timesteps=TOTAL_SECONDS, callback=LivePlotCallback())
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    print("Done.")
    plt.ioff()
    plt.show()