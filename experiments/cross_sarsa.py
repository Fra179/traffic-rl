import argparse
import os
import sys

import matplotlib.pyplot as plt

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from sumo_rl import SumoEnvironment

import os

os.environ["LIBSUMO_AS_TRACI"] = "1"

class LivePlotter:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.steps = []
        self.rewards, self.ema_rewards = [], []
        self.waits, self.ema_waits = [], []
        self.queues, self.ema_queues = [], []
        self.speeds, self.ema_speeds = [], []
        self.total_cars, self.ema_total_cars = [], []

        plt.ion()
        self.fig, self.axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
        self.fig.suptitle(f"SARSA Lambda - Minimize Queue Length | EMA={self.alpha}")

        labels = ["Reward", "Wait (s)", "Queue Length", "Speed (m/s)", "Total Cars"]
        self.lines_raw = []
        self.lines_ema = []

        colors = ["blue", "red", "orange", "green", "purple"]
        for i, ax in enumerate(self.axs):
            ax.set_ylabel(labels[i])
            raw_line, = ax.plot([], [], color="gray", alpha=0.2, linewidth=1)
            ema_line, = ax.plot([], [], color=colors[i], alpha=1.0, linewidth=1.5, label="EMA")
            self.lines_raw.append(raw_line)
            self.lines_ema.append(ema_line)
            if i == 0:
                ax.legend(loc="upper left")

        self.axs[-1].set_xlabel("Steps")

    def _calc_ema(self, new_val, history_list):
        if not history_list:
            return new_val
        return (self.alpha * new_val) + ((1 - self.alpha) * history_list[-1])

    def update(self, step, reward, info):
        wait_time = info.get("system_total_waiting_time", 0)
        queue_len = info.get("system_total_stopped", 0)
        mean_speed = info.get("system_mean_speed", 0)

        try:
            total_vehicles = traci.vehicle.getIDCount()
        except Exception:
            total_vehicles = 0

        ema_reward = self._calc_ema(reward, self.ema_rewards)
        ema_wait = self._calc_ema(wait_time, self.ema_waits)
        ema_queue = self._calc_ema(queue_len, self.ema_queues)
        ema_speed = self._calc_ema(mean_speed, self.ema_speeds)
        ema_total = self._calc_ema(total_vehicles, self.ema_total_cars)

        self.steps.append(step)
        self.rewards.append(reward)
        self.ema_rewards.append(ema_reward)
        self.waits.append(wait_time)
        self.ema_waits.append(ema_wait)
        self.queues.append(queue_len)
        self.ema_queues.append(ema_queue)
        self.speeds.append(mean_speed)
        self.ema_speeds.append(ema_speed)
        self.total_cars.append(total_vehicles)
        self.ema_total_cars.append(ema_total)

        if step % 50 == 0:
            self._render()

    def _render(self):
        data_pairs = [
            (self.rewards, self.ema_rewards),
            (self.waits, self.ema_waits),
            (self.queues, self.ema_queues),
            (self.speeds, self.ema_speeds),
            (self.total_cars, self.ema_total_cars),
        ]

        for i, (raw, ema) in enumerate(data_pairs):
            self.lines_raw[i].set_data(self.steps, raw)
            self.lines_ema[i].set_data(self.steps, ema)
            self.axs[i].relim()
            self.axs[i].autoscale_view()

        plt.draw()
        plt.pause(0.001)

    def close(self):
        plt.close(self.fig)


def reward_minimize_queue(ts):
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
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""True Online SARSA Lambda - Cross Intersection""",
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="scenarios/cross/cross.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-a", dest="alpha", type=float, default=0.0001, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.95, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.01, required=False, help="Epsilon.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=3600, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=100, help="Number of runs.\n")
    prs.add_argument("-plot", action="store_true", default=True, help="Enable live plotting.\n")
    args = prs.parse_args()

    out_csv = "outputs/cross_sarsa_run"

    env = SumoEnvironment(
        net_file="scenarios/cross/cross.net.xml",
        single_agent=True,
        route_file=args.route,
        out_csv_name=out_csv,
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        reward_fn=reward_minimize_queue,
        add_system_info=True,
    )

    agent = TrueOnlineSarsaLambda(
        env.observation_space,
        env.action_space,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        fourier_order=7,
        lamb=0.95,
    )

    plotter = LivePlotter(alpha=0.05) if args.plot else None
    total_steps = 0

    for run in range(1, args.runs + 1):
        obs, info = env.reset()
        step_count = 0

        terminated, truncated = False, False
        if args.fixed:
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step({})
        else:
            while not (terminated or truncated):
                action = agent.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(action=action)
                agent.learn(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=terminated,
                )
                obs = next_obs
                step_count += 1
                total_steps += 1
                if plotter:
                    plotter.update(total_steps, reward, info)

        env.save_csv(out_csv, run)
    
    if plotter:
        plotter.close()

    env.close()

