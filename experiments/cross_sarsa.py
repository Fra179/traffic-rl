import argparse
import os
import sys

import numpy as np
import wandb

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from sumo_rl import SumoEnvironment
from utils import run_baseline, reward_minimize_queue

import os

os.environ["LIBSUMO_AS_TRACI"] = "1"


def run_validation(agent, eval_env, baseline_metrics, step):
    print(f"Running Validation at step {step}...")
    
    # Save original epsilon and set to 0 for greedy evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    obs, info = eval_env.reset()
    done = False
    truncated = False
    
    wait_times = []
    queues = []
    speeds = []
    
    while not (done or truncated):
        action = agent.act(obs)
        obs, reward, done, truncated, info = eval_env.step(action)
        
        wait_times.append(info.get('system_total_waiting_time', 0))
        queues.append(info.get('system_total_stopped', 0))
        speeds.append(info.get('system_mean_speed', 0))

    mean_wait = np.mean(wait_times)
    mean_queue = np.mean(queues)
    mean_speed = np.mean(speeds)

    # Log comparison
    wandb.log({
        "validation/step": step,
        "validation/mean_waiting_time": mean_wait,
        "validation/mean_queue_length": mean_queue,
        "validation/mean_speed": mean_speed,
        
        "baseline/mean_waiting_time": baseline_metrics['mean_waiting_time'],
        "baseline/mean_queue_length": baseline_metrics['mean_queue_length'],
        "baseline/mean_speed": baseline_metrics['mean_speed'],
    })
    
    print(f"Validation Complete. Speed: {mean_speed:.2f} vs Baseline: {baseline_metrics['mean_speed']:.2f}")
    
    # Restore epsilon
    agent.epsilon = original_epsilon



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
    prs.add_argument("-runs", dest="runs", type=int, default=300, help="Number of runs.\n") # Increased runs default 
    args = prs.parse_args()

    # 1. Compute Baseline
    baseline_metrics = run_baseline("scenarios/cross/cross.net.xml", args.route, args.seconds)

    # 2. Wandb Init
    wandb.init(
        entity="fds-final-project",
        project="rl-traffic-light",
        name="sarsa-cross-queue",
        config=vars(args)
    )
    
    # Add baseline metrics to config
    wandb.config.update({"baseline_metrics": baseline_metrics})

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

    # Setup Eval Env
    eval_env = SumoEnvironment(
        net_file="scenarios/cross/cross.net.xml",
        single_agent=True,
        route_file=args.route,
        use_gui=False,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        reward_fn=reward_minimize_queue,
        add_system_info=True,
        sumo_seed='42'
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

    total_steps = 0
    # Calculate eval freq: We want to validate roughly every episode or fraction of it. 
    # Let's say once per episode for now, or every 720 steps as in DQN/PPO.
    # EPISODE_STEPS = args.seconds // 5.
    eval_freq = 720 

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
                
                # Logging
                step_count += 1
                total_steps += 1
                
                wait_time = info.get("system_total_waiting_time", 0)
                queue_len = info.get("system_total_stopped", 0)
                mean_speed = info.get("system_mean_speed", 0)
                try:
                    total_vehicles = traci.vehicle.getIDCount()
                except:
                    total_vehicles = 0
                
                wandb.log({
                    "step": total_steps,
                    "reward": reward,
                    "waiting_time": wait_time,
                    "queue_length": queue_len,
                    "mean_speed": mean_speed,
                    "total_vehicles": total_vehicles,
                })
                
                # Validation check
                if total_steps > 0 and total_steps % eval_freq == 0:
                     run_validation(agent, eval_env, baseline_metrics, total_steps)

        env.save_csv(out_csv, run)

    # finish
    eval_env.close()
    env.close()
    wandb.finish()


