import argparse
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sumo_rl
import os
import pandas as pd
from pathlib import Path
from gymnasium import spaces
import traci
import tempfile

from traffic_rl.callbacks import run_baseline
from traffic_rl.rewards import reward_minimize_queue
from traffic_rl.observations import GridObservationFunction
from traffic_rl.utils import read_summary_arrived, read_summary_metrics, read_tripinfo_metrics

os.environ["LIBSUMO_AS_TRACI"] = "1"


class ParameterSharingEvalWrapper(gym.Env):
    """Clone one policy action across all agents in a parallel PettingZoo env."""

    def __init__(self, pz_env):
        super().__init__()
        self.pz_env = pz_env
        self.agents = []

        self.pz_env.reset()
        if self.pz_env.agents:
            sample_agent = self.pz_env.agents[0]
            self.observation_space = self.pz_env.observation_space(sample_agent)
            self.action_space = self.pz_env.action_space(sample_agent)

    def reset(self, seed=None, options=None):
        observations, infos = self.pz_env.reset(seed=seed, options=options)
        self.agents = list(self.pz_env.agents)
        if self.agents:
            return observations[self.agents[0]], infos
        return None, infos

    def step(self, action):
        if not self.agents:
            return None, 0, True, True, {}

        actions = {}
        for agent in self.agents:
            actions[agent] = self._coerce_action(action, self.pz_env.action_space(agent))

        observations, rewards, terminations, truncations, infos = self.pz_env.step(actions)

        total_reward = sum(rewards.values()) if rewards else 0.0
        avg_reward = total_reward / len(self.agents) if self.agents else 0.0
        done = all(terminations.values()) if terminations else True
        truncated = all(truncations.values()) if truncations else False
        next_obs = observations.get(self.agents[0]) if not done else None
        info = infos.get(self.agents[0], {}) if infos else {}
        try:
            sumo_conn = self.pz_env.env.sumo if hasattr(self.pz_env, "env") else self.pz_env.sumo
            arrived_ids = sumo_conn.simulation.getArrivedIDList()
            info["system_arrived_now"] = len(arrived_ids) if arrived_ids is not None else 0
        except Exception:
            pass
        return next_obs, avg_reward, done, truncated, info

    def close(self):
        self.pz_env.close()

    @staticmethod
    def _coerce_action(action, action_space):
        if isinstance(action_space, spaces.Discrete):
            if isinstance(action, np.ndarray):
                if action.size == 0:
                    return action_space.sample()
                action = action.reshape(-1)[0]
            try:
                value = int(action)
            except Exception:
                return action_space.sample()
            return value if action_space.contains(value) else action_space.sample()

        if isinstance(action_space, spaces.MultiDiscrete):
            try:
                arr = np.asarray(action, dtype=np.int64).reshape(action_space.nvec.shape)
            except Exception:
                return action_space.sample()
            return arr if action_space.contains(arr) else action_space.sample()

        if isinstance(action_space, spaces.MultiBinary):
            try:
                arr = np.asarray(action, dtype=np.int8).reshape(action_space.shape)
            except Exception:
                return action_space.sample()
            arr = np.clip(arr, 0, 1)
            return arr if action_space.contains(arr) else action_space.sample()

        if isinstance(action_space, spaces.Box):
            try:
                arr = np.asarray(action, dtype=action_space.dtype).reshape(action_space.shape)
            except Exception:
                return action_space.sample()
            arr = np.clip(arr, action_space.low, action_space.high)
            return arr if action_space.contains(arr) else action_space.sample()

        return action if action_space.contains(action) else action_space.sample()


def _arrived_from_sumo_conn(sumo_conn):
    """Prefer ID-list based arrivals (more reliable in some multi-agent runs)."""
    try:
        arrived_ids = sumo_conn.simulation.getArrivedIDList()
        if arrived_ids is not None:
            return max(0, int(len(arrived_ids)))
    except Exception:
        pass
    try:
        return max(0, int(sumo_conn.simulation.getArrivedNumber()))
    except Exception:
        return None


def _arrived_from_global_traci():
    """Fallback for cases where env wrappers hide the active TraCI connection."""
    try:
        arrived_ids = traci.simulation.getArrivedIDList()
        if arrived_ids is not None:
            return max(0, int(len(arrived_ids)))
    except Exception:
        pass
    try:
        return max(0, int(traci.simulation.getArrivedNumber()))
    except Exception:
        return None


def _extract_arrived_now(info, env):
    """Best-effort retrieval of arrivals in the current step."""
    arrived_now = info.get("system_arrived_now") if isinstance(info, dict) else None
    if arrived_now is not None:
        try:
            arrived_now = max(0, int(arrived_now))
            if arrived_now > 0:
                return arrived_now
        except Exception:
            pass

    candidates = [env]
    for attr in ("env", "venv", "unwrapped", "pz_env"):
        obj = getattr(env, attr, None)
        if obj is not None:
            candidates.append(obj)
    envs = getattr(env, "envs", None)
    if envs:
        candidates.extend(envs)

    for obj in candidates:
        try:
            if hasattr(obj, "sumo"):
                value = _arrived_from_sumo_conn(obj.sumo)
                if value is not None and value > 0:
                    return value
            if hasattr(obj, "env") and hasattr(obj.env, "sumo"):
                value = _arrived_from_sumo_conn(obj.env.sumo)
                if value is not None and value > 0:
                    return value
            if hasattr(obj, "pz_env"):
                pz_env = obj.pz_env
                if hasattr(pz_env, "sumo"):
                    value = _arrived_from_sumo_conn(pz_env.sumo)
                    if value is not None and value > 0:
                        return value
                if hasattr(pz_env, "env") and hasattr(pz_env.env, "sumo"):
                    value = _arrived_from_sumo_conn(pz_env.env.sumo)
                    if value is not None and value > 0:
                        return value
        except Exception:
            continue

    # Final fallback: query global traci/libsumo handle directly.
    direct_value = _arrived_from_global_traci()
    if direct_value is not None and direct_value > 0:
        return direct_value

    if arrived_now is not None:
        try:
            return max(0, int(arrived_now))
        except Exception:
            pass
    return 0


def evaluate_model(model, env, n_episodes=10, render=False, summary_path=None):
    """
    Evaluate a trained model on the environment.
    
    Args:
        model: Trained RL model
        env: Evaluation environment
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment (GUI)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    episode_metrics = {
        'system_mean_waiting_time': [],
        'system_total_stopped': [],
        'system_mean_speed': [],
        'total_arrived': []
    }
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        episode_arrived = 0
        arrival_probe_logged = False
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            arrived_now = _extract_arrived_now(info, env)
            episode_arrived += arrived_now
            if not arrival_probe_logged and episode_length == 1:
                print(f"  [arrival-probe] first-step arrived_now={arrived_now}")
                arrival_probe_logged = True
        
        summary_arrived = read_summary_arrived(summary_path)
        if summary_arrived is not None:
            episode_arrived = summary_arrived

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_metrics['total_arrived'].append(episode_arrived)
        
        # Extract system-level metrics from final info
        if 'system_mean_waiting_time' in info:
            episode_metrics['system_mean_waiting_time'].append(info['system_mean_waiting_time'])
        if 'system_total_stopped' in info:
            episode_metrics['system_total_stopped'].append(info['system_total_stopped'])
        if 'system_mean_speed' in info:
            episode_metrics['system_mean_speed'].append(info['system_mean_speed'])
        
        print(
            f"Episode {episode + 1}/{n_episodes} - Reward: {episode_reward:.2f}, "
            f"Length: {episode_length}, Arrived: {episode_arrived}"
        )
    
    # Compute statistics
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_waiting_time': np.mean(episode_metrics['system_mean_waiting_time']) if episode_metrics['system_mean_waiting_time'] else None,
        'mean_total_stopped': np.mean(episode_metrics['system_total_stopped']) if episode_metrics['system_total_stopped'] else None,
        'mean_speed': np.mean(episode_metrics['system_mean_speed']) if episode_metrics['system_mean_speed'] else None,
        'mean_total_arrived': np.mean(episode_metrics['total_arrived']) if episode_metrics['total_arrived'] else 0.0,
    }
    
    return results, episode_rewards, episode_metrics


def main(args):
    if args.scenario_dir:
        net_name = args.net_file if args.net_file else "net.xml"
        route_name = args.route_file if args.route_file else "eval.rou.xml"
        NET_FILE = str(Path(args.scenario_dir) / net_name)
        ROUTE_FILE = str(Path(args.scenario_dir) / route_name)
    else:
        NET_FILE = "scenarios/cross_dynamic/cross_turn.net.xml"
        ROUTE_FILE = "scenarios/cross_dynamic/eval.rou.xml"
    
    # Check if model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    print(f"Loading model from: {args.model_path}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Mode: {'multi-agent parameter sharing' if args.multiagent else 'single-agent'}")
    
    # Load the model
    if args.algorithm == "dqn":
        model = DQN.load(args.model_path)
    elif args.algorithm == "ppo":
        model = PPO.load(args.model_path)
    elif args.algorithm == "a2c":
        model = A2C.load(args.model_path)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Create evaluation environment
    tmp_summary = tempfile.NamedTemporaryFile(prefix="evaluate_summary_", suffix=".xml", delete=False)
    summary_path = tmp_summary.name
    tmp_summary.close()
    tmp_tripinfo = tempfile.NamedTemporaryFile(prefix="evaluate_tripinfo_", suffix=".xml", delete=False)
    tripinfo_path = tmp_tripinfo.name
    tmp_tripinfo.close()

    extra_sumo_cmd = f"--summary-output {summary_path} --tripinfo-output {tripinfo_path}"
    if args.additional_sumo_cmd:
        extra_sumo_cmd = f"{extra_sumo_cmd} {args.additional_sumo_cmd}"

    if args.multiagent:
        pz_env = sumo_rl.parallel_env(
            net_file=NET_FILE,
            route_file=ROUTE_FILE,
            use_gui=args.gui,
            num_seconds=args.episode_seconds,
            delta_time=args.delta_time,
            add_system_info=True,
            reward_fn=reward_minimize_queue,
            observation_class=GridObservationFunction,
            sumo_seed=args.seed if args.seed else 'random',
            additional_sumo_cmd=extra_sumo_cmd
        )
        env = ParameterSharingEvalWrapper(pz_env)
    else:
        env = gym.make('sumo-rl-v0',
                       net_file=NET_FILE,
                       route_file=ROUTE_FILE,
                       use_gui=args.gui,
                       num_seconds=args.episode_seconds,
                       delta_time=args.delta_time,
                       add_system_info=True,
                       reward_fn=reward_minimize_queue,
                       observation_class=GridObservationFunction,
                       sumo_seed=args.seed if args.seed else 'random',
                       additional_sumo_cmd=extra_sumo_cmd)
    
    # Load VecNormalize stats if they exist
    if args.normalize:
        normalize_path = Path(args.normalize)
        if normalize_path.exists():
            print(f"Loading VecNormalize stats from: {args.normalize}")
            # Wrap environment for normalization
            env = DummyVecEnv([lambda: env])
            env = VecNormalize.load(args.normalize, env)
            env.training = False  # Don't update stats during evaluation
            env.norm_reward = False
        else:
            print(f"Warning: VecNormalize file not found: {args.normalize}")
    
    # Compute baseline if requested
    baseline_metrics = None
    if args.compare_baseline:
        print("\nComputing baseline metrics...")
        baseline_metrics = run_baseline(
            NET_FILE,
            ROUTE_FILE,
            args.episode_seconds,
            delta_time=args.delta_time
        )
        print(f"Baseline - Mean Waiting Time: {baseline_metrics['mean_waiting_time']:.2f}s")
        print(f"Baseline - Mean Stopped Vehicles: {baseline_metrics['mean_queue_length']:.2f}")
        print(f"Baseline - Mean Speed: {baseline_metrics['mean_speed']:.2f} m/s\n")
    
    # Initialize Wandb if requested
    if args.use_wandb:
        wandb.init(
            entity="fds-final-project",
            project="rl-traffic-light",
            name=f"eval-{args.algorithm}-{args.run_name}" if args.run_name else f"eval-{args.algorithm}",
            config={
                "algorithm": args.algorithm,
                "model_path": str(args.model_path),
                "n_eval_episodes": args.n_episodes,
                "episode_seconds": args.episode_seconds,
                "seed": args.seed,
            }
        )
    
    # Evaluate the model
    print(f"\nEvaluating model for {args.n_episodes} episodes...")
    results, episode_rewards, episode_metrics = evaluate_model(
        model, env, n_episodes=args.n_episodes, render=args.gui, summary_path=summary_path
    )

    # Ensure SUMO has flushed summary output before using it for throughput.
    env.close()
    summary_arrived_final = read_summary_arrived(summary_path)
    summary_metrics = read_summary_metrics(summary_path)
    tripinfo_metrics = read_tripinfo_metrics(tripinfo_path)
    results.update(summary_metrics)
    results.update(tripinfo_metrics)
    if summary_arrived_final is not None and args.n_episodes == 1:
        episode_metrics["total_arrived"] = [summary_arrived_final]
        results["mean_total_arrived"] = float(summary_arrived_final)
        if summary_metrics:
            if episode_metrics["system_mean_waiting_time"]:
                episode_metrics["system_mean_waiting_time"][0] = summary_metrics.get(
                    "summary_mean_waiting_time_end", episode_metrics["system_mean_waiting_time"][0]
                )
            if episode_metrics["system_mean_speed"]:
                episode_metrics["system_mean_speed"][0] = summary_metrics.get(
                    "summary_mean_speed_end", episode_metrics["system_mean_speed"][0]
                )
            if episode_metrics["system_total_stopped"]:
                episode_metrics["system_total_stopped"][0] = summary_metrics.get(
                    "summary_halting_end", episode_metrics["system_total_stopped"][0]
                )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_episode_length']:.1f} steps")
    print(f"Mean Total Arrived (throughput): {results['mean_total_arrived']:.2f}")
    if "summary_total_teleports" in results:
        print(f"Teleports: {results['summary_total_teleports']}")
    if "summary_total_collisions" in results:
        print(f"Collisions: {results['summary_total_collisions']}")
    
    if results['mean_waiting_time'] is not None:
        print(f"\nTraffic Metrics:")
        print(f"  Mean Waiting Time: {results['mean_waiting_time']:.2f}s")
        print(f"  Mean Total Stopped: {results['mean_total_stopped']:.2f}")
        print(f"  Mean Speed: {results['mean_speed']:.2f} m/s")
        
        # Compare with baseline
        if baseline_metrics:
            print(f"\nComparison with Baseline:")
            waiting_improvement = ((baseline_metrics['mean_waiting_time'] - results['mean_waiting_time']) 
                                   / baseline_metrics['mean_waiting_time'] * 100)
            stopped_improvement = ((baseline_metrics['mean_queue_length'] - results['mean_total_stopped']) 
                                   / baseline_metrics['mean_queue_length'] * 100)
            speed_improvement = ((results['mean_speed'] - baseline_metrics['mean_speed']) 
                                / baseline_metrics['mean_speed'] * 100)
            throughput_improvement = ((results['mean_total_arrived'] - baseline_metrics['total_arrived'])
                                     / baseline_metrics['total_arrived'] * 100) if baseline_metrics['total_arrived'] else 0.0
            
            print(f"  Waiting Time: {waiting_improvement:+.1f}%")
            print(f"  Stopped Vehicles: {stopped_improvement:+.1f}%")
            print(f"  Speed: {speed_improvement:+.1f}%")
            print(f"  Throughput (Arrived): {throughput_improvement:+.1f}%")
    
    print("="*60 + "\n")
    
    # Log to Wandb
    if args.use_wandb:
        wandb_log = {
            "eval/mean_reward": results['mean_reward'],
            "eval/std_reward": results['std_reward'],
            "eval/mean_episode_length": results['mean_episode_length'],
        }
        
        if results['mean_waiting_time'] is not None:
            wandb_log.update({
                "eval/mean_waiting_time": results['mean_waiting_time'],
                "eval/mean_total_stopped": results['mean_total_stopped'],
                "eval/mean_speed": results['mean_speed'],
                "eval/mean_total_arrived": results['mean_total_arrived'],
            })
        
        if baseline_metrics:
            wandb_log.update({
                "eval/baseline_waiting_time": baseline_metrics['mean_waiting_time'],
                "eval/baseline_stopped": baseline_metrics['mean_queue_length'],
                "eval/baseline_speed": baseline_metrics['mean_speed'],
                "eval/baseline_total_arrived": baseline_metrics['total_arrived'],
                "eval/waiting_improvement_pct": waiting_improvement,
                "eval/stopped_improvement_pct": stopped_improvement,
                "eval/speed_improvement_pct": speed_improvement,
                "eval/throughput_improvement_pct": throughput_improvement,
            })
        
        wandb.log(wandb_log)
    
    # Save results to CSV if requested
    if args.output_csv:
        output_path = Path(args.output_csv)
        df = pd.DataFrame({
            'episode': range(1, args.n_episodes + 1),
            'reward': episode_rewards,
            'waiting_time': episode_metrics['system_mean_waiting_time'],
            'total_stopped': episode_metrics['system_total_stopped'],
            'mean_speed': episode_metrics['system_mean_speed'],
            'total_arrived': episode_metrics['total_arrived'],
        })
        # Explicit aggregate evaluation means (repeated per row for convenience).
        df["eval_mean_waiting_time"] = results.get("mean_waiting_time")
        df["eval_mean_speed"] = results.get("mean_speed")
        df["eval_mean_queue_length"] = results.get("mean_total_stopped")

        for k, v in {**summary_metrics, **tripinfo_metrics}.items():
            df[k] = v
        if baseline_metrics:
            df["baseline_mean_waiting_time"] = baseline_metrics.get("mean_waiting_time")
            df["baseline_mean_speed"] = baseline_metrics.get("mean_speed")
            df["baseline_mean_queue_length"] = baseline_metrics.get("mean_queue_length")
            df["baseline_total_arrived"] = baseline_metrics["total_arrived"]
            df["throughput_improvement_pct"] = throughput_improvement
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    # Cleanup
    try:
        Path(summary_path).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        Path(tripinfo_path).unlink(missing_ok=True)
    except Exception:
        pass
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained RL traffic light controller models"
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the pretrained model (.zip file)"
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        required=True,
        choices=["dqn", "ppo", "a2c"],
        help="Algorithm used to train the model"
    )
    
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=3600,
        help="Duration of each episode in seconds"
    )

    parser.add_argument(
        "--delta-time",
        type=int,
        default=5,
        help="SUMO control step in seconds (must be > yellow_time; use training value for fair comparison)"
    )

    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Scenario directory containing net/route files"
    )

    parser.add_argument(
        "--net-file",
        type=str,
        default=None,
        help="Network file name (relative to --scenario-dir)"
    )

    parser.add_argument(
        "--route-file",
        type=str,
        default=None,
        help="Route file name (relative to --scenario-dir)"
    )

    parser.add_argument(
        "--multiagent",
        action="store_true",
        help="Clone one policy action across all traffic lights in a parallel env"
    )
    
    parser.add_argument(
        "--normalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats file (.pkl)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for evaluation (for reproducibility)"
    )
    
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI visualization"
    )
    
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Compare results with baseline (fixed time traffic light)"
    )
    
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log evaluation results to Weights & Biases"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this evaluation run (for Wandb)"
    )
    
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to save evaluation results as CSV"
    )

    parser.add_argument(
        "--additional-sumo-cmd",
        type=str,
        default=None,
        help="Extra SUMO CLI args (e.g. \"--delay 5\")"
    )
    
    args = parser.parse_args()
    main(args)
