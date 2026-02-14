import argparse
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import sumo_rl
import os
from gymnasium import spaces

from traffic_rl.callbacks import TrafficWandbCallback, ValidationCallback, run_baseline
from traffic_rl.rewards import reward_minimize_queue, reward_vidali_waiting_time, reward_minimize_max_queue
from traffic_rl.observations import GridObservationFunction
from traffic_rl.utils import detect_route_duration_seconds

os.environ["LIBSUMO_AS_TRACI"] = "1"


class PettingZooToGymWrapper(gym.Env):
    """
    Wrapper to convert PettingZoo parallel environment to Gym environment.
    Uses parameter sharing: all agents share the same policy.
    """
    
    def __init__(self, pz_env, warmup_steps=0):
        super().__init__()
        self.pz_env = pz_env
        self.warmup_steps = max(0, int(warmup_steps))
        self.agents = []
        self.cumulative_switches = 0
        self.last_green_phases = {}
        
        # Get a sample agent to determine observation and action spaces
        self.pz_env.reset()
        if self.pz_env.agents:
            sample_agent = self.pz_env.agents[0]
            self.observation_space = self.pz_env.observation_space(sample_agent)
            self.action_space = self.pz_env.action_space(sample_agent)
        
        self.episode_rewards = {}
        
    def reset(self, seed=None, options=None):
        """Reset the environment and return the first agent's observation."""
        observations, infos = self.pz_env.reset(seed=seed, options=options)
        self.agents = list(self.pz_env.agents)
        self.episode_rewards = {agent: 0 for agent in self.agents}
        self.last_green_phases = self._get_current_green_phases()

        # Optional warmup: advance simulation before learning starts so the
        # network is not always observed from a cold/empty initial condition.
        for _ in range(self.warmup_steps):
            if not self.agents:
                break
            warmup_actions = {}
            for agent in self.agents:
                agent_space = self.pz_env.action_space(agent)
                warmup_actions[agent] = self._coerce_action(0, agent_space)
            observations, _, terminations, truncations, infos = self.pz_env.step(warmup_actions)
            if terminations and all(terminations.values()):
                break
            if truncations and all(truncations.values()):
                break
        
        if self.agents:
            return observations[self.agents[0]], infos
        return None, infos
    
    def step(self, action):
        """
        Execute action for all agents (parameter sharing).
        """
        if not self.agents:
            return None, 0, True, True, {}
        
        # Use same action for all agents (parameter sharing), but cast to each
        # agent's exact action space type/shape expected by PettingZoo.
        actions = {}
        for agent in self.agents:
            agent_space = self.pz_env.action_space(agent)
            actions[agent] = self._coerce_action(action, agent_space)
        
        # Step the PettingZoo environment
        observations, rewards, terminations, truncations, infos = self.pz_env.step(actions)
        self._update_switch_count()
        
        # Calculate aggregate reward
        total_reward = sum(rewards.values()) if rewards else 0
        avg_reward = total_reward / len(self.agents) if self.agents else 0
        
        # Check if episode is done
        done = all(terminations.values()) if terminations else True
        truncated = all(truncations.values()) if truncations else False
        
        # Return first agent's observation
        next_obs = observations.get(self.agents[0]) if not done else None
        
        info = infos.get(self.agents[0], {}) if infos else {}
        try:
            sumo_conn = self.pz_env.env.sumo if hasattr(self.pz_env, "env") else self.pz_env.sumo
            info["system_total_vehicles"] = int(sumo_conn.vehicle.getIDCount())
            arrived_ids = sumo_conn.simulation.getArrivedIDList()
            info["system_arrived_now"] = len(arrived_ids) if arrived_ids is not None else 0
        except Exception:
            pass
        info['total_reward'] = total_reward
        info['avg_reward'] = avg_reward
        
        return next_obs, avg_reward, done, truncated, info

    def get_sumo_metrics(self):
        """Return SUMO metrics as plain data for cross-process callbacks."""
        total_vehicles = 0
        arrived_now = 0
        try:
            sumo_conn = self.pz_env.env.sumo if hasattr(self.pz_env, "env") else self.pz_env.sumo
            total_vehicles = sumo_conn.vehicle.getIDCount()
            arrived_ids = sumo_conn.simulation.getArrivedIDList()
            arrived_now = len(arrived_ids) if arrived_ids is not None else 0
        except Exception:
            pass
        return {
            "total_vehicles": total_vehicles,
            "arrived_now": arrived_now,
            "cumulative_switches": self.cumulative_switches,
        }

    def _get_current_green_phases(self):
        try:
            if hasattr(self.pz_env, "env"):
                ts_ids = self.pz_env.env.ts_ids
                ts_dict = {ts_id: self.pz_env.env._traffic_signals[ts_id] for ts_id in ts_ids}
            else:
                ts_ids = self.pz_env.ts_ids
                ts_dict = {ts_id: self.pz_env._traffic_signals[ts_id] for ts_id in ts_ids}
            return {ts_id: ts_obj.green_phase for ts_id, ts_obj in ts_dict.items()}
        except Exception:
            return {}

    def _update_switch_count(self):
        current_green_phases = self._get_current_green_phases()
        if not current_green_phases:
            return
        if not self.last_green_phases:
            self.last_green_phases = current_green_phases
            return
        for ts_id, curr_green in current_green_phases.items():
            if curr_green != self.last_green_phases.get(ts_id, curr_green):
                self.cumulative_switches += 1
        self.last_green_phases = current_green_phases

    @staticmethod
    def _coerce_action(action, action_space):
        """
        Convert SB3 action outputs (often numpy arrays) to a valid action
        object for the target Gymnasium action space.
        """
        if isinstance(action_space, spaces.Discrete):
            if isinstance(action, np.ndarray):
                if action.size == 0:
                    return action_space.sample()
                action = action.reshape(-1)[0]
            try:
                value = int(action)
            except Exception:
                return action_space.sample()
            if action_space.contains(value):
                return value
            return action_space.sample()

        if isinstance(action_space, spaces.MultiDiscrete):
            try:
                arr = np.asarray(action, dtype=np.int64).reshape(action_space.nvec.shape)
            except Exception:
                return action_space.sample()
            if action_space.contains(arr):
                return arr
            return action_space.sample()

        if isinstance(action_space, spaces.MultiBinary):
            try:
                arr = np.asarray(action, dtype=np.int8).reshape(action_space.shape)
            except Exception:
                return action_space.sample()
            arr = np.clip(arr, 0, 1)
            if action_space.contains(arr):
                return arr
            return action_space.sample()

        if isinstance(action_space, spaces.Box):
            try:
                arr = np.asarray(action, dtype=action_space.dtype).reshape(action_space.shape)
            except Exception:
                return action_space.sample()
            arr = np.clip(arr, action_space.low, action_space.high)
            if action_space.contains(arr):
                return arr
            return action_space.sample()

        return action if action_space.contains(action) else action_space.sample()
    
    def render(self):
        """Render the environment."""
        return self.pz_env.render()
    
    def close(self):
        """Close the environment."""
        self.pz_env.close()


# Algorithm configurations - base parameters (episode-dependent params set in main())
# Tuned for fair comparison across algorithms
ALGORITHM_CONFIGS = {
    "dqn": {
        "class": DQN,
        "base_hyperparams": {
            "learning_rate": 0.0005,        # Standard for DQN
            "batch_size": 64,                # Standard batch size
            "train_freq": 4,                 # Update every 4 steps
            "gradient_steps": 1,             # 1 gradient step per update
            "exploration_fraction": 0.3,     # Explore for 30% of training
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "gamma": 0.99,                   # Discount factor
        },
        # Functions to calculate episode-dependent hyperparameters
        "adaptive_hyperparams": lambda steps_per_ep: {
            "buffer_size": max(50000, steps_per_ep * 10),  # At least 10 episodes
            "learning_starts": steps_per_ep * 2,            # 2 full episodes before learning
            "target_update_interval": steps_per_ep // 2,    # Update target every 0.5 episodes
        }
    },
    "ppo": {
        "class": PPO,
        "base_hyperparams": {
            "learning_rate": 0.0003,         # Standard for PPO
            "batch_size": 128,               # Minibatch size for updates
            "n_epochs": 10,                  # 10 passes through collected data
            "gamma": 0.99,                   # Discount factor
            "gae_lambda": 0.95,              # GAE parameter for advantage estimation
            "clip_range": 0.2,               # PPO clipping parameter
            "ent_coef": 0.01,                # Entropy bonus for exploration
            "vf_coef": 0.5,                  # Value function coefficient
            "max_grad_norm": 0.5,            # Gradient clipping
        },
        "adaptive_hyperparams": lambda steps_per_ep: {
            "n_steps": steps_per_ep,         # 1 full episode of varied traffic before update
        }
    },
    "a2c": {
        "class": A2C,
        "base_hyperparams": {
            "learning_rate": 0.0005,         # Matched to DQN for comparison
            "gamma": 0.99,                   # Discount factor
            "gae_lambda": 0.95,              # GAE parameter (match PPO)
            "ent_coef": 0.01,                # Entropy bonus (match PPO)
            "vf_coef": 0.5,                  # Value function coefficient
            "max_grad_norm": 0.5,            # Gradient clipping
            "normalize_advantage": True,     # Stabilizes training
        },
        "adaptive_hyperparams": lambda steps_per_ep: {
            "n_steps": steps_per_ep // 2,    # 0.5 episodes between updates
        }
    }
}


def main(args):
    # Setup file paths
    if args.scenario_dir:
        # Custom scenario directory provided
        BASE_DIR = args.scenario_dir
        NET_FILE = f"{BASE_DIR}/{args.net_file}" if args.net_file else f"{BASE_DIR}/net.xml"
        ROUTE_FILE = f"{BASE_DIR}/{args.train_route_file}" if args.train_route_file else f"{BASE_DIR}/train.rou.xml"
        EVAL_ROUTE_FILE = f"{BASE_DIR}/{args.eval_route_file}" if args.eval_route_file else f"{BASE_DIR}/eval.rou.xml"
    else:
        # Default to cross_dynamic scenario
        NET_FILE = "scenarios/cross_dynamic/cross.net.xml"
        ROUTE_FILE = "scenarios/cross_dynamic/train.rou.xml"
        EVAL_ROUTE_FILE = "scenarios/cross_dynamic/eval.rou.xml"
    
    # Validate algorithm choice
    if args.algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(f"Algorithm must be one of {list(ALGORITHM_CONFIGS.keys())}")
    
    algo_config = ALGORITHM_CONFIGS[args.algorithm]
    
    # Auto-detect episode duration from route files if not specified
    episode_seconds = args.episode_seconds
    eval_episode_seconds = args.eval_episode_seconds
    
    if args.auto_duration:
        print("Auto-detecting episode duration from route files...")
        try:
            train_duration, train_source = detect_route_duration_seconds(ROUTE_FILE)
            if train_duration is not None:
                episode_seconds = train_duration
                print(
                    f"  Detected training duration: {episode_seconds}s "
                    f"({episode_seconds/3600:.2f}h) via {train_source}"
                )
            else:
                print("  Warning: Could not detect training duration, keeping provided value")

            eval_duration, eval_source = detect_route_duration_seconds(EVAL_ROUTE_FILE)
            if eval_duration is not None:
                eval_episode_seconds = eval_duration
                print(
                    f"  Detected eval duration: {eval_episode_seconds}s "
                    f"({eval_episode_seconds/3600:.2f}h) via {eval_source}"
                )
            else:
                print("  Warning: Could not detect eval duration, keeping provided value")
        except Exception as e:
            print(f"  Warning: Could not auto-detect duration ({e}), using provided values")
    
    # 0. Compute Baseline First (uses single-agent env for both single and multi-agent training)
    print("Computing baseline metrics...")
    baseline_metrics = run_baseline(NET_FILE, EVAL_ROUTE_FILE, eval_episode_seconds)
    
    print(f"Training episode length: {episode_seconds}s ({episode_seconds/3600:.2f}h)")
    print(f"Evaluation episode length: {eval_episode_seconds}s ({eval_episode_seconds/3600:.2f}h)")
    
    # Calculate steps per episode (assuming 5s delta_time)
    STEPS_PER_EPISODE = episode_seconds // 5
    print(f"Steps per episode: {STEPS_PER_EPISODE}")
    
    # Warn if not using auto-duration (hyperparams may be mismatched)
    if not args.auto_duration:
        print(f"\n⚠️  WARNING: Using default episode duration ({episode_seconds}s).")
        print(f"   If your route files have different durations, use --auto-duration flag!")
        print(f"   Adaptive hyperparameters are based on this duration.\n")
    
    # Build adaptive hyperparameters based on episode length
    hyperparams = algo_config["base_hyperparams"].copy()
    adaptive_params = algo_config["adaptive_hyperparams"](STEPS_PER_EPISODE)
    hyperparams.update(adaptive_params)

    # Make adaptive params vectorization-aware so update cadence stays reasonable
    # when num_timesteps is shared across parallel environments.
    if args.n_envs > 1:
        if args.algorithm == "dqn":
            # Keep replay warmup and buffer sized to the total transitions generated
            # across all environments.
            hyperparams["buffer_size"] = max(hyperparams["buffer_size"], STEPS_PER_EPISODE * 10 * args.n_envs)
            # SB3 tracks learning_starts in global timesteps; VecEnv already advances
            # num_timesteps by n_envs each collector step.
            hyperparams["learning_starts"] = STEPS_PER_EPISODE * 2

            # With VecEnv, each collector step yields n_envs transitions.
            # Increase gradient_steps to keep a similar update/data ratio.
            hyperparams["train_freq"] = 1
            hyperparams["gradient_steps"] = max(1, args.n_envs // 4)
            # SB3 applies vector-env compensation when checking target updates, so
            # keep this in global timestep units (do not multiply by n_envs here).
            hyperparams["target_update_interval"] = max(500, STEPS_PER_EPISODE // 2)

        elif args.algorithm in {"ppo", "a2c"}:
            # Keep rollout size in a practical range:
            # total_rollout_steps ~= n_steps * n_envs.
            target_rollout_steps = min(max(2048, args.n_envs * 64), max(2048, STEPS_PER_EPISODE * 2))
            n_steps_per_env = max(64, target_rollout_steps // args.n_envs)
            # Do not exceed one episode worth of steps in a single env rollout.
            n_steps_per_env = min(n_steps_per_env, STEPS_PER_EPISODE)
            hyperparams["n_steps"] = int(n_steps_per_env)

    # Keep adaptive choices compatible with the actual training budget for both
    # single-agent and multi-agent runs (global timesteps in SB3 semantics).
    if args.algorithm == "dqn":
        # Ensure replay warmup does not consume nearly the whole run.
        # Keep at least a minimal warmup, but leave room for meaningful updates.
        budget_capped_learning_starts = max(1000, args.total_timesteps // 5)
        hyperparams["learning_starts"] = min(hyperparams["learning_starts"], budget_capped_learning_starts)

        # Keep target updates frequent enough within short runs.
        hyperparams["target_update_interval"] = min(
            hyperparams["target_update_interval"],
            max(250, args.total_timesteps // 4)
        )
    
    print(f"\nAdaptive hyperparameters for {args.algorithm.upper()}:")
    for key, value in adaptive_params.items():
        print(f"  {key}: {value}")
    if args.n_envs > 1:
        print("Vectorized adjustments:")
        for key in sorted(hyperparams.keys()):
            if key in adaptive_params and hyperparams[key] != adaptive_params[key]:
                print(f"  {key}: {adaptive_params[key]} -> {hyperparams[key]}")

        if args.algorithm in {"ppo", "a2c"}:
            total_rollout = hyperparams["n_steps"] * args.n_envs
            est_updates = max(1, args.total_timesteps // total_rollout)
            print(f"  rollout per update: {hyperparams['n_steps']} x {args.n_envs} = {total_rollout} timesteps")
            print(f"  expected policy updates: ~{est_updates}")
        elif args.algorithm == "dqn":
            print(f"  effective transitions per update trigger: {hyperparams['train_freq'] * args.n_envs}")
            print(f"  gradient steps per trigger: {hyperparams['gradient_steps']}")
    
    # 1. Setup Training Environment (Multi-agent or Single-agent)
    # Assumes SUMO-RL default delta_time=5s (used elsewhere in this script).
    warmup_steps = max(0, args.warmup_seconds // 5)
    if args.warmup_seconds > 0:
        print(f"Applying warmup: {args.warmup_seconds}s ({warmup_steps} steps)")

    def make_env():
        if args.multiagent:
            # Use PettingZoo parallel environment for multi-agent
            pz_env = sumo_rl.parallel_env(
                net_file=NET_FILE,
                route_file=ROUTE_FILE,
                out_csv_name=f"outputs/{args.output_prefix}_{args.algorithm}_queue_run" if args.output_prefix else None,
                use_gui=args.gui,
                num_seconds=episode_seconds,
                add_system_info=True,
                reward_fn=reward_minimize_max_queue,
                observation_class=GridObservationFunction
            )
            # Wrap PettingZoo env to make it compatible with SB3
            return PettingZooToGymWrapper(pz_env, warmup_steps=warmup_steps)
        else:
            # Use standard Gym environment for single-agent
            return gym.make('sumo-rl-v0',
                           net_file=NET_FILE,
                           route_file=ROUTE_FILE,
                           out_csv_name=f"outputs/{args.output_prefix}_{args.algorithm}_queue_run" if args.output_prefix else None,
                           use_gui=args.gui,
                           num_seconds=episode_seconds,
                           add_system_info=True,
                           reward_fn=reward_minimize_max_queue,
                           observation_class=GridObservationFunction)
    
    # Create vectorized environment with support for parallel environments
    if args.n_envs > 1:
        print(f"Creating {args.n_envs} parallel training environments with SubprocVecEnv...")
        env = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    else:
        print("Creating single training environment with DummyVecEnv...")
        env = DummyVecEnv([make_env])
    
    # Setup Evaluation Environment (Single separate instance, no GUI)
    print("Creating evaluation environment...")
    if args.multiagent:
        pz_eval_env = sumo_rl.parallel_env(
            net_file=NET_FILE,
            route_file=EVAL_ROUTE_FILE,
            use_gui=False,
            num_seconds=eval_episode_seconds,
            add_system_info=True,
            reward_fn=reward_minimize_max_queue,
            observation_class=GridObservationFunction,
            sumo_seed='42'
        )
        eval_env = PettingZooToGymWrapper(pz_eval_env, warmup_steps=warmup_steps)
    else:
        eval_env = gym.make('sumo-rl-v0',
                            net_file=NET_FILE,
                            route_file=EVAL_ROUTE_FILE,
                            use_gui=False,
                            num_seconds=eval_episode_seconds,
                            add_system_info=True,
                            reward_fn=reward_minimize_max_queue,
                            observation_class=GridObservationFunction,
                            sumo_seed='42')
    
    # Optional: Apply normalization
    if args.normalize:
        print("Applying observation normalization...")
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    # 2. Create the agent
    AlgorithmClass = algo_config["class"]
    
    # Allow command-line override of learning rate
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    
    model = AlgorithmClass(
        "MlpPolicy",
        env,
        verbose=1,
        **hyperparams
    )
    
    mode_str = "MULTI-AGENT" if args.multiagent else "SINGLE-AGENT"
    print(f"\nStarting {args.algorithm.upper()} training in {mode_str} mode...")
    print(f"  Algorithm: {args.algorithm.upper()}")
    print(f"  Number of parallel environments: {args.n_envs}")
    print(f"  Total timesteps: {args.total_timesteps}")
    
    # 3. Initialize Wandb
    run_name = f"{args.algorithm}-{args.run_name}" if args.run_name else f"{args.algorithm}-queue"
    if args.multiagent:
        run_name = f"ma-{run_name}"  # Prefix with 'ma' for multi-agent
    
    wandb.init(
        entity="fds-final-project",
        project="rl-traffic-light",
        name=run_name,
        config={
            "algorithm": args.algorithm,
            "multiagent": args.multiagent,
            "parameter_sharing": args.multiagent,  # Currently always True for multi-agent
            "n_envs": args.n_envs,
            "scenario_dir": args.scenario_dir if args.scenario_dir else "scenarios/cross_dynamic",
            "net_file": NET_FILE,
            "route_file": ROUTE_FILE,
            "eval_route_file": EVAL_ROUTE_FILE,
            "episode_seconds": episode_seconds,
            "eval_episode_seconds": eval_episode_seconds,
            "total_timesteps": args.total_timesteps,
            "normalize": args.normalize,
            "warmup_seconds": args.warmup_seconds,
            "baseline_metrics": baseline_metrics,
            **hyperparams
        }
    )
    
    # Log baseline once at run start so it is always visible in charts,
    # even before the first periodic validation callback fires.
    if baseline_metrics:
        baseline_log = {
            "validation/step": 0,
            "baseline/mean_waiting_time": baseline_metrics["mean_waiting_time"],
            "baseline/mean_queue_length": baseline_metrics["mean_queue_length"],
            "baseline/mean_speed": baseline_metrics["mean_speed"],
            "baseline/total_arrived": baseline_metrics["total_arrived"],
            "baseline/total_switches": baseline_metrics.get("total_switches", 0),
        }
        wandb.log(baseline_log, step=0)
        wandb.summary.update({
            "baseline/mean_waiting_time": baseline_metrics["mean_waiting_time"],
            "baseline/mean_queue_length": baseline_metrics["mean_queue_length"],
            "baseline/mean_speed": baseline_metrics["mean_speed"],
            "baseline/total_arrived": baseline_metrics["total_arrived"],
            "baseline/total_switches": baseline_metrics.get("total_switches", 0),
        })
    
    # 4. Setup callbacks
    # Setup path for best model
    model_name = f"{args.output_prefix}_{args.algorithm}_model" if args.output_prefix else f"{args.algorithm}_model"
    if args.multiagent:
        model_name = f"{model_name}_multiagent"
    best_model_path = f"weights/{model_name}_best"
    os.makedirs("weights", exist_ok=True)
    
    # Scale eval frequency by number of parallel environments
    # (num_timesteps increments by n_envs each step, so we need to scale accordingly)
    eval_freq_timesteps = STEPS_PER_EPISODE * args.n_envs
    print(f"\nEvaluation frequency: {eval_freq_timesteps} timesteps (1 episode × {args.n_envs} envs)")
    print(f"Expected number of evaluations: {args.total_timesteps // eval_freq_timesteps}")
    
    callbacks = [
        TrafficWandbCallback(),
        ValidationCallback(
            eval_env, 
            baseline_metrics, 
            eval_freq=eval_freq_timesteps,
            best_model_save_path=best_model_path  # Save best based on throughput
        )
    ]
    
    # 5. Train
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True)
    finally:
        # Save final model (for comparison with best)
        final_model_path = f"weights/{model_name}_final"
        model.save(final_model_path)
        print(f"\nTraining complete!")
        print(f"  Final model saved to {final_model_path}.zip")
        print(f"  Best model saved to {best_model_path}/best_model.zip")
        
        # Save normalization stats if used
        if args.normalize:
            norm_name = f"{args.output_prefix}_{args.algorithm}_vec_normalize.pkl" if args.output_prefix else f"{args.algorithm}_vec_normalize.pkl"
            if args.multiagent:
                norm_name = f"multiagent_{norm_name}"
            env.save(f"weights/{norm_name}")
            print(f"  Normalization stats saved to weights/{norm_name}")
        
        # Cleanup
        eval_env.close()
        env.close()
        wandb.finish()
    
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train traffic light controller with various RL algorithms (single or multi-agent)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Multi-agent configuration
    parser.add_argument(
        "--multiagent",
        action="store_true",
        help="Enable multi-agent training using PettingZoo parallel environment with parameter sharing"
    )
    
    # Parallel training configuration
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel training environments (use SubprocVecEnv if > 1)"
    )
    
    # Scenario configuration
    parser.add_argument(
        "--scenario-dir",
        type=str,
        default=None,
        help="Path to scenario directory"
    )
    
    parser.add_argument(
        "--net-file",
        type=str,
        default=None,
        help="Network file name relative to scenario-dir"
    )
    
    parser.add_argument(
        "--train-route-file",
        type=str,
        default=None,
        help="Training route file name relative to scenario-dir"
    )
    
    parser.add_argument(
        "--eval-route-file",
        type=str,
        default=None,
        help="Evaluation route file name relative to scenario-dir"
    )
    
    parser.add_argument(
        "--auto-duration",
        action="store_true",
        help="Auto-detect episode duration from route file comments"
    )
    
    # Algorithm configuration
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        required=True,
        choices=["dqn", "ppo", "a2c"],
        help="RL algorithm to use"
    )
    
    # Episode configuration
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=16200,
        help="Duration of each training episode in seconds (ignored if --auto-duration)"
    )
    
    parser.add_argument(
        "--eval-episode-seconds",
        type=int,
        default=5400,
        help="Duration of each evaluation episode in seconds (ignored if --auto-duration)"
    )

    parser.add_argument(
        "--warmup-seconds",
        type=int,
        default=0,
        help="Advance simulation this many seconds after each reset before learning/evaluation starts"
    )
    
    # Training configuration
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=50000,
        help="Total training timesteps"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides default for selected algorithm)"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply observation normalization (VecNormalize)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output CSV files"
    )
    
    # Visualization
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with SUMO GUI visualization"
    )
    
    # Wandb configuration
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for this run (for Wandb)"
    )
    
    args = parser.parse_args()
    main(args)
