#!/usr/bin/env python3
"""Train a single PPO agent to control all traffic lights on the 3x3 grid.

This script wraps SUMO-RL's multi-agent env into a *single-agent* Gymnasium env:
one PPO policy outputs one action per traffic light each decision step.

Reward goal:
  - Minimize global waiting time
  - Keep vehicles moving (penalize halts)
  - Waiting / stopping emergency vehicles is penalized *very severely*

Examples:
  uv run experiments/3x3_training.py --timesteps 200_000
  uv run experiments/3x3_training.py --gui --timesteps 50_000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from sumo_rl import SumoEnvironment


DEFAULT_SCENARIO_DIR = Path("/home/andrea/traffic-rl/scenarios/grid3x3")
os.env["LIBSUMO_AS_TRACI"] = 1

def _resolve_scenario_dir(user_value: str | None) -> Path:
	if user_value is None:
		if DEFAULT_SCENARIO_DIR.exists():
			return DEFAULT_SCENARIO_DIR
		return (Path(__file__).resolve().parents[1] / "scenarios" / "grid3x3").resolve()
	return Path(user_value).expanduser().resolve()


def _require_sumo_in_path(use_gui: bool) -> None:
	binary = "sumo-gui" if use_gui else "sumo"
	if shutil.which(binary) is None:
		raise RuntimeError(
			f"Could not find '{binary}' in PATH. Install SUMO and ensure 'sumo'/'sumo-gui' is available."
		)


@dataclass(frozen=True)
class RewardConfig:
	waiting_penalty: float = 1.0
	halting_penalty: float = 0.2
	speed_reward: float = 0.05
	emergency_waiting_penalty: float = 50.0
	emergency_stopped_penalty: float = 200.0
	stopped_speed_threshold: float = 0.1


class SingleAgentAllTrafficLightsEnv(gym.Env):
	"""Single-agent wrapper around SUMO-RL multi-agent env.

	Action: MultiDiscrete, one discrete phase-change action per traffic light.
	Observation: Dict[str, Box], one observation vector per traffic light.
	Reward: global scalar computed from vehicle-level TraCI metrics.
	"""

	metadata = {"render_modes": []}

	def __init__(
		self,
		net_file: str,
		route_file: str,
		use_gui: bool,
		num_seconds: int,
		delta_time: int,
		yellow_time: int,
		min_green: int,
		max_green: int,
		sumo_seed: int,
		reward_cfg: RewardConfig,
		additional_sumo_cmd: str | None = None,
	) -> None:
		super().__init__()

		self.reward_cfg = reward_cfg
		self._prev_total_waiting: float | None = None

		self.env = SumoEnvironment(
			net_file=net_file,
			route_file=route_file,
			use_gui=use_gui,
			num_seconds=num_seconds,
			delta_time=delta_time,
			yellow_time=yellow_time,
			min_green=min_green,
			max_green=max_green,
			single_agent=False,
			reward_fn="diff-waiting-time",  # ignored by wrapper reward, but keeps env consistent
			sumo_seed=sumo_seed,
			out_csv_name=None,
			sumo_warnings=True,
			additional_sumo_cmd=additional_sumo_cmd,
		)

		# Stable ordering for vector actions
		self.ts_ids = list(self.env.ts_ids)

		self.action_space = spaces.MultiDiscrete(
			np.array([int(self.env.action_spaces(ts_id).n) for ts_id in self.ts_ids], dtype=np.int64)
		)

		self.observation_space = spaces.Dict({ts_id: self.env.observation_spaces(ts_id) for ts_id in self.ts_ids})

	def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
		obs = self.env.reset(seed=seed)
		self._prev_total_waiting = None
		info = {}
		return obs, info

	def close(self) -> None:
		self.env.close()

	def _is_emergency_vehicle(self, veh_id: str) -> bool:
		# Heuristic: match by vehicle class or type id.
		try:
			vclass = self.env.sumo.vehicle.getVehicleClass(veh_id)
			if vclass == "emergency":
				return True
		except Exception:
			pass
		try:
			type_id = self.env.sumo.vehicle.getTypeID(veh_id)
			return "emerg" in type_id.lower()
		except Exception:
			return False

	def _global_reward(self) -> tuple[float, dict[str, float]]:
		veh_ids = list(self.env.sumo.vehicle.getIDList())
		if not veh_ids:
			return 0.0, {
				"total_waiting": 0.0,
				"halted": 0.0,
				"mean_speed": 0.0,
				"emergency_waiting": 0.0,
				"emergency_stopped": 0.0,
			}

		total_waiting = 0.0
		halted = 0.0
		speed_sum = 0.0

		emergency_waiting = 0.0
		emergency_stopped = 0.0

		for vid in veh_ids:
			wt = float(self.env.sumo.vehicle.getWaitingTime(vid))
			spd = float(self.env.sumo.vehicle.getSpeed(vid))
			total_waiting += wt
			speed_sum += spd
			if spd <= self.reward_cfg.stopped_speed_threshold:
				halted += 1.0

			if self._is_emergency_vehicle(vid):
				emergency_waiting += wt
				if spd <= self.reward_cfg.stopped_speed_threshold:
					emergency_stopped += 1.0

		mean_speed = speed_sum / max(1.0, float(len(veh_ids)))

		# Prefer improvement in total waiting (diff) to reduce scale sensitivity.
		if self._prev_total_waiting is None:
			delta_waiting = 0.0
		else:
			delta_waiting = total_waiting - self._prev_total_waiting
		self._prev_total_waiting = total_waiting

		reward = 0.0
		reward -= self.reward_cfg.waiting_penalty * delta_waiting
		reward -= self.reward_cfg.halting_penalty * halted
		reward += self.reward_cfg.speed_reward * mean_speed

		# Emergency vehicles: punish waiting/stopping extremely strongly.
		reward -= self.reward_cfg.emergency_waiting_penalty * emergency_waiting
		reward -= self.reward_cfg.emergency_stopped_penalty * emergency_stopped

		metrics = {
			"total_waiting": total_waiting,
			"halted": halted,
			"mean_speed": mean_speed,
			"emergency_waiting": emergency_waiting,
			"emergency_stopped": emergency_stopped,
		}
		return float(reward), metrics

	def step(self, action: np.ndarray):
		# Map vector action -> per-TL dict expected by SUMO-RL multi-agent env.
		if isinstance(action, (list, tuple)):
			action = np.asarray(action)
		action = np.asarray(action, dtype=np.int64).reshape(-1)
		if action.shape[0] != len(self.ts_ids):
			raise ValueError(f"Expected {len(self.ts_ids)} actions, got shape={action.shape}")
		actions = {ts_id: int(action[i]) for i, ts_id in enumerate(self.ts_ids)}

		obs, _rewards, dones, info = self.env.step(actions)
		truncated = bool(dones.get("__all__", False))
		terminated = False

		reward, metrics = self._global_reward()
		info = dict(info) if isinstance(info, dict) else {}
		info.update(metrics)
		return obs, reward, terminated, truncated, info


def main() -> int:
	parser = argparse.ArgumentParser(description="Train PPO on grid3x3 using SUMO-RL (single agent controls all TLs)")
	parser.add_argument("--scenario-dir", default=None, help="Scenario folder (defaults to /home/andrea/traffic-rl/scenarios/grid3x3)")
	parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI")
	parser.add_argument("--seconds", type=int, default=3600, help="Simulation horizon (seconds)")
	parser.add_argument(
		"--traffic-scale",
		type=float,
		default=1.0,
		help="Scale traffic demand in SUMO via --scale (e.g. 0.5 fewer vehicles, 2.0 more)",
	)
	parser.add_argument("--delta-time", type=int, default=5, help="Seconds between decisions")
	parser.add_argument("--yellow-time", type=int, default=2)
	parser.add_argument("--min-green", type=int, default=5)
	parser.add_argument("--max-green", type=int, default=50)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--timesteps", type=int, default=200_000)
	parser.add_argument(
		"--n-steps",
		type=int,
		default=2048,
		help=(
			"PPO rollout length per update. Important for TensorBoard: if you stop training before the first update, "
			"the run folder may exist but contain no scalar data. Smaller values log/update more frequently."
		),
	)
	parser.add_argument(
		"--progress-bar",
		action="store_true",
		help="Show a training progress bar (requires tqdm + rich; auto-disables if missing)",
	)
	parser.add_argument("--tensorboard", default=str((Path(__file__).resolve().parents[1] / "runs").resolve()))
	parser.add_argument("--model-out", default=str((Path(__file__).resolve().parent / "ppo_grid3x3.zip").resolve()))
	parser.add_argument("--log-out", default=str((Path(__file__).resolve().parent / "ppo_grid3x3_monitor.csv").resolve()))

	# Reward weights
	parser.add_argument("--waiting-penalty", type=float, default=1.0)
	parser.add_argument("--halting-penalty", type=float, default=0.2)
	parser.add_argument("--speed-reward", type=float, default=0.05)
	parser.add_argument("--emergency-waiting-penalty", type=float, default=50.0)
	parser.add_argument("--emergency-stopped-penalty", type=float, default=200.0)
	args = parser.parse_args()

	if args.n_steps <= 0:
		raise SystemExit("--n-steps must be > 0")
	if args.timesteps < args.n_steps:
		print(
			f"Warning: --timesteps ({args.timesteps}) is < --n-steps ({args.n_steps}). PPO will not complete an update, "
			"so TensorBoard may show 'No scalar data was found'.",
		)

	scenario_dir = _resolve_scenario_dir(args.scenario_dir)
	net_file = scenario_dir / "grid3x3.net.xml"
	route_file = scenario_dir / "grid3x3.rou.xml"
	if not net_file.exists() or not route_file.exists():
		raise SystemExit(f"Missing scenario files in {scenario_dir} (expected grid3x3.net.xml and grid3x3.rou.xml)")

	_require_sumo_in_path(args.gui)
	if args.traffic_scale <= 0:
		raise SystemExit("--traffic-scale must be > 0")

	reward_cfg = RewardConfig(
		waiting_penalty=float(args.waiting_penalty),
		halting_penalty=float(args.halting_penalty),
		speed_reward=float(args.speed_reward),
		emergency_waiting_penalty=float(args.emergency_waiting_penalty),
		emergency_stopped_penalty=float(args.emergency_stopped_penalty),
	)

	def make_env():
		additional_sumo_cmd = f"--scale {float(args.traffic_scale)}"
		env = SingleAgentAllTrafficLightsEnv(
			net_file=str(net_file),
			route_file=str(route_file),
			use_gui=bool(args.gui),
			num_seconds=int(args.seconds),
			delta_time=int(args.delta_time),
			yellow_time=int(args.yellow_time),
			min_green=int(args.min_green),
			max_green=int(args.max_green),
			sumo_seed=int(args.seed),
			reward_cfg=reward_cfg,
			additional_sumo_cmd=additional_sumo_cmd,
		)
		return Monitor(env, filename=str(Path(args.log_out).expanduser().resolve()))

	vec_env = DummyVecEnv([make_env])

	model = PPO(
		policy="MultiInputPolicy",
		env=vec_env,
		verbose=1,
		tensorboard_log=str(Path(args.tensorboard).expanduser().resolve()),
		seed=int(args.seed),
		n_steps=int(args.n_steps),
	)

	try:
		try:
			model.learn(total_timesteps=int(args.timesteps), progress_bar=bool(args.progress_bar))
		except ImportError as e:
			if args.progress_bar:
				print(f"Progress bar disabled ({e}). Re-running without progress bar.")
				model.learn(total_timesteps=int(args.timesteps), progress_bar=False)
			else:
				raise

		model.save(str(Path(args.model_out).expanduser().resolve()))
		print(f"Saved model to: {Path(args.model_out).expanduser().resolve()}")
		return 0
	finally:
		vec_env.close()


if __name__ == "__main__":
	raise SystemExit(main())
