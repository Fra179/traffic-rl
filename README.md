# Traffic-RL

Reinforcement learning for adaptive traffic-light control with [SUMO](https://www.eclipse.org/sumo/), [SUMO-RL](https://github.com/LucasAlegre/sumo-rl), and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

This project was developed for the **Reinforcement Learning course at Sapienza University of Rome**.

## What is in this repository

- Single-agent and multi-agent traffic signal control
- Parameter sharing for multi-agent training (one shared policy)
- Algorithms: `DQN`, `PPO`, `A2C`
- Prebuilt scenarios: `cross`, `grid2x2`, `grid3x3`, `grid4x4`, `berlin-small` (A-J + static)
- Batch scripts for large experiment runs and evaluations

## Repository structure

```text
traffic-rl/
├── experiments/                  # Training and evaluation entrypoints
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_multi_agent.py
│   └── cross_sarsa.py
├── traffic_rl/                   # Rewards, observations, callbacks, SUMO XML utils
├── scenarios/                    # SUMO networks and routes
├── weights/                      # Saved models
├── outputs/                      # CSV outputs from SUMO/SB3 runs
├── logs/                         # Batch script logs
├── train_all_cross.sh
├── train_all_berlin.sh
├── evaluate_all_cross_2x2.sh
└── evaluate_all_berlin.sh
```

## Requirements

- Python `>= 3.10`
- SUMO `1.26.0`
- `uv` package manager

Check SUMO version:

```bash
sumo --version
```

### Compile SUMO 1.26.0 from source

If your installed SUMO version is different, build `v1_26_0` explicitly:

1. Install build dependencies (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake ninja-build g++ \
  libxerces-c-dev libfox-1.6-dev \
  libgdal-dev libproj-dev libgl2ps-dev \
  python3-dev swig
```

2. Download and build:

```bash
wget https://github.com/eclipse-sumo/sumo/archive/refs/tags/v1_26_0.zip
unzip v1_26_0.zip
cd sumo-1_26_0

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/sumo-1.26.0" \
  -DCMAKE_INSTALL_LIBDIR=lib

cmake --build build -j"$(nproc)"
cmake --install build
```

3. Verify:

```bash
"$HOME/opt/sumo-1.26.0/bin/sumo" --version
```

## Setup

1. Install Python dependencies:

```bash
uv sync
```

2. Configure SUMO (example if installed in `$HOME/opt/sumo-1.26.0`):

```bash
export SUMO_PREFIX="$HOME/opt/sumo-1.26.0"
export SUMO_HOME="$SUMO_PREFIX/share/sumo"
export PATH="$SUMO_PREFIX/bin:$PATH"
```

3. Configure Weights & Biases (training logs there):

```bash
wandb login
```

If you do not want online logging, run with offline mode:

```bash
export WANDB_MODE=offline
```

## Quick start

### 1) Train on the `cross` scenario (single-agent)

```bash
uv run experiments/train.py \
  --algorithm ppo \
  --scenario-dir scenarios/cross \
  --net-file cross.net.xml \
  --train-route-file train_generated.rou.xml \
  --eval-route-file eval_generated.rou.xml \
  --auto-duration \
  --total-timesteps 50000 \
  --output-prefix cross_ppo
```

### 2) Train on one Berlin intersection (multi-agent)

```bash
uv run experiments/train.py \
  --algorithm ppo \
  --multiagent \
  --scenario-dir scenarios/berlin-small/\
  --net-file a.net.xml \
  --train-route-file train_generated.rou.xml \
  --eval-route-file eval_generated.rou.xml \
  --auto-duration \
  --total-timesteps 100000 \
  --output-prefix berlin_ppo
```

### 3) Evaluate a trained model

```bash
uv run experiments/evaluate.py \
  weights/cross_ppo_ppo_model_best/best_model.zip \
  --algorithm ppo \
  --scenario-dir scenarios/cross \
  --net-file cross.net.xml \
  --route-file eval_generated.rou.xml \
  --episode-seconds 3600 \
  --compare-baseline
```

## Batch scripts

Train all algorithms on `cross`:

```bash
./train_all_cross.sh 50
```

Train all algorithms on Berlin intersections `A-J`:

```bash
./train_all_berlin.sh 50
```

Evaluate heterogeneous Berlin system:

```bash
./evaluate_all_berlin.sh ppo
```

Evaluate cross-trained models on `grid2x2` transfer setting:

```bash
./evaluate_all_cross_2x2.sh
```

## Scenario generation

Generate traffic patterns for `cross`:

```bash
uv run scenarios/generate_intersection_patterns.py
```

Generate traffic patterns for Berlin static network:

```bash
uv run scenarios/berlin-small/generate_intersection_patterns.py berlin
```

Generate traffic patterns for Berlin intersections `A-J`:

```bash
uv run scenarios/berlin-small/generate_intersection_patterns.py intersections
```

## Notes

- Prefer `--auto-duration` so training/evaluation episode length is inferred from route files.
- `experiments/train.py` and `experiments/evaluate.py` have defaults pointing to `scenarios/cross_dynamic`, which is not included here; pass `--scenario-dir` and file names explicitly as in the examples above.
- Models are saved in `weights/` (best and final checkpoints).

## License

This repository is released under the terms of the `LICENSE` file.
