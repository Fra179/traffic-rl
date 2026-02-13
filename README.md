# Traffic Light Control with Reinforcement Learning

A reinforcement learning framework for optimizing traffic light control using SUMO (Simulation of Urban MObility) and Stable Baselines3.

## Features

- 🚦 **Single-agent & multi-agent** traffic light control
- 🤖 Multiple RL algorithms: **DQN, PPO, A2C**
- 🚀 **Parallel training** with multiple environments
- 📊 **Experiment tracking** with Weights & Biases
- 🗺️ **Multiple scenarios**: Berlin intersections, grid networks, custom scenarios
- 🔧 **Flexible configuration** via command-line arguments

## Requirements

### SUMO 1.26.0

This project **requires SUMO version 1.26.0** exactly. Other versions may not work correctly.

#### Check if SUMO 1.26.0 is installed

```bash
sumo --version
```

If you see `Eclipse SUMO sumo Version v1_26_0`, you're good to go! Otherwise, follow the installation instructions below.

#### Installing SUMO 1.26.0

<details>
<summary><b>Option 1: Pre-built Packages (if available)</b></summary>

Check the [SUMO downloads page](https://sumo.dlr.de/docs/Downloads.php) for pre-built packages for your system.

</details>

<details>
<summary><b>Option 2: Build from Source (Recommended)</b></summary>

**Prerequisites**: cmake, ninja-build, g++, and SUMO dependencies

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y \
    cmake ninja-build g++ \
    libxerces-c-dev libfox-1.6-dev \
    libgdal-dev libproj-dev libgl2ps-dev \
    python3-dev swig
```

**Build SUMO 1.26.0**:

```bash
# 1. Download SUMO 1.26.0
wget https://github.com/eclipse-sumo/sumo/archive/refs/tags/v1_26_0.zip
unzip v1_26_0.zip
cd sumo-1_26_0

# 2. Configure build
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/sumo-1.26.0" \
  -DCMAKE_INSTALL_LIBDIR=lib

# 3. Build (using all CPU cores)
cmake --build build -j"$(nproc)"

# 4. Install to $HOME/opt/sumo-1.26.0
cmake --install build

# 5. Verify installation
"$HOME/opt/sumo-1.26.0/bin/sumo" --version
```

**Set environment variables**:

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
export SUMO_PREFIX="$HOME/opt/sumo-1.26.0"
export SUMO_HOME="$SUMO_PREFIX/share/sumo"
export PATH="$SUMO_PREFIX/bin:$PATH"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

</details>

### Python Dependencies

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

## Quick Start

### Single-Agent Training

Train a policy to control **one** traffic light:

```bash
python experiments/train.py \
    --algorithm ppo \
    --scenario-dir scenarios/cross_dynamic \
    --auto-duration \
    --total-timesteps 50000
```

### Multi-Agent Training

Train a shared policy to control **all** traffic lights in a scenario:

```bash
python experiments/train.py \
    --algorithm ppo \
    --multiagent \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 100000
```

### Parallel Training (Faster)

Use multiple environments for faster training:

```bash
python experiments/train.py \
    --algorithm ppo \
    --multiagent \
    --n-envs 8 \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 200000
```

### Batch Training

Train all Berlin scenarios with all algorithms:

```bash
./train_all_berlin.sh 10 8  # 10 episodes, 8 parallel jobs
```

## Project Structure

```
traffic-rl/
├── experiments/
│   ├── train.py              # Main unified training script
│   ├── evaluate.py           # Model evaluation
│   ├── cross_sarsa.py        # SARSA experiments
│   └── TRAINING_README.md    # Detailed training guide ←
├── scenarios/
│   ├── berlin-small/         # Berlin intersection scenarios (A-J)
│   ├── cross/                # Simple cross intersection
│   ├── cross_dynamic/        # Cross with dynamic traffic
│   └── grid4x4/              # 4x4 grid network
├── traffic_rl/
│   ├── callbacks/            # Training callbacks
│   ├── observations/         # Custom observation functions
│   └── rewards/              # Custom reward functions
├── train_all_berlin.sh       # Batch training script
└── README.md                 # This file
```

## Documentation

📖 **[Complete Training Guide](experiments/TRAINING_README.md)** - Detailed documentation on:
- Single-agent vs multi-agent modes
- All command-line arguments
- Algorithm configurations
- Creating custom scenarios
- Troubleshooting

## Algorithms

All algorithms are from [Stable Baselines3](https://stable-baselines3.readthedocs.io/):

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **DQN** | Deep Q-Network | Discrete action spaces, off-policy |
| **PPO** | Proximal Policy Optimization | Most stable, good default choice |
| **A2C** | Advantage Actor-Critic | Fast training, on-policy |

## Scenarios

### Berlin Intersections (A-J)

Real-world intersection networks from Berlin with multiple traffic signals.

```bash
python experiments/train.py -a ppo --multiagent \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration
```

### Cross Intersection

Simple single intersection for testing.

```bash
python experiments/train.py -a dqn \
    --scenario-dir scenarios/cross_dynamic \
    --auto-duration
```

### Grid Networks

4x4 grid network with coordinated traffic signals.

```bash
python experiments/train.py -a ppo --multiagent \
    --scenario-dir scenarios/grid4x4 \
    --auto-duration
```

## Custom Scenarios

Place your SUMO scenario files in a directory:

```
scenarios/my_scenario/
├── net.xml              # Network definition
├── train.rou.xml        # Training routes
└── eval.rou.xml         # Evaluation routes
```

Then train:

```bash
python experiments/train.py -a ppo \
    --scenario-dir scenarios/my_scenario \
    --auto-duration
```

## Experiment Tracking

Training metrics are automatically logged to [Weights & Biases](https://wandb.ai/).

Configure your W&B entity in the training scripts (default: `fds-final-project`).

## Key Features

### Automatic Duration Detection

Use `--auto-duration` to automatically detect episode duration from route files:

```xml
<!-- In your .rou.xml file -->
<!-- Total Duration: 16200s -->
```

### Parameter Sharing

Multi-agent mode uses **parameter sharing**: one policy controls all traffic signals, learning from all simultaneously.

### Observation Functions

- **Grid Observation** (default): Structured grid-based state representation
- Custom observations can be added in `traffic_rl/observations/`

### Reward Functions

- **Minimize Max Queue** (default): Reduce maximum queue length
- **Vidali Waiting Time**: Minimize total waiting time
- Custom rewards can be added in `traffic_rl/rewards/`

## Output

### Models
Trained models are saved in `weights/`:
- `{prefix}_{algorithm}_model_best/` - Best model during training
- `{prefix}_{algorithm}_model_final.zip` - Final model after training

### Logs
Training logs and metrics are saved in `logs/`

### CSV Output
Episode-level metrics saved in `outputs/` (optional, via `--output-prefix`)

## Environment Variables

The training scripts automatically set:
```bash
export LIBSUMO_AS_TRACI="1"  # Use libsumo for better performance
```

## Troubleshooting

### SUMO not found

```bash
# Make sure SUMO_HOME is set
echo $SUMO_HOME

# Should output: /home/user/opt/sumo-1.26.0/share/sumo
```

### Version mismatch

```bash
# Check SUMO version
sumo --version

# Must be: Eclipse SUMO sumo Version v1_26_0
```

### libsumo errors

Make sure you built SUMO with Python bindings enabled (which is default with the build instructions above).

### Out of memory with parallel training

Reduce `--n-envs`:
```bash
python experiments/train.py --n-envs 4  # Instead of 8
```

## Citation

This project uses:
- [SUMO](https://www.eclipse.org/sumo/) - Traffic simulation
- [SUMO-RL](https://github.com/LucasAlegre/sumo-rl) - RL environment wrapper
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms

## License

See [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For detailed training instructions, see [experiments/TRAINING_README.md](experiments/TRAINING_README.md).

For issues and questions, please open a GitHub issue.
