# Traffic Light Training Guide

This guide explains how to use the unified `train.py` script for both single-agent and multi-agent traffic light control.

## Overview

The `train.py` script is a unified training script that replaces both `cross_train.py` and `easy_ma.py`. It supports:

- **Single-agent mode** (default, same as `cross_train.py`): One policy controls **one traffic signal** in the scenario
- **Multi-agent mode** (`--multiagent`): One shared policy controls **all traffic signals** in the scenario simultaneously via parameter sharing
- **Parallel training** (`--n-envs N`): Run N copies of the environment in parallel for faster training (works with both modes)

### Quick Comparison

| Mode | Flag | Controls | Use Case |
|------|------|----------|----------|
| Single-agent | (none) | 1 traffic light | Simple intersections, baseline comparisons |
| Multi-agent | `--multiagent` | All traffic lights | Complex networks, coordinated control |
| Parallel | `--n-envs 8` | Same as chosen mode, but 8x faster | Speed up training |

## Key Features

### Multi-Agent Implementation
- Uses **PettingZoo parallel environment** API from SUMO-RL
- Implements **parameter sharing**: all traffic signals share the same policy
- Compatible with Stable Baselines3 (DQN, PPO, A2C) via custom wrapper
- Aggregates rewards across all agents for training signal

### PettingZooToGymWrapper
The wrapper converts PettingZoo's multi-agent environment to Gym's single-agent interface:
- Collects actions from all agents using the same policy
- Steps all agents simultaneously in the environment
- Returns average reward across all agents
- Tracks per-agent metrics and aggregates them

## Usage

### Single-Agent Training (Same as `cross_train.py`)
Train a policy to control **one traffic light** in the scenario:

```bash
python experiments/train.py \
    --algorithm dqn \
    --scenario-dir scenarios/cross_dynamic \
    --total-timesteps 50000 \
    --run-name "single-agent-test"
```

⚠️ **Note**: If your scenario has multiple traffic lights, only one will be controlled. Use `--multiagent` to control all of them.

### Multi-Agent Training (New!)
Train a shared policy to control **all traffic lights** in a scenario:

```bash
python experiments/train.py \
    --algorithm dqn \
    --multiagent \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 100000 \
    --run-name "berlin-A-multiagent"
```

✅ **Best for**: Scenarios with multiple intersections (e.g., Berlin scenarios, grid networks)

### Parallel Training (New!)
Train with multiple parallel environments for faster training:

```bash
python experiments/train.py \
    --algorithm ppo \
    --multiagent \
    --n-envs 8 \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 200000 \
    --run-name "berlin-A-parallel"
```

Note: `--n-envs > 1` uses `SubprocVecEnv` for true parallel execution.

### Recommended Scenarios for Multi-Agent

#### Berlin Scenarios
Each Berlin sub-scenario has multiple intersections:
```bash
# Train on Berlin scenario A (multiple intersections)
python experiments/train.py \
    --algorithm ppo \
    --multiagent \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 200000
```

#### Grid Networks
If you have grid network scenarios (like RESCO grids):
```bash
python experiments/train.py \
    --algorithm dqn \
    --multiagent \
    --scenario-dir scenarios/grid4x4 \
    --net-file grid4x4.net.xml \
    --train-route-file grid4x4.rou.xml \
    --total-timesteps 150000
```

## Command-Line Arguments

### New Arguments
- `--multiagent`: Enable multi-agent mode with parameter sharing
  - When set: uses `sumo_rl.parallel_env()` 
  - When not set: uses `gym.make('sumo-rl-v0')` (single-agent)

### All Arguments
```
# Multi-agent
--multiagent                    Enable multi-agent training

# Parallel training
--n-envs N                      Number of parallel environments (default: 1)
                                Uses SubprocVecEnv when N > 1

# Scenario configuration
--scenario-dir PATH             Path to scenario directory
--net-file NAME                 Network file (.net.xml)
--train-route-file NAME         Training route file
--eval-route-file NAME          Evaluation route file
--auto-duration                 Auto-detect episode duration from routes

# Algorithm
--algorithm {dqn,ppo,a2c}       RL algorithm to use (required)

# Training
--total-timesteps N             Total training timesteps (default: 50000)
--learning-rate FLOAT           Override default learning rate
--normalize                     Apply observation normalization

# Episodes
--episode-seconds N             Training episode duration (default: 16200)
--eval-episode-seconds N        Eval episode duration (default: 5400)

# Output
--output-prefix PREFIX          Prefix for output files
--run-name NAME                 Custom Wandb run name
--gui                          Enable SUMO GUI visualization
```

## How It Works

### Parameter Sharing Approach
```
┌─────────────────────────────────────┐
│  Shared Policy (Neural Network)     │
│  - Same weights for all agents      │
│  - Trained with experiences from    │
│    all traffic signals              │
└─────────────────────────────────────┘
         │         │         │
         ▼         ▼         ▼
    Agent 1   Agent 2   Agent 3
    (TLS A)   (TLS B)   (TLS C)
```

1. All traffic signals observe their local state
2. Each agent queries the same policy for an action
3. All agents step simultaneously
4. Rewards are aggregated (averaged) for policy update
5. Single policy learns to control all intersections

### Advantages
✅ **Scalability**: One policy can control any number of intersections  
✅ **Sample Efficiency**: Learn from multiple agents simultaneously  
✅ **Generalization**: Policy learns patterns across different intersections  
✅ **Simple**: Compatible with single-agent RL algorithms (DQN, PPO, A2C)  

### Limitations
⚠️ **Coordination**: Agents don't explicitly communicate  
⚠️ **Heterogeneity**: Assumes all intersections are similar  
⚠️ **Credit Assignment**: Difficult to attribute success to specific agents  

## Output Files

Multi-agent runs will have modified file names:
- `weights/{prefix}_{algorithm}_model_multiagent_best/`
- `weights/{prefix}_{algorithm}_model_multiagent_final.zip`
- Wandb run name prefixed with `ma-` (e.g., `ma-dqn-test`)

## Monitoring Training

During training, you'll see:
```
Creating multi-agent environment with parameter sharing...
Starting DQN training in MULTI-AGENT mode...
```

Wandb logs will include:
- `num_agents`: Number of traffic signals being controlled
- `total_reward`: Sum of rewards across all agents
- `avg_reward`: Average reward per agent (used for training)
- `episode_rewards`: Per-agent reward breakdown

## Example Training Run

```bash
# Full example: Train PPO on Berlin A with multi-agent and parallel envs
python experiments/train.py \
    --algorithm ppo \
    --multiagent \
    --n-envs 8 \
    --scenario-dir scenarios/berlin-small/A \
    --net-file a.net.xml \
    --train-route-file train_generated.rou.xml \
    --eval-route-file eval_generated.rou.xml \
    --auto-duration \
    --total-timesteps 200000 \
    --normalize \
    --output-prefix "berlin_a" \
    --run-name "ppo-berlin-a-v1"
```

This will create:
- Wandb run: `ma-ppo-berlin-a-v1`
- Model: `weights/berlin_a_ppo_model_multiagent_best/`
- Outputs: `outputs/berlin_a_ppo_queue_run_conn*.csv`

## Comparing Single vs Multi-Agent

Train both modes on the same scenario to compare:

```bash
# Single-agent: controls only ONE traffic light (like old cross_train.py)
python experiments/train.py \
    -a dqn \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 100000 \
    --run-name "berlin-a-single"

# Multi-agent: controls ALL traffic lights with shared policy
python experiments/train.py \
    -a dqn \
    --multiagent \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 100000 \
    --run-name "berlin-a-multi"

# Multi-agent with parallel training (8 environments)
python experiments/train.py \
    -a ppo \
    --multiagent \
    --n-envs 8 \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 100000 \
    --run-name "berlin-a-multi-parallel"
```

Compare the results in Wandb to see which approach performs better!

## Troubleshooting

### "No agents found in environment"
- Make sure your scenario has multiple traffic lights for multi-agent mode
- Check that your `.net.xml` file contains multiple traffic signals

### Observation space mismatch
- Ensure all traffic signals have the same observation space
- The current implementation assumes homogeneous agents

### Performance issues
- Multi-agent may require more timesteps to converge
- Try adjusting learning rate or use PPO for better stability
- Consider using `--normalize` for observation normalization

## Heterogeneous Multi-Agent Evaluation

The `evaluate_multi_agent.py` script allows you to evaluate **independently trained agents** together on a full network. This is useful when you train separate agents on individual intersections and want to see how they perform when coordinating on the complete traffic network.

### Use Case: Berlin Small Network

For the Berlin small scenario, you can:
1. Train individual agents on each intersection (A-J) using their isolated scenarios
2. Evaluate all trained agents together on the full berlin-small network

### Training Individual Intersection Agents

Use the `train_all_berlin.sh` script to train agents on each intersection:

```bash
# Train all intersections A-J with specified algorithm and episodes
./train_all_berlin.sh ppo 50 15

# This trains:
# - scenarios/berlin-small/A with ppo -> weights/berlin_A_ppo_model_best/
# - scenarios/berlin-small/B with ppo -> weights/berlin_B_ppo_model_best/
# - ... and so on for C-J
```

Each intersection is trained independently with its own:
- Network file (e.g., `a.net.xml`)
- Training routes (`train_generated.rou.xml`)
- Evaluation routes (`eval_generated.rou.xml`)

### Evaluating All Agents Together

After training, use `evaluate_all_berlin.sh` to test all agents on the full network:

```bash
# Evaluate PPO agents on full network, 10 episodes
./evaluate_all_berlin.sh ppo 10

# Evaluate DQN agents on full network with GUI
./evaluate_all_berlin.sh dqn 20 --gui

# With Wandb logging
./evaluate_all_berlin.sh a2c 15 --use-wandb --run-name "berlin-heterogeneous-eval"
```

This script:
1. Loads the trained model for each intersection (A-J)
2. Creates a multi-agent environment on the full berlin-small network
3. Maps each traffic light to its corresponding trained model
4. Runs evaluation episodes where each agent uses its own policy

### Manual Evaluation

You can also call the evaluation script directly:

```bash
python experiments/evaluate_multi_agent.py \
    --algorithm ppo \
    --intersections A B C D E F G H I J \
    --net-file scenarios/berlin-small/berlin-small-static.net.xml \
    --route-file scenarios/berlin-small/berlin-small-static-eval.rou.xml \
    --n-episodes 10 \
    --episode-seconds 3600 \
    --output-file evaluation_results/berlin_multiagent_ppo.json
```

### Heterogeneous vs Homogeneous Multi-Agent

**Heterogeneous (evaluate_multi_agent.py)**:
- Each agent has its **own trained policy**
- Agents are trained separately on their local intersections
- Each intersection specializes for its local traffic patterns
- ✅ Better adaptation to intersection-specific conditions
- ❌ No shared learning across intersections

**Homogeneous (train.py --multiagent)**:
- All agents share the **same policy** (parameter sharing)
- Policy is trained on all intersections simultaneously
- ✅ More sample efficient (learns from all agents)
- ✅ Better generalization across similar intersections
- ❌ May not specialize to unique intersection patterns

### Traffic Light ID Mapping

The evaluation script automatically maps intersection letters to their traffic light IDs:

```python
INTERSECTION_TO_TL_ID = {
    'A': 'cluster_29784567_310818818',
    'B': 'cluster_12614600_1860618754_...',
    # ... and so on for C-J
}
```

If your scenario has different traffic light IDs, update this mapping in `evaluate_multi_agent.py`.

### Evaluation Output

The script produces:
- Console output with per-episode metrics
- JSON results file (with `--output-file`)
- Wandb logs (with `--use-wandb`)

Example results:
```
EVALUATION RESULTS
================================================================================
Mean Reward: -125.45 ± 15.32
Mean Episode Length: 720.0 steps

Traffic Metrics:
  Mean Waiting Time: 45.23 ± 3.12 s
  Mean Total Stopped: 12.45 ± 1.23
  Mean Speed: 8.67 ± 0.45 m/s
================================================================================
```

## Next Steps

- Try different algorithms (DQN, PPO, A2C) in multi-agent mode
- Experiment with larger scenarios (more intersections)
- Compare single-agent vs multi-agent vs heterogeneous performance
- Train individual agents on isolated intersections and evaluate together
- Tune hyperparameters for your specific scenario
