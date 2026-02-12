# Multi-Agent Training Guide

This guide explains how to use the updated training script that supports both single-agent and multi-agent traffic light control.

## Overview

The `cross_train_multiagent.py` script now supports:
- **Single-agent mode**: One policy controls one traffic signal (original behavior)
- **Multi-agent mode**: One shared policy controls multiple traffic signals simultaneously (parameter sharing)

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

### Single-Agent Training (Original)
Train a policy to control a single traffic light:

```bash
python experiments/cross_train_multiagent.py \
    --algorithm dqn \
    --scenario-dir scenarios/cross_dynamic \
    --total-timesteps 50000 \
    --run-name "single-agent-test"
```

### Multi-Agent Training (New!)
Train a shared policy to control multiple traffic lights:

```bash
python experiments/cross_train_multiagent.py \
    --algorithm dqn \
    --multiagent \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 100000 \
    --run-name "berlin-A-multiagent"
```

### Recommended Scenarios for Multi-Agent

#### Berlin Scenarios
Each Berlin sub-scenario has multiple intersections:
```bash
# Train on Berlin scenario A (multiple intersections)
python experiments/cross_train_multiagent.py \
    --algorithm ppo \
    --multiagent \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 200000
```

#### Grid Networks
If you have grid network scenarios (like RESCO grids):
```bash
python experiments/cross_train_multiagent.py \
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
# Full example: Train PPO on Berlin A with multi-agent
python experiments/cross_train_multiagent.py \
    --algorithm ppo \
    --multiagent \
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
# Single-agent baseline
python experiments/cross_train_multiagent.py \
    -a dqn \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 100000 \
    --run-name "berlin-a-single"

# Multi-agent with parameter sharing
python experiments/cross_train_multiagent.py \
    -a dqn \
    --multiagent \
    --scenario-dir scenarios/berlin-small/A \
    --auto-duration \
    --total-timesteps 100000 \
    --run-name "berlin-a-multi"
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

## Next Steps

- Try different algorithms (DQN, PPO, A2C) in multi-agent mode
- Experiment with larger scenarios (more intersections)
- Compare single-agent vs multi-agent performance
- Tune hyperparameters for your specific scenario
