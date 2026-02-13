# Heterogeneous Multi-Agent Evaluation Guide

This guide explains how to train individual agents on separate intersections and evaluate them together on the full network.

## Quick Start

### 1. Train Individual Intersection Agents

Train agents on each isolated intersection (A-J):

```bash
# Train all intersections with PPO for 50 episodes (max 15 parallel jobs)
./train_all_berlin.sh ppo 50 15

# Or train with a different algorithm
./train_all_berlin.sh dqn 30 10
./train_all_berlin.sh a2c 40 8
```

This will create model files:
- `weights/berlin_A_ppo_model_best/best_model.zip`
- `weights/berlin_B_ppo_model_best/best_model.zip`
- ... (for all intersections A-J)

### 2. Evaluate All Agents Together

After training completes, evaluate the agents on the full berlin-small network:

```bash
# Evaluate PPO agents for 10 episodes
./evaluate_all_berlin.sh ppo 10

# Evaluate with SUMO GUI visualization
./evaluate_all_berlin.sh ppo 5 --gui

# Evaluate and log to Wandb
./evaluate_all_berlin.sh dqn 20 --use-wandb --run-name "berlin-eval-v1"
```

## Architecture

### Heterogeneous Multi-Agent System

Each traffic light is controlled by its own independently trained agent:

```
Full Berlin Network
├── Intersection A → Model trained on scenarios/berlin-small/A
├── Intersection B → Model trained on scenarios/berlin-small/B
├── Intersection C → Model trained on scenarios/berlin-small/C
├── ...
└── Intersection J → Model trained on scenarios/berlin-small/J
```

**Benefits:**
- ✅ Each agent specializes for its local traffic patterns
- ✅ Training can be parallelized across intersections
- ✅ Individual models can be updated independently
- ✅ Robust to heterogeneous intersection designs

**Workflow:**
1. Extract individual intersections from full network (A-J)
2. Train each agent independently on its intersection
3. Load all trained models
4. Evaluate on full network where each agent controls its intersection

## Training Details

### Intersection Scenarios

Each intersection directory contains:
- `{letter}.net.xml` - Network file with single intersection
- `train_generated.rou.xml` - Training traffic patterns
- `eval_generated.rou.xml` - Evaluation traffic patterns

Example for intersection A:
```
scenarios/berlin-small/A/
├── a.net.xml
├── train_generated.rou.xml
├── eval_generated.rou.xml
└── ...
```

### Training Command

The `train_all_berlin.sh` script runs training for each intersection:

```bash
uv run experiments/train.py \
    --algorithm ppo \
    --scenario-dir scenarios/berlin-small/A \
    --net-file a.net.xml \
    --train-route-file train_generated.rou.xml \
    --eval-route-file eval_generated.rou.xml \
    --auto-duration \
    --total-timesteps <calculated> \
    --output-prefix berlin_A_ppo \
    --run-name berlin-A-ppo
```

## Evaluation Details

### Evaluation Command

The `evaluate_all_berlin.sh` script loads all models and evaluates them:

```bash
uv run experiments/evaluate_multi_agent.py \
    --algorithm ppo \
    --intersections A B C D E F G H I J \
    --net-file scenarios/berlin-small/berlin-small-static.net.xml \
    --route-file scenarios/berlin-small/berlin-small-static-eval.rou.xml \
    --n-episodes 10 \
    --episode-seconds 3600 \
    --output-file evaluation_results/berlin_multiagent_ppo.json
```

### What Happens During Evaluation

1. **Model Loading**: Loads trained model for each intersection
   - Prefers `best_model.zip` over `final_model.zip`
   - Warns if models are missing

2. **Traffic Light Mapping**: Maps models to traffic lights using IDs
   ```python
   INTERSECTION_TO_TL_ID = {
       'A': 'cluster_29784567_310818818',
       'B': 'cluster_12614600_1860618754_...',
       # ... etc
   }
   ```

3. **Environment Setup**: Creates PettingZoo multi-agent environment
   - Full berlin-small network
   - Evaluation route file
   - All traffic lights active

4. **Evaluation Loop**: For each episode:
   - Each agent observes its local state
   - Each agent queries its own trained model
   - All agents step simultaneously
   - System metrics are collected

5. **Results**: Displays aggregated metrics:
   - Mean reward across episodes
   - Mean waiting time
   - Mean stopped vehicles
   - Mean speed

### Output Files

Evaluation creates:
- `evaluation_results/berlin_multiagent_{algorithm}_{timestamp}.json`
  - Contains configuration, results, and per-episode metrics
  - Can be used for further analysis

Example output structure:
```json
{
  "config": {
    "algorithm": "ppo",
    "intersections": ["A", "B", "C", ...],
    "n_episodes": 10
  },
  "results": {
    "mean_reward": -125.45,
    "mean_waiting_time": 45.23,
    "mean_speed": 8.67
  },
  "episode_rewards": [...]
}
```

## Comparison with Other Approaches

### 1. Heterogeneous Multi-Agent (This Approach)
- Each agent has its own trained policy
- Agents trained independently on isolated intersections
- **Use when**: Intersections are very different, or you want specialization

### 2. Homogeneous Multi-Agent (Parameter Sharing)
- All agents share the same policy
- Train with `train.py --multiagent`
- **Use when**: Intersections are similar, want faster training

### 3. Single-Agent
- One agent controls one intersection
- Train with `train.py` (default)
- **Use when**: Only one intersection in scenario

### Performance Comparison

You can compare all three approaches:

```bash
# 1. Train heterogeneous (this approach)
./train_all_berlin.sh ppo 50 15
./evaluate_all_berlin.sh ppo 10 --output-file results_heterogeneous.json

# 2. Train homogeneous (parameter sharing)
uv run experiments/train.py --algorithm ppo --multiagent \
    --scenario-dir scenarios/berlin-small \
    --net-file berlin-small-static.net.xml \
    --train-route-file berlin-small-static-train.rou.xml \
    --auto-duration --total-timesteps 500000

# 3. Baseline (fixed-time signals)
# Automatically computed during evaluation with --compare-baseline
```

## Advanced Usage

### Selective Intersection Evaluation

Evaluate only specific intersections:

```bash
python experiments/evaluate_multi_agent.py \
    --algorithm ppo \
    --intersections A B C \  # Only evaluate A, B, C
    --n-episodes 10 \
    --net-file scenarios/berlin-small/berlin-small-static.net.xml \
    --route-file scenarios/berlin-small/berlin-small-static-eval.rou.xml
```

### Using Different Model Directories

If models are in a different location:

```bash
python experiments/evaluate_multi_agent.py \
    --algorithm ppo \
    --model-dir /path/to/models \
    --model-prefix custom_prefix \
    --intersections A B C D E F G H I J
```

### Visualization

Run with SUMO GUI to see the agents in action:

```bash
./evaluate_all_berlin.sh ppo 5 --gui
```

### Wandb Integration

Log evaluation metrics to Wandb:

```bash
./evaluate_all_berlin.sh ppo 10 --use-wandb --run-name "berlin-heterogeneous-v1"
```

## Troubleshooting

### No models found
```
❌ ERROR: No trained models found!
```
- Make sure you've run `./train_all_berlin.sh` first
- Check that model files exist in `weights/` directory
- Verify the algorithm matches (ppo, dqn, or a2c)

### Missing models for some intersections
```
[C] WARNING: No model found for C
```
- Some intersections might have failed training
- Check log files in `logs/` directory
- Evaluation will use default action (phase 0) for missing models

### SUMO errors
```
Error: Could not load network file
```
- Verify SUMO installation: `sumo --version`
- Check `SUMO_HOME` environment variable
- Ensure network files exist in scenario directories

### Memory issues
- Reduce number of parallel training jobs: `./train_all_berlin.sh ppo 50 5`
- Evaluate with fewer episodes: `./evaluate_all_berlin.sh ppo 5`
- Close GUI when not needed (remove `--gui` flag)

## Files Created

### Training
```
weights/
├── berlin_A_ppo_model_best/
│   └── best_model.zip
├── berlin_A_ppo_model_final.zip
├── berlin_B_ppo_model_best/
│   └── best_model.zip
├── ...
logs/
├── berlin_A_ppo.log
├── berlin_B_ppo.log
└── ...
```

### Evaluation
```
evaluation_results/
└── berlin_multiagent_ppo_20260213_120000.json
```

## Next Steps

1. **Analyze Results**: Compare metrics across different algorithms
2. **Visualize Performance**: Use GUI mode to watch agents in action
3. **Fine-tune**: Adjust training parameters for better performance
4. **Scale Up**: Apply to larger networks with more intersections
5. **Compare Approaches**: Evaluate heterogeneous vs homogeneous multi-agent

## References

- Main training documentation: [experiments/TRAINING_README.md](experiments/TRAINING_README.md)
- Multi-agent evaluation script: [experiments/evaluate_multi_agent.py](experiments/evaluate_multi_agent.py)
- Training script: [train_all_berlin.sh](train_all_berlin.sh)
- Evaluation script: [evaluate_all_berlin.sh](evaluate_all_berlin.sh)
