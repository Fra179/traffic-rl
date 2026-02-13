# Heterogeneous Multi-Agent Setup - Summary

This document summarizes the new heterogeneous multi-agent evaluation system for the traffic-rl project.

## What Was Created

### 1. Multi-Agent Evaluation Script: `experiments/evaluate_multi_agent.py`

A comprehensive evaluation script that:
- Loads **independently trained agents** for different intersections
- Creates a **heterogeneous multi-agent environment** where each agent uses its own policy
- Maps traffic lights to their corresponding trained models
- Evaluates system-wide performance metrics

**Key Features:**
- Automatic traffic light ID mapping for Berlin intersections (A-J)
- Support for DQN, PPO, and A2C algorithms
- Wandb integration for experiment tracking
- JSON output for results storage
- SUMO GUI support for visualization

### 2. Evaluation Shell Script: `evaluate_all_berlin.sh`

A convenient wrapper script that:
- Checks for trained models before evaluation
- Auto-detects episode duration from route files
- Manages SUMO environment variables
- Creates timestamped result files
- Provides clear status messages

**Usage:**
```bash
./evaluate_all_berlin.sh [algorithm] [episodes] [extra_args]

# Examples
./evaluate_all_berlin.sh ppo 10
./evaluate_all_berlin.sh dqn 20 --gui
./evaluate_all_berlin.sh a2c 15 --use-wandb
```

### 3. Documentation

- **`HETEROGENEOUS_EVAL_GUIDE.md`**: Complete guide for heterogeneous multi-agent evaluation
- **Updates to `experiments/TRAINING_README.md`**: Added section on heterogeneous evaluation

## Architecture Overview

### Heterogeneous Multi-Agent System

```
┌─────────────────────────────────────────────────────┐
│           Full Berlin-Small Network                  │
│                                                      │
│  TL_A  ←→  Model_A (trained on intersection A)     │
│  TL_B  ←→  Model_B (trained on intersection B)     │
│  TL_C  ←→  Model_C (trained on intersection C)     │
│  ...                                                │
│  TL_J  ←→  Model_J (trained on intersection J)     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

Each agent:
1. Observes its local state (queue lengths, waiting times, etc.)
2. Queries its own trained model for an action
3. Executes the action (changes traffic light phase)
4. All agents step simultaneously

## Workflow

### Training Phase

1. **Extract Individual Intersections**: Each intersection (A-J) has its own scenario
   ```
   scenarios/berlin-small/A/  # Isolated intersection A
   scenarios/berlin-small/B/  # Isolated intersection B
   ...
   ```

2. **Train Individual Agents**: Use `train_all_berlin.sh`
   ```bash
   ./train_all_berlin.sh ppo 50 15
   ```
   
   This trains each agent independently:
   - Agent A learns optimal control for intersection A
   - Agent B learns optimal control for intersection B
   - etc.

3. **Save Models**: Each agent's model is saved separately
   ```
   weights/berlin_A_ppo_model_best/best_model.zip
   weights/berlin_B_ppo_model_best/best_model.zip
   ...
   ```

### Evaluation Phase

1. **Load All Models**: Load each agent's trained model

2. **Create Full Network Environment**: Set up PettingZoo environment on complete network

3. **Map Traffic Lights**: Link each TL in the network to its trained model
   ```python
   INTERSECTION_TO_TL_ID = {
       'A': 'cluster_29784567_310818818',
       'B': 'cluster_12614600_1860618754_...',
       # etc.
   }
   ```

4. **Run Episodes**: Evaluate agents working together
   - Each agent uses its own policy
   - System metrics are collected (waiting time, throughput, etc.)

5. **Analyze Results**: Compare performance vs baselines

## Key Components

### HeterogeneousMultiAgentWrapper Class

```python
class HeterogeneousMultiAgentWrapper(gym.Env):
    """
    Wrapper for heterogeneous multi-agent evaluation.
    Each agent uses its own trained policy (no parameter sharing).
    """
    
    def __init__(self, pz_env, agent_models_map):
        # Maps each agent_id to its trained model
        self.agent_models_map = agent_models_map
        
    def step(self, action=None):
        # Compute actions for each agent using its own model
        for agent in self.agents:
            if agent in self.agent_models_map:
                obs = observations[agent]
                action, _ = self.agent_models_map[agent].predict(obs)
                actions[agent] = action
```

### Traffic Light ID Mapping

The mapping between intersection letters and traffic light IDs:

```python
INTERSECTION_TO_TL_ID = {
    'A': 'cluster_29784567_310818818',
    'B': 'cluster_12614600_1860618754_1860618762_...',  # Long cluster ID
    'C': 'cluster_28150269_29271707_4377814009_...',
    # ... for all intersections A-J
}
```

**Note**: Intersections B and G share the same traffic light ID in the full network.

## Comparison with Other Approaches

### 1. Heterogeneous Multi-Agent (New)
```bash
./train_all_berlin.sh ppo 50 15  # Train individual agents
./evaluate_all_berlin.sh ppo 10  # Evaluate together
```
- ✅ Each agent specializes for its intersection
- ✅ Parallel training across intersections
- ✅ Robust to heterogeneous designs
- ❌ No shared learning between agents

### 2. Homogeneous Multi-Agent (Existing)
```bash
uv run experiments/train.py --multiagent --algorithm ppo ...
```
- ✅ Parameter sharing - all agents use same policy
- ✅ Sample efficient - learns from all agents
- ✅ Good for similar intersections
- ❌ May not specialize to unique patterns

### 3. Single-Agent (Baseline)
```bash
uv run experiments/train.py --algorithm ppo ...
```
- ✅ Simple and straightforward
- ❌ Only controls one intersection
- ❌ Not suitable for multi-intersection scenarios

## Usage Examples

### Basic Evaluation

```bash
# After training with train_all_berlin.sh
./evaluate_all_berlin.sh ppo 10
```

### Evaluation with Visualization

```bash
./evaluate_all_berlin.sh ppo 5 --gui
```

### Advanced Evaluation

```bash
python experiments/evaluate_multi_agent.py \
    --algorithm ppo \
    --intersections A B C D E \  # Evaluate subset
    --n-episodes 20 \
    --episode-seconds 1800 \
    --use-wandb \
    --run-name "berlin-subset-eval" \
    --output-file results.json \
    --gui
```

## Expected Results

The evaluation provides:

1. **Per-episode metrics**:
   - Episode reward
   - Episode length
   - Waiting time
   - Stopped vehicles
   - Average speed

2. **Aggregated statistics**:
   - Mean ± std for all metrics
   - System-wide performance

3. **Output files**:
   - JSON file with complete results
   - Wandb logs (if enabled)

Example output:
```
================================================================================
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

## Files Created/Modified

### New Files
- `experiments/evaluate_multi_agent.py` - Main evaluation script
- `evaluate_all_berlin.sh` - Convenience wrapper
- `HETEROGENEOUS_EVAL_GUIDE.md` - Complete usage guide
- `SETUP_SUMMARY.md` - This file

### Modified Files
- `experiments/TRAINING_README.md` - Added heterogeneous evaluation section

### Generated Files (during use)
- `evaluation_results/*.json` - Evaluation results
- Console output and logs

## Next Steps

1. **Train the agents**: Run `./train_all_berlin.sh ppo 50 15`
2. **Wait for completion**: Monitor `logs/` directory
3. **Evaluate**: Run `./evaluate_all_berlin.sh ppo 10`
4. **Analyze**: Review results and compare with baselines
5. **Iterate**: Try different algorithms, hyperparameters, or training durations

## Troubleshooting

### No models found
- Run training first: `./train_all_berlin.sh`
- Check `weights/` directory for model files

### SUMO errors
- Verify SUMO installation: `sumo --version`
- Check `SUMO_HOME` variable: `echo $SUMO_HOME`

### Import errors
- Use `uv run` instead of plain `python`
- Check dependencies: `uv pip list`

### Out of memory
- Reduce parallel jobs during training
- Evaluate with fewer episodes
- Close GUI when not needed

## References

- **Training script**: `train_all_berlin.sh`
- **Evaluation script**: `evaluate_all_berlin.sh`
- **Python evaluation**: `experiments/evaluate_multi_agent.py`
- **Training guide**: `experiments/TRAINING_README.md`
- **Eval guide**: `HETEROGENEOUS_EVAL_GUIDE.md`
- **Main README**: `README.md`

---

**Created**: 2026-02-13  
**Purpose**: Enable evaluation of independently trained agents on full network  
**Status**: Ready to use after training completion
