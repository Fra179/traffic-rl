#!/bin/bash
# Train CROSS scenario with all algorithms.
# Usage: ./train_all_cross.sh [optional: episodes] [optional: max_parallel]
#
# Examples:
#   ./train_all_cross.sh           # default 50 episodes, 3 parallel jobs
#   ./train_all_cross.sh 20        # 20 episodes, 3 parallel jobs
#   ./train_all_cross.sh 20 2      # 20 episodes, 2 parallel jobs

NUM_EPISODES=${1:-50}
MAX_PARALLEL=${2:-3}
DELTA_TIME=5
N_ENVS=1

# export CUDA_VISIBLE_DEVICES=""
export SUMO_PREFIX="$HOME/opt/sumo-1.26.0"
export SUMO_HOME="$SUMO_PREFIX/share/sumo"
export PATH="$SUMO_PREFIX/bin:$PATH"

echo "Using SUMO: $(command -v sumo)"
sumo --version | head -n 1

echo "======================================"
echo "Training CROSS Scenario"
echo "Algorithms: A2C, PPO, DQN"
echo "Episodes: $NUM_EPISODES"
echo "Vectorized envs per run (--n-envs): $N_ENVS"
echo "Max Parallel Jobs: $MAX_PARALLEL"
echo "======================================"
echo ""

get_episode_duration() {
    local ROUTE_FILE=$1
    grep 'Total Duration:' "$ROUTE_FILE" | head -1 | sed -n 's/.*Total Duration: \([0-9]*\)s.*/\1/p'
}

ALGORITHMS=(a2c ppo dqn)
SCENARIO_DIR="scenarios/cross"
TRAIN_ROUTE="$SCENARIO_DIR/train_generated.rou.xml"

train_one() {
    local ALGORITHM=$1
    local NUM_EPISODES=$2
    local DELTA_TIME=$3
    local N_ENVS=$4

    local EPISODE_SECONDS
    EPISODE_SECONDS=$(get_episode_duration "$TRAIN_ROUTE")

    if [ -z "$EPISODE_SECONDS" ]; then
        echo "[$(date +%H:%M:%S)] ✗ ERROR: Could not detect episode duration from $TRAIN_ROUTE"
        return 1
    fi

    local STEPS_PER_EPISODE=$((EPISODE_SECONDS / DELTA_TIME))
    local TOTAL_TIMESTEPS=$((NUM_EPISODES * STEPS_PER_EPISODE))

    echo "[$(date +%H:%M:%S)] Starting: $ALGORITHM (${EPISODE_SECONDS}s episodes, $STEPS_PER_EPISODE steps/ep, $TOTAL_TIMESTEPS total steps, n_envs=$N_ENVS)"

    uv run experiments/train.py \
        --algorithm "$ALGORITHM" \
        --scenario-dir "$SCENARIO_DIR" \
        --net-file "cross.net.xml" \
        --train-route-file "train_generated.rou.xml" \
        --eval-route-file "eval_generated.rou.xml" \
        --auto-duration \
        --n-envs "$N_ENVS" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --output-prefix "cross_${ALGORITHM}" \
        --run-name "cross-${ALGORITHM}" \
        > "logs/cross_${ALGORITHM}.log" 2>&1

    local EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ FAILED: $ALGORITHM (exit code: $EXIT_CODE)"
        return 1
    else
        echo "[$(date +%H:%M:%S)] ✓ Completed: $ALGORITHM"
        return 0
    fi
}

export -f train_one
export -f get_episode_duration

mkdir -p logs

TOTAL_JOBS=${#ALGORITHMS[@]}

echo "Total training runs: $TOTAL_JOBS"
echo ""

for ALGORITHM in "${ALGORITHMS[@]}"; do
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done

    train_one "$ALGORITHM" "$NUM_EPISODES" "$DELTA_TIME" "$N_ENVS" &
done

echo ""
echo "Waiting for all training jobs to complete..."
wait

echo ""
echo "======================================"
echo "All CROSS training jobs finished!"
echo "======================================"
echo "Models: weights/"
echo "Logs: logs/cross_*.log"
