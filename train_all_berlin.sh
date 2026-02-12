#!/bin/bash
# Train all Berlin intersections A-J with all algorithms in parallel
# Usage: ./train_all_berlin.sh [optional: timesteps] [optional: max_parallel]
#
# Examples:
#   ./train_all_berlin.sh              # Train all with default 50k timesteps, 8 parallel
#   ./train_all_berlin.sh 100000       # Train all with 100k timesteps, 8 parallel
#   ./train_all_berlin.sh 50000 4      # Train all with 50k timesteps, 4 parallel

TIMESTEPS=${1:-50000}
MAX_PARALLEL=${2:-16}

export CUDA_VISIBLE_DEVICES=""

echo "======================================"
echo "Training All Berlin Intersections"
echo "Algorithms: DQN, PPO, A2C"
echo "Timesteps: $TIMESTEPS"
echo "Max Parallel Jobs: $MAX_PARALLEL"
echo "======================================"
echo ""

# Arrays
ALGORITHMS=(dqn ppo a2c)
INTERSECTIONS=(A C H I J)

# Function to train a single intersection with an algorithm
train_one() {
    local ALGORITHM=$1
    local INTERSECTION=$2
    local TIMESTEPS=$3
    
    # Convert intersection letter to lowercase for file naming
    local INTERSECTION_LOWER=$(echo "$INTERSECTION" | tr '[:upper:]' '[:lower:]')
    
    echo "[$(date +%H:%M:%S)] Starting: $ALGORITHM on Intersection $INTERSECTION"
    
    uv run experiments/cross_train.py \
        --algorithm "$ALGORITHM" \
        --scenario-dir "scenarios/berlin-small/$INTERSECTION" \
        --net-file "${INTERSECTION_LOWER}.net.xml" \
        --train-route-file "train_generated.rou.xml" \
        --eval-route-file "eval_generated.rou.xml" \
        --auto-duration \
        --total-timesteps "$TIMESTEPS" \
        --output-prefix "berlin_${INTERSECTION}_${ALGORITHM}" \
        --run-name "berlin-${INTERSECTION}-${ALGORITHM}" \
        > "logs/berlin_${INTERSECTION}_${ALGORITHM}.log" 2>&1
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ FAILED: $ALGORITHM on Intersection $INTERSECTION (exit code: $EXIT_CODE)"
        return 1
    else
        echo "[$(date +%H:%M:%S)] ✓ Completed: $ALGORITHM on Intersection $INTERSECTION"
        return 0
    fi
}

# Export function so it's available in subshells
export -f train_one

# Create logs directory if it doesn't exist
mkdir -p logs

# Job control: track running jobs
TOTAL_JOBS=$((${#ALGORITHMS[@]} * ${#INTERSECTIONS[@]}))
COMPLETED=0
FAILED=0

echo "Total training runs: $TOTAL_JOBS (${#ALGORITHMS[@]} algorithms × ${#INTERSECTIONS[@]} intersections)"
echo ""

# Generate all training jobs and run them with parallel limit
for ALGORITHM in "${ALGORITHMS[@]}"; do
    for INTERSECTION in "${INTERSECTIONS[@]}"; do
        # Wait if we've hit the parallel limit
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
            sleep 1
        done
        
        # Start training in background
        train_one "$ALGORITHM" "$INTERSECTION" "$TIMESTEPS" &
    done
done

# Wait for all background jobs to complete
echo ""
echo "Waiting for all training jobs to complete..."
wait

echo ""
echo "======================================"
echo "All training jobs finished!"
echo "======================================"
echo ""
echo "Trained models saved in: weights/"
echo "Training logs saved in: logs/"
echo "Results logged to Wandb project: rl-traffic-light"
