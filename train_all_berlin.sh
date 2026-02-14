#!/bin/bash
# Train all Berlin intersections A-J with all algorithms in parallel
# Usage: ./train_all_berlin.sh [optional: episodes] [optional: max_parallel]
#
# Examples:
#   ./train_all_berlin.sh              # Train all with default 10 episodes, 8 parallel
#   ./train_all_berlin.sh 20           # Train all with 20 episodes, 8 parallel
#   ./train_all_berlin.sh 10 4         # Train all with 10 episodes, 4 parallel

NUM_EPISODES=${1:-50}
MAX_PARALLEL=${2:-15}
DELTA_TIME=5  # Default delta_time from cross_train.py

# export CUDA_VISIBLE_DEVICES=""
export SUMO_PREFIX="$HOME/opt/sumo-1.26.0"
export SUMO_HOME="$SUMO_PREFIX/share/sumo"
export PATH="$SUMO_PREFIX/bin:$PATH"

# opzionale: stampa una volta per conferma
echo "Using SUMO: $(command -v sumo)"
sumo --version | head -n 1

echo "======================================"
echo "Training All Berlin Intersections"
echo "Algorithms: DQN, PPO, A2C"
echo "Episodes per intersection: $NUM_EPISODES"
echo "Delta time: ${DELTA_TIME}s"
echo "Max Parallel Jobs: $MAX_PARALLEL"
echo "======================================"
echo ""

# Function to get episode duration from route file
get_episode_duration() {
    local ROUTE_FILE=$1
    grep 'Total Duration:' "$ROUTE_FILE" | head -1 | sed -n 's/.*Total Duration: \([0-9]*\)s.*/\1/p'
}

# Arrays
ALGORITHMS=(dqn ppo a2c)
INTERSECTIONS=(A B C D E F G H I J)

# Function to train a single intersection with an algorithm
train_one() {
    local ALGORITHM=$1
    local INTERSECTION=$2
    local NUM_EPISODES=$3
    local DELTA_TIME=$4
    
    # Convert intersection letter to lowercase for file naming
    local INTERSECTION_LOWER=$(echo "$INTERSECTION" | tr '[:upper:]' '[:lower:]')
    
    # Get episode duration from training route file
    local TRAIN_ROUTE="scenarios/berlin-small/$INTERSECTION/train_generated.rou.xml"
    local EPISODE_SECONDS=$(get_episode_duration "$TRAIN_ROUTE")
    
    if [ -z "$EPISODE_SECONDS" ]; then
        echo "[$(date +%H:%M:%S)] ERROR: Could not detect episode duration for $INTERSECTION"
        return 1
    fi
    
    # Calculate total timesteps: episodes * (episode_seconds / delta_time)
    local STEPS_PER_EPISODE=$((EPISODE_SECONDS / DELTA_TIME))
    local TOTAL_TIMESTEPS=$((NUM_EPISODES * STEPS_PER_EPISODE))
    
    echo "[$(date +%H:%M:%S)] Starting: $ALGORITHM on $INTERSECTION (${EPISODE_SECONDS}s episodes, $STEPS_PER_EPISODE steps/ep, $TOTAL_TIMESTEPS total steps)"
    
    uv run experiments/train.py \
        --algorithm "$ALGORITHM" \
        --scenario-dir "scenarios/berlin-small/$INTERSECTION" \
        --net-file "${INTERSECTION_LOWER}.net.xml" \
        --train-route-file "train_generated.rou.xml" \
        --eval-route-file "eval_generated.rou.xml" \
        --auto-duration \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --output-prefix "berlin_${INTERSECTION}_${ALGORITHM}" \
        --run-name "berlin-${INTERSECTION}-${ALGORITHM}" \
        > "logs/berlin_${INTERSECTION}_${ALGORITHM}.log" 2>&1
    
    local EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] FAILED: $ALGORITHM on Intersection $INTERSECTION (exit code: $EXIT_CODE)"
        return 1
    else
        echo "[$(date +%H:%M:%S)] OK: Completed $ALGORITHM on Intersection $INTERSECTION"
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
echo "Each run will train for $NUM_EPISODES episodes (timesteps vary by episode duration)"
echo ""

# Generate all training jobs and run them with parallel limit
for ALGORITHM in "${ALGORITHMS[@]}"; do
    for INTERSECTION in "${INTERSECTIONS[@]}"; do
        # Wait if we've hit the parallel limit
        while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
            sleep 1
        done
        
        # Start training in background
        train_one "$ALGORITHM" "$INTERSECTION" "$NUM_EPISODES" "$DELTA_TIME" &
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
