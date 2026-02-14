#!/bin/bash
# Evaluate all trained Berlin intersection agents together on the full network
# Usage: ./evaluate_all_berlin.sh [algorithm] [optional: n_episodes] [optional: extra_args]
#
# Examples:
#   ./evaluate_all_berlin.sh ppo              # Evaluate PPO agents, 10 episodes
#   ./evaluate_all_berlin.sh dqn 20           # Evaluate DQN agents, 20 episodes
#   ./evaluate_all_berlin.sh a2c 15 --gui     # Evaluate A2C agents with GUI
#   ./evaluate_all_berlin.sh ppo 10 --compare-baseline  # Compare against baseline

ALGORITHM=${1:-ppo}
N_EPISODES=${2:-10}
EXTRA_ARGS="${@:3}"

export SUMO_PREFIX="$HOME/opt/sumo-1.26.0"
export SUMO_HOME="$SUMO_PREFIX/share/sumo"
export PATH="$SUMO_PREFIX/bin:$PATH"

echo "======================================"
echo "Evaluating Heterogeneous Multi-Agent System"
echo "Algorithm: ${ALGORITHM^^}"
echo "Intersections: A B C D E F G H I J"
echo "Evaluation Episodes: $N_EPISODES"
echo "======================================"
echo ""

# Check if any trained models exist
echo "Checking for trained models..."
MODEL_COUNT=0
for INTERSECTION in A B C D E F G H I J; do
    BEST_MODEL="weights/berlin_${INTERSECTION}_${ALGORITHM}_${ALGORITHM}_model_best/best_model.zip"
    FINAL_MODEL="weights/berlin_${INTERSECTION}_${ALGORITHM}_${ALGORITHM}_model_final.zip"
    if [ -f "$BEST_MODEL" ] || [ -f "$FINAL_MODEL" ]; then
        MODEL_COUNT=$((MODEL_COUNT + 1))
        echo "  OK: Found model for intersection $INTERSECTION"
    else
        echo "  FAILED: No model found for intersection $INTERSECTION"
    fi
done

echo ""
if [ $MODEL_COUNT -eq 0 ]; then
    echo "ERROR: No trained models found!"
    echo "Please train the models first using: ./train_all_berlin.sh"
    exit 1
fi

echo "OK: Found $MODEL_COUNT/10 trained models"
echo ""

# Auto-detect episode duration from evaluation route file
EVAL_ROUTE="scenarios/berlin-small/berlin-small-static-eval.rou.xml"
EPISODE_SECONDS=$(grep 'Total Duration:' "$EVAL_ROUTE" | head -1 | sed -n 's/.*Total Duration: \([0-9]*\)s.*/\1/p')

if [ -z "$EPISODE_SECONDS" ]; then
    echo "WARNING: Could not auto-detect episode duration, using default 3600s"
    EPISODE_SECONDS=3600
else
    HOURS=$(awk "BEGIN {printf \"%.2f\", $EPISODE_SECONDS/3600}")
    echo "Detected episode duration: ${EPISODE_SECONDS}s (${HOURS}h)"
    echo ""
fi

# Create output directory for results
mkdir -p evaluation_results

# Run evaluation
OUTPUT_FILE="evaluation_results/berlin_multiagent_${ALGORITHM}_$(date +%Y%m%d_%H%M%S).json"

echo "Starting evaluation..."
echo "Results will be saved to: $OUTPUT_FILE"
echo ""
echo "Command:"
echo "uv run experiments/evaluate_multi_agent.py \\"
echo "  --algorithm $ALGORITHM \\"
echo "  --intersections A B C D E F G H I J \\"
echo "  --n-episodes $N_EPISODES \\"
echo "  --episode-seconds $EPISODE_SECONDS \\"
echo "  --output-file $OUTPUT_FILE \\"
echo "  $EXTRA_ARGS"
echo ""

uv run experiments/evaluate_multi_agent.py \
    --algorithm "$ALGORITHM" \
    --intersections A B C D E F G H I J \
    --n-episodes "$N_EPISODES" \
    --episode-seconds "$EPISODE_SECONDS" \
    --output-file "$OUTPUT_FILE" \
    $EXTRA_ARGS

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "OK: Evaluation completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "FAILED: Evaluation failed (exit code: $EXIT_CODE)"
    echo "======================================"
    exit $EXIT_CODE
fi
