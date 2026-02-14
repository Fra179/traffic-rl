#!/bin/bash
# Evaluate cross-trained single-agent models on grid2x2 scenario.
# Usage: ./evaluate_all_cross_2x2.sh [optional: n_episodes] [optional: extra_args]
#
# Examples:
#   ./evaluate_all_cross_2x2.sh
#   ./evaluate_all_cross_2x2.sh 20
#   ./evaluate_all_cross_2x2.sh 10 --gui

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

N_EPISODES=${1:-1}
EXTRA_ARGS="${@:2}"

export SUMO_PREFIX="$HOME/opt/sumo-1.26.0"
export SUMO_HOME="$SUMO_PREFIX/share/sumo"
export PATH="$SUMO_PREFIX/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""

SCENARIO_DIR="scenarios/grid2x2"
NET_FILE="grid2x2.net.xml"
ROUTE_FILE="eval_generated.rou.xml"

ALGORITHMS=(a2c ppo dqn)

mkdir -p "$SCRIPT_DIR/evaluation_results"

echo "======================================"
echo "Evaluating CROSS-trained models on GRID2x2"
echo "Algorithms: A2C, PPO, DQN"
echo "Episodes: $N_EPISODES"
echo "Scenario: $SCENARIO_DIR"
echo "======================================"
echo ""

EVAL_ROUTE="$SCENARIO_DIR/$ROUTE_FILE"
EPISODE_SECONDS=$(grep 'Total Duration:' "$EVAL_ROUTE" | head -1 | sed -n 's/.*Total Duration: \([0-9]*\)s.*/\1/p')
if [ -z "$EPISODE_SECONDS" ]; then
    echo "⚠ Warning: Could not auto-detect episode duration from $EVAL_ROUTE, using 3600s"
    EPISODE_SECONDS=3600
else
    HOURS=$(awk "BEGIN {printf \"%.2f\", $EPISODE_SECONDS/3600}")
    echo "Detected evaluation duration: ${EPISODE_SECONDS}s (${HOURS}h)"
fi

echo ""

for ALGORITHM in "${ALGORITHMS[@]}"; do
    BEST_MODEL="weights/cross_${ALGORITHM}_${ALGORITHM}_model_best/best_model.zip"
    FINAL_MODEL="weights/cross_${ALGORITHM}_${ALGORITHM}_model_final.zip"

    MODEL_PATH=""
    if [ -f "$BEST_MODEL" ]; then
        MODEL_PATH="$BEST_MODEL"
    elif [ -f "$FINAL_MODEL" ]; then
        MODEL_PATH="$FINAL_MODEL"
    else
        echo "[$(date +%H:%M:%S)] ✗ Skipping ${ALGORITHM^^}: no model found (expected $BEST_MODEL or $FINAL_MODEL)"
        continue
    fi

    OUTPUT_CSV="$SCRIPT_DIR/evaluation_results/cross_to_grid2x2_${ALGORITHM}_$(date +%Y%m%d_%H%M%S).csv"

    echo "[$(date +%H:%M:%S)] Starting ${ALGORITHM^^}"
    echo "  model: $MODEL_PATH"
    echo "  output: $OUTPUT_CSV"

    uv run experiments/evaluate.py \
        "$MODEL_PATH" \
        --algorithm "$ALGORITHM" \
        --multiagent \
        --compare-baseline \
        --delta-time 5 \
        --n-episodes "$N_EPISODES" \
        --episode-seconds "$EPISODE_SECONDS" \
        --scenario-dir "$SCENARIO_DIR" \
        --net-file "$NET_FILE" \
        --route-file "$ROUTE_FILE" \
        --output-csv "$OUTPUT_CSV" \
        $EXTRA_ARGS

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ FAILED ${ALGORITHM^^} (exit $EXIT_CODE)"
    else
        echo "[$(date +%H:%M:%S)] ✓ Completed ${ALGORITHM^^}"
    fi
    echo ""
done

echo "Done. Results in $SCRIPT_DIR/evaluation_results/"
