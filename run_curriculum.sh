#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$HOME/TSN_optimizer_platform"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
SCRIPT_NAME="curriculum_train_drl.py"

OUT_DIR="outputs_curriculum_size_only"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/curriculum_size_only_${TIMESTAMP}.log"

cd "$PROJECT_DIR"

echo "Starting curriculum run..."
echo "Project dir : $PROJECT_DIR"
echo "Log file    : $LOG_FILE"

nohup "$PYTHON_BIN" "$SCRIPT_NAME" \
  --config uc_config.cfg \
  --train-instances instance1.json \
  --seed 42 \
  --out-dir "$OUT_DIR" \
  --reward-mode hole_improvement_binary \
  --easy-incoming 500 \
  --easy-frame-size 512 \
  --easy-timesteps 15000 \
  --medium-incoming 500 \
  --medium-frame-size 1500 \
  --medium-timesteps 15000 \
  --hard-incoming 500 \
  --hard-frame-size 2500 \
  --hard-timesteps 20000 \
  --eval-steps 500 \
  --reward-ma-window 100 \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Launched with PID: $PID"
echo "Logs: tail -f $LOG_FILE"
echo "Stop: kill $PID"