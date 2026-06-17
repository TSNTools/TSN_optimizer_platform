OUT_DIR="clean/tmskhir9"
mkdir -p "$OUT_DIR"

nohup bash -c '
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 2 python -u /home/ws/TSN_optimizer_platform/train.py \
  --config uc_config.cfg \
  --instance /home/ws/TSN_optimizer_platform/3imla9.json \
  --out-dir clean/tmskhir9 \
  --train-episodes 400 \
  --timesteps 300 \
  --decision-points-per-episode 80 \
  --lambda-arrival 0.25 \
  --mu-departure 1.0 \
  --dt 1.0 \
  --max-candidates 10 \
  --max-steps 500 \
  --per-hop-candidate-limit 5 \
  --beam-width 30 \
  --reward-alpha 0.5 \
  --invalid-penalty 1.0 \
  --defrag-cycle-penalty 0.07 \
  --reallocation-penalty 0.005 \
  --min-reallocations-per-cycle 5 \
  --early-stop-penalty 0.0 \
  --stream-cooldown-decisions 5 \
  --learning-rate 1e-4 \
  --buffer-size 30000 \
  --batch-size 64 \
  --learning-starts 200 \
  --target-update-interval 500 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-fraction 0.35 \
  --exploration-initial-eps 1.0 \
  --exploration-final-eps 0.05 \
  --gamma 0.97 \
  --seed 2004 \
  --debug-reward-steps 0
' > "$OUT_DIR/train.log" 2>&1 &