mkdir -p new_runs/erasmus_1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 1 python -u /home/ws/TSN_optimizer_platform/train.py \
  --config uc_config.cfg \
  --instance /home/ws/TSN_optimizer_platform/banana2.json \
  --out-dir new_runs/erasmus_1 \
  --train-episodes 300 \
  --timesteps 20000 \
  --decision-points-per-episode 100 \
  --lambda-arrival 0.35 \
  --mu-departure 1.0 \
  --dt 1.0 \
  --max-candidates 10 \
  --max-hops 6 \
  --max-steps 100 \
  --per-hop-candidate-limit 5 \
  --beam-width 30 \
  --top-k-oldest-random 15 \
  --learning-rate 1e-4 \
  --buffer-size 20000 \
  --batch-size 64 \
  --learning-starts 150 \
  --target-update-interval 250 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-initial-eps 1.0 \
  --exploration-fraction 0.45 \
  --exploration-final-eps 0.05 \
  --gamma 0.95 \
  --useless-gap-threshold-bytes 512 \
  --expected-incoming-bytes 512 \
  --defrag-cycle-penalty 0.15 \
  --reallocation-penalty 0.025 \
  --invalid-penalty 5 \
  --reward-alpha 0.50 \
  --min-reallocations-per-cycle 1 \
  --early-stop-penalty 0.0 \
  --stream-cooldown-decisions 5 \
  > new_runs/erasmus_1/train.log 2>&1 &