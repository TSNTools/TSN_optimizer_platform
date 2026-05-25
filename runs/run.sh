mkdir -p new_runs/ed2_alpha05_01

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 1 python -u /home/ws/TSN_optimizer_platform/train.py \
  --config uc_config.cfg \
  --instance /home/ws/TSN_optimizer_platform/banana2.json \
  --out-dir new_runs/ed2_alpha05_01 \
  --train-episodes 60 \
  --timesteps 12000 \
  --decision-points-per-episode 100 \
  --lambda-arrival 0.35 \
  --mu-departure 1.0 \
  --dt 1.0 \
  --max-candidates 10 \
  --max-hops 6 \
  --max-steps 180 \
  --per-hop-candidate-limit 5 \
  --beam-width 30 \
  --top-k-oldest-random 15 \
  --learning-rate 1e-4 \
  --buffer-size 40000 \
  --batch-size 64 \
  --learning-starts 300 \
  --target-update-interval 400 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-initial-eps 1.0 \
  --exploration-fraction 0.35 \
  --exploration-final-eps 0.05 \
  --gamma 0.95 \
  --useless-gap-threshold-bytes 512 \
  --expected-incoming-bytes 512 \
  --defrag-cycle-penalty 0.006 \
  --reallocation-penalty 0.0012 \
  --invalid-penalty 5 \
  --reward-alpha 0.50 \
  --min-reallocations-per-cycle 1 \
  --early-stop-penalty 0.0 \
  --stream-cooldown-decisions 5 \
  > new_runs/ed2_alpha05_01/train.log 2>&1 &