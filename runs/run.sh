mkdir -p real/e_feature_safe_filter_mid_2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 1 python -u /home/ws/TSN_optimizer_platform/train.py \
  --config uc_config.cfg \
  --instance /home/ws/TSN_optimizer_platform/banana3.json \
  --out-dir real/e_feature_safe_filter_mid_2 \
  --train-episodes 60 \
  --timesteps 9000 \
  --decision-points-per-episode 100 \
  --lambda-arrival 1.2 \
  --mu-departure 1.0 \
  --dt 1.0 \
  --max-candidates 10 \
  --max-hops 6 \
  --max-steps 300 \
  --per-hop-candidate-limit 5 \
  --beam-width 60 \
  --top-k-oldest-random 10 \
  --learning-rate 1e-4 \
  --buffer-size 30000 \
  --batch-size 64 \
  --learning-starts 300 \
  --target-update-interval 400 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-initial-eps 1.0 \
  --exploration-fraction 0.45 \
  --exploration-final-eps 0.05 \
  --gamma 0.95 \
  --defrag-cycle-penalty 0.005 \
  --reallocation-penalty 0.001 \
  --invalid-penalty 5 \
  --reward-alpha 0.7 \
  --min-reallocations-per-cycle 5 \
  --early-stop-penalty 0.20 \
  --stream-cooldown-decisions 3 \
  > real/e_feature_safe_filter_mid_2/train.log 2>&1 &