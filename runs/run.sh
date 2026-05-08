mkdir -p run_32_env_v2_frss2_focus_penalty_100

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 5 python -u /home/ws/TSN_optimizer_platform/train.py \
  --config uc_config.cfg \
  --instance /home/ws/TSN_optimizer_platform/banana3.json \
  --out-dir run_32_env_v2_frss2_focus_penalty_100 \
  --train-episodes 50 \
  --timesteps 2500 \
  --decision-points-per-episode 100 \
  --lambda-arrival 0.75 \
  --mu-departure 1.0 \
  --dt 1.0 \
  --max-candidates 5 \
  --max-hops 6 \
  --max-steps 500 \
  --per-hop-candidate-limit 5 \
  --beam-width 30 \
  --top-k-oldest-random 10 \
  --learning-rate 1e-4 \
  --buffer-size 100000 \
  --batch-size 64 \
  --learning-starts 1000 \
  --target-update-interval 1000 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-initial-eps 1.0 \
  --exploration-fraction 0.2 \
  --exploration-final-eps 0.05 \
  --gamma 0.99 \
  --defrag-cycle-penalty 0.25 \
  --reallocation-penalty 0.05 \
  --invalid-penalty 10 \
  > run_32_env_v2_frss2_focus_penalty_100/train.log 2>&1 &