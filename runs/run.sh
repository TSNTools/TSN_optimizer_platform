mkdir -p run_testou_agressive

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 1 python -u /home/ws/TSN_optimizer_platform/train.py \
  --config uc_config.cfg \
  --instance /home/ws/TSN_optimizer_platform/banana3.json \
  --out-dir run_testou_agressive \
  --train-episodes 300 \
  --timesteps 30000 \
  --decision-points-per-episode 100 \
  --lambda-arrival 0.75 \
  --mu-departure 1.0 \
  --dt 1.0 \
  --max-candidates 10 \
  --max-hops 6 \
  --max-steps 200 \
  --per-hop-candidate-limit 5 \
  --beam-width 60 \
  --top-k-oldest-random 10 \
  --learning-rate 1e-4 \
  --buffer-size 50000 \
  --batch-size 64 \
  --learning-starts 1000 \
  --target-update-interval 300 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-initial-eps 1.0 \
  --exploration-fraction 0.4 \
  --exploration-final-eps 0.1 \
  --gamma 0.95 \
  --defrag-cycle-penalty 0.07 \
  --reallocation-penalty 0.02 \
  --invalid-penalty 5 \
  > run_testou_agressive/train.log 2>&1 &