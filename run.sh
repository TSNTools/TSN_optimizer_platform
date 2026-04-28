mkdir -p run_07

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 6 python -u train.py \
  --config uc_config.cfg \
  --instance banana2.json \
  --out-dir run_07 \
  --train-episodes 500 \
  --timesteps 100000 \
  --decision-points-per-episode 130 \
  --lambda-arrival 1 \
  --mu-departure 1.3 \
  --dt 1.0 \
  --max-candidates 5 \
  --max-steps 500 \
  --learning-rate 1e-4 \
  --buffer-size 100000 \
  --batch-size 64 \
  --learning-starts 1000 \
  --target-update-interval 1000 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-initial-eps 1.0 \
  --exploration-final-eps 0.05 \
  --gamma 0.99 \
  --defrag-cycle-penalty 0.3 \
  --reallocation-penalty 0.05 \
  --invalid-penalty 5 \
  > run_07/train.log 2>&1 &