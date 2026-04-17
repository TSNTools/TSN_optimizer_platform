#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Run 1 on core 0 --> cliffy 1m 42 seed
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

mkdir -p run_env2_100k
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 0 nohup python -u train_dqn_env2.py \
  --config uc_config.cfg \
  --instance simpleeee_scenario.json \
  --out-dir run_env2_100k \
  --timesteps 1000000 \
  --max-steps 100 \
  --step-ns 1000 \
  --invalid-penalty 10 \
  --largest-hole-weight 1 \
  --useless-gap-weight 2 \
  --acceptance-weight 2 \
  --expected-incoming-bytes 1024 \
  --learning-rate 1e-4 \
  --buffer-size 200000 \
  --batch-size 256 \
  --learning-starts 10000 \
  --target-update-interval 5000 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-fraction 0.2 \
  --exploration-final-eps 0.02 \
  --reset-mode never \
  --seed 42 \
  > run_env2_100k/train.log 2>&1 &