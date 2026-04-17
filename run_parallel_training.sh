#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Run 1 on core 0
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

mkdir -p run_50m_100_no_reset

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 0 nohup python -u train_dqn_env2.py \
  --config uc_config.cfg \
  --instance simple_scenario.json \
  --out-dir run_50m_100_no_reset \
  --timesteps 50000000 \
  --max-steps 100 \
  --invalid-penalty 200 \
  --largest-hole-weight 2 \
  --useless-gap-weight 2 \
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
  > run_50m_100_no_reset/train.log 2>&1 &


#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Run 2 on core 1
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

mkdir -p run_50m_100_freeze

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 1 nohup python -u train_dqn_env2.py \
  --config uc_config.cfg \
  --instance simple_scenario.json \
  --out-dir run_50m_100_freeze \
  --timesteps 50000000 \
  --max-steps 100 \
  --invalid-penalty 200 \
  --largest-hole-weight 2 \
  --useless-gap-weight 2 \
  --learning-rate 1e-4 \
  --buffer-size 200000 \
  --batch-size 256 \
  --learning-starts 10000 \
  --target-update-interval 5000 \
  --train-freq 4 \
  --gradient-steps 1 \
  --exploration-fraction 0.2 \
  --exploration-final-eps 0.02 \
  --reset-mode freeze_after \
  --freeze-reset-after-episode 100000 \
  --seed 42 \
  > run_50m_100_freeze/train.log 2>&1 &