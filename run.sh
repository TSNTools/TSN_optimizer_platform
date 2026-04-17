# ======================================================
# Run 1: NEVER RESET
# ======================================================
mkdir -p run_50k_lh_only

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 2 nohup python -u train_dqn_env2.py \
  --config uc_config.cfg \
  --instance simple_scenario.json \
  --out-dir run_50k_lh_only \
  --timesteps 50000 \
  --max-steps 24 \
  --invalid-penalty 1 \
  --largest-hole-weight 1 \
  --acceptance-weight 1 \
  --useless-gap-weight 0 \
  --reset-mode never \
  --seed 42 \
  > run_50k_lh_only/train.log 2>&1 &