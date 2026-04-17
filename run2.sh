#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Run 3 on core 2 --> candidates 1m 42 seed
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

mkdir -p run_cand_s42_10m
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 2 nohup python -u train_dqn_candidates.py \
  --config uc_config.cfg \
  --instance simple_scenario.json \
  --out-dir run_cand_s42_10m \
  --timesteps 10000000 \
  --max-steps 48 \
  --max-candidates 10 \
  --invalid-penalty 200 \
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
  --seed 42 \
  > run_cand_s42_10m/train.log 2>&1 &

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Run 4 on core 3 --> candidates 1m 43 seed
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


mkdir -p run_cand_s43_10m
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 3 nohup python -u train_dqn_candidates.py \
  --config uc_config.cfg \
  --instance simple_scenario.json \
  --out-dir run_cand_s43_10m \
  --timesteps 10000000 \
  --max-steps 48 \
  --max-candidates 10 \
  --invalid-penalty 200 \
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
  --seed 43 \
  > run_cand_s43_10m/train.log 2>&1 &