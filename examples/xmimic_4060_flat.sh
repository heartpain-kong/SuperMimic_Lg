#!/usr/bin/env bash
# Activate unitree_rl_mjlab first. Requires 0.7.0 or later.
set -euo pipefail
cd "${SUPERMIMIC_ROOT:-/home/heartpain/SuperMimic}"
robot_root="${MJLAB_LGDOG_ROOT:-/home/heartpain/mjlab/unitree_rl_mjlab_lgdog}"
run_dir="${1:-logs/xmimic_4060_flat_$(date +%Y%m%d_%H%M%S)}"

python -m supermimic_lg.scripts.train_xmimic \
  --mjlab-root "$robot_root" --backend mjlab --device cuda:0 --terrain flat \
  --rollout-mode batched --precision bf16 \
  --num-envs 1024 --iterations 20000 --seed 42 --torch-threads 4 \
  --batch-size 512 --wm-batch-size 512 --policy-batch-size 256 \
  --history 4 --horizon 32 --wm-horizon 8 --replay-capacity 128 \
  --collect-steps 32 --update-to-data-ratio 0.4 \
  --warmup-steps 64 --warmup-iterations 20 --wm-updates 5 --policy-updates 1 \
  --wm-hidden 1024,1024,1024 --actor-hidden 512,256,128 --critic-hidden 256,128 \
  --wm-lr 0.0004 --actor-lr 0.0001 --critic-lr 0.0004 \
  --actor-beta2 0.95 --actor-grad-value 0.1 --no-normalize-actor-loss \
  --gamma 0.99 --gae-lambda 0.95 --critic-epochs 5 --critic-minibatches 4 \
  --policy-ema-decay 0.995 --wm-mirror-fraction 0 \
  --policy-lr-mult 4 --policy-lr-decay-start 1400 --policy-lr-decay-length 300 \
  --lr-decay-iterations 250000 \
  --wm-loss-scale fixed --action-bound-weight 1 --reward-limit 10 --value-target-limit 200 \
  --motion-reward-mode progress --command-mode fixed --command-vx 0.4 --command-vy 0 --command-yaw 0 \
  --standing-fraction 0 --command-curriculum-iterations 0 --command-curriculum-start 0.25 \
  --explore-noise 1 --action-noise 0.1 --explore-iterations 100 \
  --reward-velocity 2 --reward-progress 0.5 --reward-yaw 1 \
  --reward-upright 0.5 --reward-height 0.5 --reward-gait 0.3 --reward-lift 0.3 \
  --reward-action-rate -0.05 --reward-slip -0.05 --reward-energy -0.002 \
  --reward-joint-limit -2 --reward-trunk-contact -2 \
  --target-height 0.3 --swing-height 0.06 --velocity-tolerance 0.5 --yaw-tolerance 0.5 \
  --episode-steps 500 --save-interval 50 --max-training-seconds 1200 \
  --output-dir "$run_dir"

# Evaluation is separate from the 1200-second training budget. It does not
# update the checkpoint and reports needs_training when locomotion is unproven.
python -m supermimic_lg.scripts.evaluate_velocity \
  --checkpoint "$run_dir/checkpoint.pt" --mjlab-root "$robot_root" \
  --backend mjlab --device cuda:0 --terrain flat --policy all \
  --command 0.4 0 0 --seconds 10 --seeds 0 1 42 \
  --output "$run_dir/evaluation.json"
