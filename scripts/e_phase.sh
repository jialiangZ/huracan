#!/usr/bin/env bash
# E-phase orchestration: run AFTER runs/bsef/r0b assim training completes.
# GPU2 only.  ~2 h total.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=configs/bsef_r0b.yaml
WD=runs/bsef/r0b
OUT=runs/bsef
LOG=$OUT/e_phase.log
exec >> "$LOG" 2>&1
echo "===== E phase start $(date) ====="

# 1) background cache v2: 3 obs variants x 4 seasonal starts, K=8,
#    with same-background fcst/zero-obs control branches (21 d collect,
#    7 d flagged spin-up)
for V in full thin50 hold20; do
  case $V in
    full)   DD="" ;;
    thin50) DD="--data-dir data/era5_v1b_thin50" ;;
    hold20) DD="--data-dir data/era5_v1b_hold20" ;;
  esac
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/cache_background.py \
    --config $CFG --workdir $WD --out $OUT/bg_cache_v2 \
    --starts 2012-01-15 2012-04-15 2012-07-15 2012-10-15 \
    --days 21 --pre-days 7 --members 8 --ic-noise 0.5 --ckpt best \
    --variant-name $V $DD
done

# 2) warm-up sensitivity (full variant only): pre 30 d and 60 d on the
#    three starts where the longer pre-window fits inside val 2012
for PRE in 30 60; do
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/cache_background.py \
    --config $CFG --workdir $WD --out $OUT/bg_cache_v2 \
    --starts 2012-04-15 2012-07-15 2012-10-15 \
    --days 21 --pre-days $PRE --members 8 --ic-noise 0.5 --ckpt best \
    --variant-name full_pre${PRE}d
done

# 3) diagnostics v2 (CPU-heavy; no GPU needed)
.venv/bin/python scripts/diagnose_background.py \
  --config $CFG --cache $OUT/bg_cache_v2 --out $OUT/bg_diag_v2

# 4) r0b three-control evaluation with the corrected framing
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/eval_r0.py \
  --config $CFG --workdir $WD --out $OUT/r0b_eval_v2 --ckpt best

echo "===== E phase done $(date) ====="
