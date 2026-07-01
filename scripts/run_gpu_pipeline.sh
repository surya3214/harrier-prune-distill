#!/usr/bin/env bash
# Run full GPU distillation pipeline (4x H100 via torchrun).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/distill.yaml}"
NPROC="${NPROC:-4}"

run_embed() {
  local lang="$1"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/02_generate_teacher_embeddings.py" \
    --config "$CONFIG" \
    --lang "$lang"
}

run_train() {
  local lang="$1"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/03_train_distill_mse.py" \
    --config "$CONFIG" \
    --lang "$lang"
}

echo "=== Baseline eval (populate paths in distill.yaml first) ==="
echo "python $ROOT/scripts/04_eval_sts.py --config $CONFIG --model <student_model> --label pruned_baseline"
echo "python $ROOT/scripts/04_eval_sts.py --config $CONFIG --model <teacher_model> --label teacher"

echo "=== EN: embed + train ==="
run_embed en
run_train en

echo "=== KO: embed + train ==="
run_embed ko
# KO step auto-resumes checkpoint_en when present
torchrun --standalone --nproc_per_node="$NPROC" \
  "$ROOT/scripts/03_train_distill_mse.py" \
  --config "$CONFIG" \
  --lang ko

echo "=== Final eval ==="
echo "python $ROOT/scripts/04_eval_sts.py --config $CONFIG --model <output_dir>/checkpoint_final --label distilled_final"
