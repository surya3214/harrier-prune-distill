#!/usr/bin/env bash
# Run full GPU distillation pipeline (4x H100 via torchrun) for all 16 languages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/distill.yaml}"
NPROC="${NPROC:-4}"

read_training_order() {
  PYTHONPATH="$ROOT/src" python3 -c "from harrier_distill.config import get_training_order; print(' '.join(get_training_order()))"
}

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

LANGS=($(read_training_order))

echo "=== Baseline eval (populate paths in distill.yaml first) ==="
echo "python $ROOT/scripts/04_eval_sts.py --config $CONFIG --model <student_model> --label pruned_baseline --local-sts"

for lang in "${LANGS[@]}"; do
  echo "=== STS ${lang}: embed + train (sequential resume) ==="
  run_embed "$lang"
  run_train "$lang"
done

echo "=== Final eval ==="
echo "python $ROOT/scripts/04_eval_sts.py --config $CONFIG --model <output_dir>/checkpoint_final --label distilled_final --local-sts --suite all16"
