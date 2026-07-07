#!/usr/bin/env bash
# Run retrieval distillation pipeline (4x GPU via torchrun) for all 16 languages.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/distill.yaml}"
NPROC="${NPROC:-4}"
PHASE="retrieval"

read_training_order() {
  PYTHONPATH="$ROOT/src" python3 -c "from harrier_distill.config import get_training_order; print(' '.join(get_training_order()))"
}

run_embed() {
  local lang="$1"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/02_generate_teacher_embeddings.py" \
    --config "$CONFIG" \
    --phase "$PHASE" \
    --lang "$lang"
}

run_train() {
  local lang="$1"
  torchrun --standalone --nproc_per_node="$NPROC" \
    "$ROOT/scripts/03_train_distill_mse.py" \
    --config "$CONFIG" \
    --phase "$PHASE" \
    --lang "$lang"
}

LANGS=($(read_training_order))

echo "=== Retrieval baseline eval (populate paths in distill.yaml first) ==="
echo "python $ROOT/scripts/05_compare_retrieval.py --config $CONFIG --student <checkpoint_final> --suite all16"

for lang in "${LANGS[@]}"; do
  echo "=== Retrieval ${lang}: embed + train (sequential resume) ==="
  run_embed "$lang"
  run_train "$lang"
done

echo "=== Final retrieval eval ==="
echo "python $ROOT/scripts/04_eval_retrieval.py --config $CONFIG --model <output_dir>/retrieval/checkpoint_final --label retrieval_final --suite all16"
echo "python $ROOT/scripts/05_compare_retrieval.py --config $CONFIG --student <output_dir>/retrieval/checkpoint_final --suite all16 --local-retrieval"
