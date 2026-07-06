#!/usr/bin/env bash
# Run retrieval distillation pipeline (4x GPU via torchrun).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/distill.yaml}"
NPROC="${NPROC:-4}"
PHASE="retrieval"

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

echo "=== Retrieval baseline eval (populate paths in distill.yaml first) ==="
echo "python $ROOT/scripts/04_eval_retrieval.py --config $CONFIG --model <checkpoint_final> --label sts_distilled --suite en_ko"
echo "python $ROOT/scripts/05_compare_retrieval.py --config $CONFIG --student <output_dir>/retrieval/checkpoint_final --suite en_ko"

echo "=== EN retrieval: embed + train (init from STS checkpoint_final) ==="
run_embed en
run_train en

echo "=== KO retrieval: embed + train (resume retrieval/checkpoint_en) ==="
run_embed ko
run_train ko

echo "=== Final retrieval eval ==="
echo "python $ROOT/scripts/04_eval_retrieval.py --config $CONFIG --model <output_dir>/retrieval/checkpoint_final --label retrieval_final --suite en_ko"
echo "python $ROOT/scripts/05_compare_retrieval.py --config $CONFIG --student <output_dir>/retrieval/checkpoint_final --suite en_ko"
echo "python $ROOT/scripts/05_compare_retrieval.py --config $CONFIG --student <output_dir>/retrieval/checkpoint_final --suite en_ko --local-retrieval"
