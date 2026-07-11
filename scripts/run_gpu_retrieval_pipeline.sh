#!/usr/bin/env bash
# Run retrieval distillation pipeline (4x GPU via torchrun) for all 16 languages.
# Resumes from the next incomplete language after the last saved checkpoint.
# Use --force or FORCE=1 to redo embed+train for every language.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/distill.yaml}"
NPROC="${NPROC:-4}"
PHASE="retrieval"
FORCE="${FORCE:-0}"

usage() {
  echo "Usage: $0 [--force] [--config PATH]"
  echo "  --force     Ignore completion markers; redo all langs"
  echo "  --config    Distill config (default: CONFIG env or configs/distill.yaml)"
  echo "  FORCE=1     Same as --force"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

read_training_order() {
  PYTHONPATH="$ROOT/src" python3 -c "from harrier_distill.config import get_training_order; print(' '.join(get_training_order()))"
}

lang_status() {
  local lang="$1"
  PYTHONPATH="$ROOT/src" python3 "$ROOT/scripts/retrieval_resume_status.py" \
    --config "$CONFIG" --mode lang --lang "$lang"
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

if [[ "$FORCE" == "1" ]]; then
  echo "=== Retrieval pipeline: FORCE=1 (redo all languages) ==="
  START_LANG="${LANGS[0]}"
else
  PYTHONPATH="$ROOT/src" python3 "$ROOT/scripts/retrieval_resume_status.py" --config "$CONFIG" --mode summary
  NEXT_LINE="$(PYTHONPATH="$ROOT/src" python3 "$ROOT/scripts/retrieval_resume_status.py" --config "$CONFIG" --mode next)"
  if [[ "$NEXT_LINE" == "ALL_DONE" ]]; then
    echo "=== All retrieval languages already complete; exiting. Re-run with --force to redo. ==="
    exit 0
  fi
  START_LANG="${NEXT_LINE#START }"
  echo "=== Resuming retrieval pipeline from language: ${START_LANG} ==="
fi

STARTED=0
for lang in "${LANGS[@]}"; do
  if [[ "$STARTED" -eq 0 && "$lang" != "$START_LANG" ]]; then
    echo "=== Skip ${lang} (already complete) ==="
    continue
  fi
  STARTED=1

  if [[ "$FORCE" == "1" ]]; then
    NEED_EMBED=1
    NEED_TRAIN=1
  else
    mapfile -t STATUS < <(lang_status "$lang")
    NEED_EMBED=1
    NEED_TRAIN=1
    [[ "${STATUS[0]:-}" == "EMBED_DONE" ]] && NEED_EMBED=0
    [[ "${STATUS[1]:-}" == "TRAIN_DONE" ]] && NEED_TRAIN=0
  fi

  echo "=== Retrieval ${lang}: embed + train (sequential resume) ==="
  if [[ "$NEED_EMBED" -eq 1 ]]; then
    run_embed "$lang"
  else
    echo "  [skip] embeddings already present for ${lang}"
  fi
  if [[ "$NEED_TRAIN" -eq 1 ]]; then
    run_train "$lang"
  else
    echo "  [skip] train checkpoint already complete for ${lang}"
  fi
done

echo "=== Final retrieval eval ==="
echo "python $ROOT/scripts/04_eval_retrieval.py --config $CONFIG --model <output_dir>/retrieval/checkpoint_final --label retrieval_final --suite all16"
echo "python $ROOT/scripts/05_compare_retrieval.py --config $CONFIG --student <output_dir>/retrieval/checkpoint_final --suite all16 --local-retrieval"
