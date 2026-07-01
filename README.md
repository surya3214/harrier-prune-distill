# Harrier 12-Layer MSE Distillation

Distill a **12-layer pruned** Harrier student from the full **18-layer `microsoft/harrier-oss-v1-270m` teacher** using cached teacher embeddings and MSE loss. Training is split into two steps: **EN → KO**.

## Overview

```text
Local (internet)                GPU (offline, 4x H100)
─────────────────               ───────────────────────
01_download_local.py    →       rsync corpora
                                02_generate_teacher_embeddings.py (EN, KO)
                                03_train_distill_mse.py (EN → checkpoint_en)
                                03_train_distill_mse.py (KO → checkpoint_final)
                                04_eval_sts.py (STS-B + KorSTS)
```

- **Prompt:** `sts_query` on all distillation and eval text
- **Loss:** MSE on L2-normalized 640-dim embeddings
- **Data:** ~4.5M total (~2.3M EN + ~2.3M KO), no pilot phase
- **Seq length:** 512 tokens

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

Edit [`configs/distill.yaml`](configs/distill.yaml) and populate all `paths.*` fields:

```yaml
paths:
  local_data_root: "/data/harrier-distill"          # local download
  gpu_data_root: "/mnt/data/harrier-distill"        # after rsync
  teacher_model: "/models/harrier-oss-v1-270m"
  student_model: "/models/harrier-12l-pruned"
  output_dir: "/mnt/data/harrier-distill/output"
  en_corpus: "/mnt/data/harrier-distill/en/corpus.parquet"
  ko_corpus: "/mnt/data/harrier-distill/ko/corpus.parquet"
  en_embeddings: "/mnt/data/harrier-distill/output/embeddings/en_embeddings.parquet"
  ko_embeddings: "/mnt/data/harrier-distill/output/embeddings/ko_embeddings.parquet"
```

Dataset sources are defined in [`configs/datasets.yaml`](configs/datasets.yaml).

## Step 1 — Download on local (internet)

```bash
python scripts/01_download_local.py --config configs/distill.yaml
```

Outputs:

- `{local_data_root}/en/corpus.parquet` (~2.3M rows)
- `{local_data_root}/ko/corpus.parquet` (~2.3M rows)

Sources:

| Lang | Dataset | Target |
|------|---------|--------|
| EN | `allenai/c4` (en) | 2.0M |
| EN | `sentence-transformers/all-nli` (`pair`) | 300k unique |
| KO | `allenai/c4` (ko) | 2.0M |
| KO | `klue/klue` (nli) | 50k unique |
| KO | `HuggingFaceFW/fineweb-2` (`kor_Hang`, optional) | 200k |

## Step 2 — Migrate to GPU

```bash
rsync -avP /data/harrier-distill/ user@gpu-host:/mnt/data/harrier-distill/
```

Ensure teacher and pruned student checkpoints are also available on the GPU node.

## Step 3 — Baseline STS eval (recommended)

```bash
python scripts/04_eval_sts.py --config configs/distill.yaml \
  --model /models/harrier-12l-pruned --label pruned_baseline

python scripts/04_eval_sts.py --config configs/distill.yaml \
  --model /models/harrier-oss-v1-270m --label teacher
```

## Step 4 — EN distillation (embed + train)

```bash
# Generate teacher embeddings (4 GPUs)
torchrun --standalone --nproc_per_node=4 \
  scripts/02_generate_teacher_embeddings.py --config configs/distill.yaml --lang en

# Train student with MSE loss (4 GPUs)
torchrun --standalone --nproc_per_node=4 \
  scripts/03_train_distill_mse.py --config configs/distill.yaml --lang en
```

Checkpoint: `{output_dir}/checkpoint_en/`

## Step 5 — KO distillation (embed + train)

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/02_generate_teacher_embeddings.py --config configs/distill.yaml --lang ko

# Auto-resumes checkpoint_en when present
torchrun --standalone --nproc_per_node=4 \
  scripts/03_train_distill_mse.py --config configs/distill.yaml --lang ko
```

Checkpoint: `{output_dir}/checkpoint_final/`

Or run the full GPU sequence:

```bash
bash scripts/run_gpu_pipeline.sh
```

## Step 6 — Final eval

```bash
python scripts/04_eval_sts.py --config configs/distill.yaml \
  --model /mnt/data/harrier-distill/output/checkpoint_final --label distilled_final
```

Results are saved under `{output_dir}/eval/`.

## Time estimates (4x H100, ~4.5M)

| Stage | Wall clock |
|-------|------------|
| Local download | hours – 1 day |
| Teacher embed EN + KO | ~2–5 h |
| Train EN + KO (1 epoch each) | ~1–1.5 h |
| Eval | ~15–30 min |

## Extending to retrieval (phase 2)

v1 optimizes STS with `sts_query`. To recover retrieval MTEB later:

1. Add MS-MARCO (or BEIR) to a new `retrieval` phase in config
2. Cache teacher embeddings with `web_search_query` on queries and **no prompt** on passages
3. Resume from `checkpoint_final` and run a second MSE training pass

See `phases.retrieval` stub in [`configs/distill.yaml`](configs/distill.yaml).

## Fallback ladder

| Symptom | Action |
|---------|--------|
| STS gap >10% after KO step | Add `1 - cosine_sim` loss term |
| Still lagging | Small contrastive pass on NLI pairs |
| Pruned baseline very low | Check layer removal; consider hidden-state distill |
| KO noisy | Increase KLUE-NLI / KoWiki mix in `datasets.yaml` |

## Project layout

```text
configs/
  distill.yaml          # paths + training hyperparams (you populate)
  datasets.yaml         # HF dataset definitions
scripts/
  01_download_local.py
  02_generate_teacher_embeddings.py
  03_train_distill_mse.py
  04_eval_sts.py
  run_gpu_pipeline.sh
src/harrier_distill/
  config.py data.py model.py eval.py distributed.py text.py
```
