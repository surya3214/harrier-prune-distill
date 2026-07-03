# Harrier 12-Layer MSE Distillation

Distill a **12-layer pruned** Harrier student from the full **18-layer `microsoft/harrier-oss-v1-270m` teacher** using cached teacher embeddings and MSE loss. Training is split into two steps: **EN → KO**.

## Overview

```text
Local (internet)                GPU (offline, 4x H100)
─────────────────               ───────────────────────
01_download_local.py    →       rsync corpora
01_download_sts_local.py →      rsync STS parquet
                                02_generate_teacher_embeddings.py (EN, KO)
                                03_train_distill_mse.py (EN → checkpoint_en)
                                03_train_distill_mse.py (KO → checkpoint_final)
                                04_eval_sts.py (STS-B + KorSTS, --local-sts offline)
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
  sts_data_root: "/mnt/data/harrier-distill/sts"
  en_sts_test: "/mnt/data/harrier-distill/sts/en/stsbenchmark_test.parquet"
  ko_sts_test: "/mnt/data/harrier-distill/sts/ko/korsts_test.parquet"
  en_retrieval_corpus: "/mnt/data/harrier-distill/retrieval/en/corpus.parquet"
  ko_retrieval_corpus: "/mnt/data/harrier-distill/retrieval/ko/corpus.parquet"
  en_retrieval_embeddings: "/mnt/data/harrier-distill/output/retrieval/embeddings/en_embeddings.parquet"
  ko_retrieval_embeddings: "/mnt/data/harrier-distill/output/retrieval/embeddings/ko_embeddings.parquet"
  retrieval_checkpoint_en: "/mnt/data/harrier-distill/output/retrieval/checkpoint_en"
  retrieval_checkpoint_final: "/mnt/data/harrier-distill/output/retrieval/checkpoint_final"
```

Dataset sources: [`configs/datasets.yaml`](configs/datasets.yaml) (STS), [`configs/sts_datasets.yaml`](configs/sts_datasets.yaml) (STS eval), [`configs/retrieval_datasets.yaml`](configs/retrieval_datasets.yaml) (retrieval).

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

## Step 1b — Download STS benchmarks (local, internet)

```bash
python scripts/01_download_sts_local.py --config configs/distill.yaml
```

Outputs under `{local_data_root}/sts/`:

- `en/stsbenchmark_test.parquet` (1,379 pairs)
- `en/stsbenchmark_validation.parquet` (1,500 pairs, for debug proxy)
- `ko/korsts_test.parquet` (1,376 pairs)
- `ko/korsts_valid.parquet` (1,465 pairs)
- `manifest.json`

## Step 2 — Migrate to GPU

```bash
rsync -avP /data/harrier-distill/ user@gpu-host:/mnt/data/harrier-distill/
```

Ensure teacher and pruned student checkpoints are also available on the GPU node.

## Step 3 — Baseline STS eval (recommended)

Online (MTEB downloads from HuggingFace):

```bash
python scripts/04_eval_sts.py --config configs/distill.yaml \
  --model /models/harrier-12l-pruned --label pruned_baseline

python scripts/04_eval_sts.py --config configs/distill.yaml \
  --model /models/harrier-oss-v1-270m --label teacher
```

Offline on GPU (uses local STS parquet, no internet):

```bash
python scripts/04_eval_sts.py --config configs/distill.yaml \
  --model /models/harrier-oss-v1-270m --label teacher --local-sts
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
  --model /mnt/data/harrier-distill/output/checkpoint_final --label distilled_final --local-sts
```

Results are saved under `{output_dir}/eval/`.

## Step 7 — Compare teacher vs student (recommended)

Side-by-side STS evaluation of the teacher and your distilled checkpoint:

```bash
python scripts/05_compare_sts.py --config configs/distill.yaml \
  --student /mnt/data/harrier-distill/output/checkpoint_final \
  --suite multilingual

# EN-only STS
python scripts/05_compare_sts.py --config configs/distill.yaml \
  --student /mnt/data/harrier-distill/output/checkpoint_final \
  --suite en

# Offline on GPU
python scripts/05_compare_sts.py --config configs/distill.yaml \
  --student /mnt/data/harrier-distill/output/checkpoint_final \
  --suite multilingual --local-sts
```

Suites: `en` (STSBenchmark), `ko` (KorSTS), `multilingual` (both), `extended` (adds STS22.v2 + STSBenchmarkMultilingualSTS; online only).

## Step 8 — Debug MSE vs STS gap

If training `avg_loss` is low but STS lags the teacher, run alignment diagnostics:

```bash
python scripts/06_debug_mse_alignment.py --config configs/distill.yaml \
  --student /mnt/data/harrier-distill/output/checkpoint_final \
  --lang en

python scripts/06_debug_mse_alignment.py --config configs/distill.yaml \
  --student /mnt/data/harrier-distill/output/checkpoint_final \
  --lang ko
```

The report checks:

1. **Cache alignment** — teacher re-encode matches cached parquet embeddings (catches prompt/dtype bugs)
2. **Pointwise alignment** — MSE, cosine, angular error, per-dimension correlation on sampled training texts
3. **Pairwise STS proxy** — Spearman on STSBenchmark validation without a full MTEB run
4. **Checklist** — pass/fail hints for common failure modes

Interpretation:

| Symptom | Likely cause |
|---------|--------------|
| High cache MSE | Re-generate embeddings or fix prompt/max_length mismatch |
| Low pointwise MSE but low pairwise STS | Geometry distortion — add cosine loss or contrastive phase |
| Poor STS vs pruned baseline | Distillation may not be helping; check init checkpoint |
| Low dim_corr_min | Dimension collapse in student |

## Time estimates (4x H100, ~4.5M)

| Stage | Wall clock |
|-------|------------|
| Local download | hours – 1 day |
| Teacher embed EN + KO | ~2–5 h |
| Train EN + KO (1 epoch each) | ~1–1.5 h |
| Eval | ~15–30 min |

## Retrieval distillation (phase 2)

After STS distillation (`checkpoint_final`), recover retrieval with hard-negative corpora:

```text
Local (internet)                         GPU (offline)
────────────────                         ─────────────
01_download_retrieval_local.py  →        rsync retrieval/ corpora
                                         02_generate_teacher_embeddings.py --phase retrieval
                                         03_train_distill_mse.py --phase retrieval
                                         04_eval_retrieval.py / 05_compare_retrieval.py
```

### Datasets (EN + KO first)

| Lang | Source | HF dataset |
|------|--------|------------|
| EN | MIRACL EN hard negatives (supplement) + MS MARCO triplets (bulk) | `datalama/miracl-hard-negatives` (`en`) + `sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1` |
| KO | MIRACL hard negatives (`kor`) | `datalama/miracl-hard-negatives` |

Config: [`configs/retrieval_datasets.yaml`](configs/retrieval_datasets.yaml). Add languages by extending `languages:` and a new lang block (same MIRACL dataset, different `qrels_config`).

**Pilot run** (mini MIRACL EN + mini MIRACL KO + 50k MS MARCO):

```bash
python scripts/01_download_retrieval_local.py --config configs/distill.yaml --pilot
```

**Full download:**

```bash
python scripts/01_download_retrieval_local.py --config configs/distill.yaml
```

Outputs `{local_data_root}/retrieval/{en,ko}/corpus.parquet` with `role` column (`query` | `doc`).

### GPU training

```bash
bash scripts/run_gpu_retrieval_pipeline.sh
```

- **Prompts:** `web_search_query` on queries, no prompt on documents
- **Init:** EN pass resumes from STS `checkpoint_final`; KO pass resumes `retrieval/checkpoint_en`
- **Checkpoints:** `{output_dir}/retrieval/checkpoint_en` → `checkpoint_final`

### Retrieval eval

```bash
python scripts/04_eval_retrieval.py --config configs/distill.yaml \
  --model /path/to/retrieval/checkpoint_final --suite en_ko

python scripts/05_compare_retrieval.py --config configs/distill.yaml \
  --student /path/to/retrieval/checkpoint_final --suite en_ko
```

Tasks: MTEB `MSMARCO` (EN) + `MIRACLRetrieval` (`en`, `ko` subsets).

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
  distill.yaml              # paths + training hyperparams (you populate)
  datasets.yaml             # STS phase HF dataset definitions
  retrieval_datasets.yaml   # retrieval phase HF dataset definitions (EN/KO)
scripts/
  01_download_local.py
  01_download_retrieval_local.py
  01_download_sts_local.py
  02_generate_teacher_embeddings.py
  03_train_distill_mse.py
  04_eval_sts.py
  04_eval_retrieval.py
  05_compare_sts.py
  05_compare_retrieval.py
  06_debug_mse_alignment.py
  run_gpu_pipeline.sh
  run_gpu_retrieval_pipeline.sh
src/harrier_distill/
  config.py data.py model.py eval.py retrieval.py sts.py debug.py distributed.py text.py
```
