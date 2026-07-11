# Harrier 12-Layer MSE Distillation

Distill a **12-layer pruned** Harrier student from the full **18-layer `microsoft/harrier-oss-v1-270m` teacher** using cached teacher embeddings. Supports **16 languages** with sequential checkpoint resume (see [`configs/languages.yaml`](configs/languages.yaml)).

## Overview

```text
Local (internet)                GPU (offline, 4x H100)
─────────────────               ───────────────────────
01_download_local.py    →       rsync corpora
01_download_sts_local.py →      rsync STS parquet
                                02_generate_teacher_embeddings.py (per lang)
                                03_train_distill_mse.py (sequential 16-lang chain)
                                04_eval_sts.py (--suite all16, --local-sts offline)
```

- **Languages:** `en`, `ko`, `ar`, `zh`, `fr`, `de`, `hi`, `id`, `it`, `ja`, `pt`, `ru`, `es`, `vi`, `th`, `pl` (training order in `languages.yaml`)
- **Prompt:** `sts_query` (STS) / `web_search_query` (retrieval queries)
- **Loss:** MSE + cosine (STS); MSE + cosine + score_kl (retrieval)
- **Production data:** 1M texts/lang × 16 ≈ 16M STS rows; ~350k retrieval triplets/lang × 7 hard negatives (ko/th much smaller)
- **Seq length:** 512 tokens

## Training losses

Configure weights in [`configs/distill.yaml`](configs/distill.yaml):

```yaml
training:
  losses:
    mse: 0.8
    cosine: 0.2
    pairwise_mse: 0.0
    score_kl: 0.0
  pairwise_triplets_per_batch: 48

phases:
  retrieval:
    score_kl_temperature: 0.05
    losses:
      mse: 0.2
      cosine: 0.4
      pairwise_mse: 0.0
      score_kl: 0.4
```

| Loss | Description | STS phase | Retrieval phase |
|------|-------------|-----------|-----------------|
| `mse` | Pointwise MSE vs cached teacher embeddings | Yes | Yes (lower weight) |
| `cosine` | `1 - cosine_similarity(student, teacher)` on normalized vectors | Yes | Yes |
| `pairwise_mse` | MSE on query↔doc dot products per triplet vs teacher | No (weight 0) | Optional / off by default |
| `score_kl` | Softmax-KL on query↔doc score distributions per triplet | No (weight 0) | Yes |

**Contrastive loss is not implemented** — optional Phase-4 fallback only (see Fallback ladder).

**Triplet loss requirement:** Retrieval embedding parquet must include `triplet_id` (from `02_generate_teacher_embeddings.py --phase retrieval`). MIRACL train corpora use `negatives_per_query: 8` hard negatives per query.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=src
```

Edit [`configs/distill.yaml`](configs/distill.yaml) and populate `paths.*` (per-language paths resolve automatically):

```yaml
paths:
  local_data_root: "/data/harrier-distill"
  gpu_data_root: "/mnt/data/harrier-distill"
  teacher_model: "/models/harrier-oss-v1-270m"
  student_model: "/models/harrier-12l-pruned"
  output_dir: "/mnt/data/harrier-distill/output"
  sts_data_root: "/mnt/data/harrier-distill/sts"
  retrieval_eval_data_root: "/mnt/data/harrier-distill/retrieval_eval"

data:
  full_samples_per_lang: 1000000
  pilot_samples_per_lang: 0
```

Configs: [`languages.yaml`](configs/languages.yaml), [`datasets.yaml`](configs/datasets.yaml), [`sts_datasets.yaml`](configs/sts_datasets.yaml), [`retrieval_datasets.yaml`](configs/retrieval_datasets.yaml).

## Step 1 — Download on local (internet)

```bash
python scripts/01_download_local.py --config configs/distill.yaml --lang all
python scripts/01_download_local.py --lang ar,de --skip-existing   # incremental
python scripts/01_download_local.py --lang en --force            # rebuild
```

Outputs `{local_data_root}/{lang}/corpus.parquet` + `manifest.json`. Default **1M texts/lang** (85% C4 + 15% multilingual-NLI).

## Step 1b — Download STS benchmarks (local, internet)

```bash
python scripts/01_download_sts_local.py --config configs/distill.yaml
```

Outputs under `{local_data_root}/sts/`:

**EN — MTEB(eng, v2) STS (9 tasks):**

| Parquet | Pairs |
|---------|------:|
| `en/biosses_test.parquet` | 100 |
| `en/sickr_test.parquet` | 9,927 |
| `en/sts12_test.parquet` | 3,108 |
| `en/sts13_test.parquet` | 1,500 |
| `en/sts14_test.parquet` | 3,750 |
| `en/sts15_test.parquet` | 3,000 |
| `en/stsbenchmark_test.parquet` | 1,379 |
| `en/sts17_en_en_test.parquet` | 250 |
| `en/sts22_v2_en_test.parquet` | 197 |
| `en/stsbenchmark_validation.parquet` | 1,500 (debug proxy) |

**KO:**

- `ko/korsts_test.parquet` (1,376 pairs)
- `ko/korsts_valid.parquet` (1,465 pairs)

Also writes `manifest.json`.

Download EN only: `--lang en`. All langs: `--lang all`. Specific tasks: `--tasks STSBenchmark JSICK`.

STS22 subsets for ar/de/en/es/fr/it/pl/ru/zh, plus JSICK (ja), ASSIN2 (pt), SemRel24 (ar/hi/id) are in [`configs/sts_datasets.yaml`](configs/sts_datasets.yaml). Run `python scripts/validate_dataset_splits.py` to verify Hub paths before downloading.

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

## Step 4–5 — GPU distillation (16 languages)

Each language resumes the previous checkpoint (order in `languages.yaml`). Run the full chain:

```bash
bash scripts/run_gpu_pipeline.sh
```

Or one language:

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/02_generate_teacher_embeddings.py --config configs/distill.yaml --lang ar
torchrun --standalone --nproc_per_node=4 \
  scripts/03_train_distill_mse.py --config configs/distill.yaml --lang ar
```

Checkpoints: `{output_dir}/checkpoints/sts/{lang}/` (legacy: `checkpoint_en`, `checkpoint_final` for en/pl).

Epoch count: `default_num_epochs` in `distill.yaml`, per-lang overrides in `languages.yaml`, or legacy `num_epochs_en` keys.

### Training performance knobs

Configured under `training:` in [`configs/distill.yaml`](configs/distill.yaml):

| Knob | Default | Notes |
|------|---------|-------|
| `gradient_checkpointing` | `true` | Trades compute for activation memory; often allows a larger `train_batch_size_per_gpu` |
| `attn_implementation` | `sdpa` | `sdpa`, `flash_attention_2` (needs `flash-attn`), or `none` (eager) |
| `fused_adamw` | `true` | CUDA fused AdamW when available; falls back automatically |
| `enable_tf32` | `true` | TF32 matmul on Ampere+ (A100/H100); no-op on V100 |

After enabling checkpointing, probe VRAM (`nvidia-smi`) and raise `train_batch_size_per_gpu` if you were memory-bound. FlashAttention-2 and `torch.compile` are not defaults.

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

# Parallel 3-way compare on separate GPUs (teacher / student / baseline)
python scripts/05_compare_sts.py --config configs/distill.yaml \
  --teacher /models/harrier-oss-v1-270m \
  --student /mnt/data/harrier-distill/output/checkpoint_final \
  --baseline /models/harrier-12l-pruned \
  --suite all16 --local-sts \
  --parallel --gpus 0,1,2
```

Eval prints stage progress (`[eval][teacher] Task 3/12: ...`) during model load and per-task scoring. Use `--quiet` to keep stage lines but disable tqdm bars. Under `--parallel`, bars are off on shared stdout by default (stage lines stay prefixed with `[label][gpu=N]`). Pass `--log-dir DIR` to write per-model detail (including tqdm) to `DIR/{teacher,student,baseline}.log`. 3-way parallel needs ~3× VRAM. Parallel compare launches one subprocess per model with `CUDA_VISIBLE_DEVICES` set **before** Python/torch starts (required for multi-GPU; spawn workers re-import the script and otherwise init all GPUs).

Suites: `en`, `ko`, `wave1`, `wave2`, `wave3`, `all16`, `multilingual`, `extended`.

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

## Time estimates (4x H100, production 16-lang)

| Stage | Rough scale |
|-------|-------------|
| Local download (16 langs) | hours – 1 day |
| STS embed + train × 16 | ~17–35 GPU-hours |
| Retrieval embed + train × 16 | ~15–30 GPU-hours |
| Eval (STS22 + MIRACL subsets) | ~4–8 hours |
| Embedding storage | ~90–100 GB total |

**Pilot** (`pilot_samples_per_lang: 100000`, retrieval `--pilot`): ~1.6M texts, ~5 GB embeddings — recommended first validation.

## Retrieval distillation (phase 2)

After STS distillation (`checkpoint_final`), recover retrieval with hard-negative corpora:

```text
Local (internet)                         GPU (offline)
────────────────                         ─────────────
01_download_retrieval_local.py  →        rsync retrieval/ corpora
01_download_retrieval_eval_local.py →    rsync retrieval_eval/ benchmarks
                                         02_generate_teacher_embeddings.py --phase retrieval
                                         03_train_distill_mse.py --phase retrieval
                                         04_eval_retrieval.py / 05_compare_retrieval.py
```

### Datasets (16 languages)

Target **~350k multi-neg triplets/lang** (7 hard negatives → 8-way `score_kl`) where public sources allow. Prefer listwise hard negatives over 1M×1-neg volume for teacher→student retrieval retention.

| Group | Langs | Retrieval train source |
|-------|-------|------------------------|
| MIRACL + hard-neg-7 | ar, de, en, es, fr, id, ja, zh | MIRACL HN + `hotchpotch/...` `{lang}-hard-negatives-7` |
| MIRACL + unicamp packed | hi, ru | MIRACL HN + unicamp train IDs grouped to 7 negs/query |
| hard-neg-7 | it | `hotchpotch/...` `italian-hard-negatives-7` |
| unicamp packed | pt | unicamp train IDs grouped to 7 negs/query |
| mMARCO packed | vi | `chieunq/mMARCO_vietnamese` grouped to 7 negs/query |
| MAUPQA | pl | `ipipan/maupqa` all train CSV subsets (`negatives_per_triplet: 7`) |
| MIRACL + Mr. TyDi | ko, th | MIRACL + `crystina-z/mrtydi-mContriever-mmarco-HN` only (≪350k; no public mMARCO) |

Config: [`configs/retrieval_datasets.yaml`](configs/retrieval_datasets.yaml).

**Pilot run:**

```bash
python scripts/01_download_retrieval_local.py --config configs/distill.yaml --pilot --lang all
```

**Full download:**

```bash
python scripts/01_download_retrieval_local.py --config configs/distill.yaml --lang all
```

Outputs `{local_data_root}/retrieval/{lang}/corpus.parquet` with `role` and `triplet_id` columns. `--skip-existing` / `--force` supported.

Streaming sources (`*-hard-negatives-7`, Vietnamese mMARCO) stop once the lang target is reached. Unicamp train-ID langs (`hi`/`ru`/`pt`) still download the shared ID triples file + translated collection TSVs (largest local inbound cost). Built corpora under `retrieval/` are what you rsync to the GPU (~1 pos + 7 negs → ~8 docs/triplet × 350k ≈ fewer GB than the old 1M×1-neg layout for hotchpotch langs).

**Train knobs for 80GB:** `training.pairwise_triplets_per_batch: 48` with 7-neg triplets keeps peak encode size around ~45–60GB (pointwise 256 + 48×8 docs).

**Regen after raising MIRACL `negatives_per_query` (or switching to `score_kl` / multi-neg corpora):** old corpora are stale — rebuild, rsync, re-embed, retrain:

```bash
# Local (internet)
python scripts/01_download_retrieval_local.py --config configs/distill.yaml --lang all --force
rsync -avP "$LOCAL_DATA_ROOT/retrieval/" user@gpu-host:"$GPU_DATA_ROOT/retrieval/"

# GPU (offline): git pull for score_kl + configs, then
bash scripts/run_gpu_retrieval_pipeline.sh
# then 04/05 --local-retrieval
```

Manifests store actual `triplet_count`, configured `target_triplets`, and `negatives_per_query`. Skip only when `triplet_count >= target`; under-target langs (e.g. `ko`/`th`) print a warning. `--force` still recommended when intentionally rebuilding.

### Download retrieval eval benchmarks (local, internet)

For offline GPU eval (no MTEB/HF at eval time):

```bash
python scripts/01_download_retrieval_eval_local.py --config configs/distill.yaml
```

Outputs under `{local_data_root}/retrieval_eval/`:

- MIRACL dev for 12 languages + MSMARCO (EN) + BEIR-PL (PL)

Config: [`configs/retrieval_eval_datasets.yaml`](configs/retrieval_eval_datasets.yaml). Set `paths.retrieval_eval_data_root` in `distill.yaml` (or rely on `gpu_data_root`).

### GPU training

```bash
bash scripts/run_gpu_retrieval_pipeline.sh
```

- **Init:** sequential resume per `languages.yaml` order
- **Checkpoints:** `{output_dir}/retrieval/checkpoints/{lang}/` (legacy `checkpoint_en` / `checkpoint_final`)

### Retrieval eval

Online (MTEB downloads from HuggingFace):

```bash
python scripts/04_eval_retrieval.py --config configs/distill.yaml \
  --model /path/to/retrieval/checkpoint_final --suite en_ko

python scripts/05_compare_retrieval.py --config configs/distill.yaml \
  --student /path/to/retrieval/checkpoint_final --suite en_ko
```

Offline on GPU (uses local retrieval parquet, no internet):

```bash
python scripts/04_eval_retrieval.py --config configs/distill.yaml \
  --model /path/to/retrieval/checkpoint_final --suite all16 --local-retrieval

python scripts/05_compare_retrieval.py --config configs/distill.yaml \
  --student /path/to/retrieval/checkpoint_final --suite all16 --local-retrieval

# Parallel teacher/student/baseline on GPUs 0,1,2
python scripts/05_compare_retrieval.py --config configs/distill.yaml \
  --student /path/to/retrieval/checkpoint_final \
  --baseline /models/harrier-12l-pruned \
  --suite all16 --local-retrieval \
  --parallel --gpus 0,1,2
```

Local retrieval encodes with `eval.retrieval.batch_size` (default **192**). If encode OOMs, lower that value in `configs/distill.yaml`. With `--local-retrieval`, query/corpus embeddings are cached under `{paths.retrieval_eval_data_root|output}/.retrieval_emb_cache/` (keyed by model path, task, subset, max length, prompt). Use `--no-emb-cache` to disable or `--refresh-emb-cache` to force re-encode. nDCG@10 uses batched CUDA matmul when available (exact search; streams corpus chunks).

Tasks: MSMARCO (EN), MIRACLRetrieval (12 langs), BEIR-PL (PL). Suites:

| Suite | Tasks | MIRACL langs |
|-------|-------|--------------|
| `en` | MSMARCO | — |
| `en_ko` | MSMARCO + MIRACL | en, ko only |
| `miracl12` / `wave1` | MIRACL | all configured (12) |
| `all16` | MSMARCO + MIRACL + BEIR-PL | all configured (12) + PL |
| `wave3` | BEIR-PL | — |

Note: `it` / `pt` / `vi` have training corpora but **no** retrieval eval benchmarks. Progress logs print per MIRACL language subset during encode/score.

## Fallback ladder

| Symptom | Action |
|---------|--------|
| STS gap >10% after KO step | Increase `training.losses.cosine` (e.g. `0.1–0.5`) |
| Low pointwise MSE but poor STS/retrieval | Ensure retrieval `score_kl` is on; re-download MIRACL with `--force` after raising `negatives_per_query`, re-embed with `triplet_id` |
| Still lagging | Small contrastive pass on NLI pairs |
| Pruned baseline very low | Check layer removal; consider hidden-state distill |
| KO noisy | Increase KLUE-NLI / KoWiki mix in `datasets.yaml` |

## Project layout

```text
configs/
  distill.yaml              # paths + training hyperparams
  languages.yaml            # 16-lang registry, training order, eval mapping
  datasets.yaml             # STS phase HF dataset definitions (16 langs)
  retrieval_datasets.yaml   # retrieval phase (MIRACL, mMARCO, MAUPQA)
  retrieval_eval_datasets.yaml  # MSMARCO, MIRACL×12, BEIR-PL
scripts/
  01_download_local.py
  01_download_retrieval_local.py
  01_download_retrieval_eval_local.py
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
  validate_dataset_splits.py
src/harrier_distill/
  config.py data.py losses.py model.py eval.py eval_parallel.py eval_progress.py
  retrieval.py retrieval_eval.py sts.py debug.py distributed.py text.py mteb_sts.py
```
