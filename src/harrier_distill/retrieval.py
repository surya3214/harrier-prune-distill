from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterator

from harrier_distill.data import append_corpus_shard, ensure_dir, merge_parquet_shards
from harrier_distill.text import normalize_text


class TextDedupeState:
    def __init__(self, *, lowercase: bool = False, normalize_whitespace: bool = True):
        self.seen: set[str] = set()
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace

    def add_if_new(self, text: str) -> bool:
        cleaned = normalize_text(text, normalize_whitespace=self.normalize_whitespace)
        key = cleaned.lower() if self.lowercase else cleaned
        if not key or key in self.seen:
            return False
        self.seen.add(key)
        return True


def retrieval_row_id(lang: str, role: str, text: str, *, triplet_idx: int | None = None) -> str:
    suffix = f":{triplet_idx}" if triplet_idx is not None else ""
    digest = hashlib.sha1(f"{lang}:{role}:{text}{suffix}".encode("utf-8")).hexdigest()[:16]
    return f"{lang}_{role}_{digest}"


def retrieval_corpus_row(
    *,
    row_id: str,
    text: str,
    lang: str,
    source: str,
    role: str,
    min_chars: int,
    triplet_id: str | None = None,
    normalize_whitespace: bool = True,
) -> dict[str, Any] | None:
    cleaned = normalize_text(text, normalize_whitespace=normalize_whitespace)
    if len(cleaned) < min_chars:
        return None
    row: dict[str, Any] = {
        "id": row_id,
        "text": cleaned,
        "lang": lang,
        "source": source,
        "role": role,
    }
    if triplet_id is not None:
        row["triplet_id"] = triplet_id
    return row


def expand_triplet_rows(
    *,
    lang: str,
    source: str,
    query: str,
    positive: str,
    negatives: list[str],
    min_chars: int,
    triplet_idx: int,
    normalize_whitespace: bool = True,
) -> list[dict[str, Any]]:
    triplet_id = f"{lang}_{source}_{triplet_idx:08d}"
    rows: list[dict[str, Any]] = []

    query_row = retrieval_corpus_row(
        row_id=retrieval_row_id(lang, "query", query, triplet_idx=triplet_idx),
        text=query,
        lang=lang,
        source=source,
        role="query",
        min_chars=min_chars,
        triplet_id=triplet_id,
        normalize_whitespace=normalize_whitespace,
    )
    if query_row is not None:
        rows.append(query_row)

    for doc_idx, passage in enumerate([positive, *negatives]):
        label = "positive" if doc_idx == 0 else "negative"
        doc_row = retrieval_corpus_row(
            row_id=retrieval_row_id(lang, f"doc_{label}", passage, triplet_idx=triplet_idx * 10 + doc_idx),
            text=passage,
            lang=lang,
            source=source,
            role="doc",
            min_chars=min_chars,
            triplet_id=triplet_id,
            normalize_whitespace=normalize_whitespace,
        )
        if doc_row is not None:
            rows.append(doc_row)
    return rows


def _load_hf_dataset(source_cfg: dict[str, Any]):
    from datasets import load_dataset

    hf_path = source_cfg["hf_path"]
    split = source_cfg.get("split", "train")
    streaming = source_cfg.get("streaming", True)
    config = source_cfg.get("config")

    kwargs: dict[str, Any] = {"path": hf_path, "split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def _load_hf_table(source_cfg: dict[str, Any], *, config_key: str, split_key: str):
    from datasets import load_dataset

    hf_path = source_cfg["hf_path"]
    config = source_cfg[config_key]
    split = source_cfg[split_key]
    return load_dataset(hf_path, config, split=split, streaming=False)


def _corpus_text(row: dict[str, Any]) -> str:
    title = row.get("title")
    body = row.get("text") or row.get("passage") or ""
    if title:
        return f"{title}\n{body}".strip()
    return str(body)


def iter_msmarco_triplets(source_cfg: dict[str, Any]) -> Iterator[tuple[str, str, list[str]]]:
    dataset = _load_hf_dataset(source_cfg)
    query_col = source_cfg["query_column"]
    positive_col = source_cfg["positive_column"]
    negative_col = source_cfg["negative_column"]
    negatives_per = int(source_cfg.get("negatives_per_triplet", 1))

    for row in dataset:
        query = row.get(query_col)
        positive = row.get(positive_col)
        negative = row.get(negative_col)
        if not query or not positive:
            continue
        negatives: list[str] = []
        if negative:
            negatives.append(negative)
        if negatives_per <= len(negatives):
            yield str(query), str(positive), negatives[:negatives_per]
        elif negatives:
            yield str(query), str(positive), negatives


def iter_miracl_triplets(source_cfg: dict[str, Any]) -> Iterator[tuple[str, str, list[str]]]:
    queries_table = _load_hf_table(source_cfg, config_key="queries_config", split_key="queries_split")
    corpus_table = _load_hf_table(source_cfg, config_key="corpus_config", split_key="corpus_split")
    qrels = _load_hf_dataset({**source_cfg, "config": source_cfg["qrels_config"], "split": source_cfg["qrels_split"]})

    query_id_col = source_cfg.get("queries_id_column", "_id")
    corpus_row_id_col = source_cfg.get("corpus_row_id_column", "_id")
    qrels_query_col = source_cfg.get("query_id_column", "query-id")
    qrels_corpus_col = source_cfg.get("qrels_corpus_id_column", source_cfg.get("corpus_id_column", "corpus-id"))
    score_col = source_cfg.get("score_column", "score")
    positives_per = int(source_cfg.get("positives_per_query", 1))
    negatives_per = int(source_cfg.get("negatives_per_query", 3))

    query_text_by_id = {str(row[query_id_col]): str(row["text"]) for row in queries_table}
    corpus_text_by_id = {str(row[corpus_row_id_col]): _corpus_text(row) for row in corpus_table}

    grouped: dict[str, dict[str, list[str]]] = {}
    for row in qrels:
        qid = str(row[qrels_query_col])
        cid = str(row[qrels_corpus_col])
        score = int(row[score_col])
        bucket = grouped.setdefault(qid, {"positive": [], "negative": []})
        if score > 0:
            bucket["positive"].append(cid)
        else:
            bucket["negative"].append(cid)

    for qid, buckets in grouped.items():
        query_text = query_text_by_id.get(qid)
        if not query_text:
            continue
        positives = [corpus_text_by_id[cid] for cid in buckets["positive"] if cid in corpus_text_by_id]
        negatives = [corpus_text_by_id[cid] for cid in buckets["negative"] if cid in corpus_text_by_id]
        if not positives or not negatives:
            continue
        yield (
            query_text,
            positives[0],
            negatives[:negatives_per],
        )
        if positives_per > 1:
            for pos in positives[1:positives_per]:
                yield query_text, pos, negatives[:negatives_per]


def collect_retrieval_rows(
    *,
    lang: str,
    lang_cfg: dict[str, Any],
    min_chars: int,
    dedupe_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    dedupe = TextDedupeState(
        lowercase=bool(dedupe_cfg.get("lowercase_for_dedupe", False)),
        normalize_whitespace=bool(dedupe_cfg.get("normalize_whitespace", True)),
    )
    target_triplets = int(lang_cfg["target_triplets"])
    rows: list[dict[str, Any]] = []
    triplet_idx = 0

    for source_cfg in lang_cfg.get("sources", []):
        name = source_cfg["name"]
        if name.startswith("miracl"):
            triplet_iter = iter_miracl_triplets(source_cfg)
        elif "msmarco" in name:
            triplet_iter = iter_msmarco_triplets(source_cfg)
        else:
            raise ValueError(f"Unknown retrieval source '{name}' for language '{lang}'")

        remaining_global = max(target_triplets - triplet_idx, 0)
        source_limit = int(source_cfg.get("target_triplets", remaining_global))
        source_limit = min(source_limit, remaining_global)
        source_added = 0

        for query, positive, negatives in triplet_iter:
            if triplet_idx >= target_triplets or source_added >= source_limit:
                break
            if not dedupe.add_if_new(query):
                continue
            expanded = expand_triplet_rows(
                lang=lang,
                source=name,
                query=query,
                positive=positive,
                negatives=negatives,
                min_chars=min_chars,
                triplet_idx=triplet_idx,
                normalize_whitespace=bool(dedupe_cfg.get("normalize_whitespace", True)),
            )
            if not expanded:
                continue
            rows.extend(expanded)
            triplet_idx += 1
            source_added += 1

        if triplet_idx >= target_triplets:
            break

    return rows


def build_retrieval_corpus(
    *,
    lang: str,
    lang_cfg: dict[str, Any],
    dedupe_cfg: dict[str, Any],
    min_chars: int,
    output_path: Path,
    shard_size: int,
) -> int:
    rows = collect_retrieval_rows(
        lang=lang,
        lang_cfg=lang_cfg,
        min_chars=min_chars,
        dedupe_cfg=dedupe_cfg,
    )
    if not rows:
        raise RuntimeError(f"No retrieval rows collected for language '{lang}'")

    shard_dir = output_path.parent / f".{lang}_retrieval_shards"
    if shard_dir.exists():
        for old_shard in shard_dir.glob("part_*.parquet"):
            old_shard.unlink()
    ensure_dir(shard_dir)

    shard_idx = 0
    buffer: list[dict[str, Any]] = []
    for row in rows:
        buffer.append(row)
        if len(buffer) >= shard_size:
            append_corpus_shard(buffer, shard_dir / f"part_{shard_idx:05d}.parquet")
            buffer = []
            shard_idx += 1
    if buffer:
        append_corpus_shard(buffer, shard_dir / f"part_{shard_idx:05d}.parquet")

    total_rows = merge_parquet_shards(shard_dir, output_path)
    triplet_count = len({row["triplet_id"] for row in rows if "triplet_id" in row})
    print(f"  {lang}: {triplet_count:,} triplets -> {total_rows:,} corpus rows")
    return total_rows


def get_retrieval_lang_configs(cfg: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    pilot = cfg.get("pilot", {})
    if pilot.get("enabled"):
        languages = list(cfg.get("languages", ["en", "ko"]))
        lang_cfgs = {}
        for lang in languages:
            if lang not in cfg.get("pilot_sources", {}):
                continue
            lang_cfgs[lang] = dict(cfg["pilot_sources"][lang])
        return languages, lang_cfgs

    languages = list(cfg.get("languages", ["en", "ko"]))
    lang_cfgs = {lang: cfg[lang] for lang in languages if lang in cfg}
    return languages, lang_cfgs
