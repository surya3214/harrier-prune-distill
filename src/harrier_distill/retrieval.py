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


def _load_hf_csv_stream(
    *,
    data_files: str | list[str],
    split: str = "train",
    sep: str = ",",
    column_names: list[str] | None = None,
):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "data_files": data_files,
        "split": split,
        "streaming": True,
    }
    if sep != ",":
        kwargs["sep"] = sep
    if column_names:
        kwargs["column_names"] = column_names
    return load_dataset("csv", **kwargs)


def iter_unicamp_mmarco_triplets(source_cfg: dict[str, Any]) -> Iterator[tuple[str, str, list[str]]]:
    """Yield triplets from unicamp-dl/mmarco TSV + BM25 run files (Parquet-free)."""
    from collections import defaultdict

    from huggingface_hub import hf_hub_download

    lang = source_cfg["config"]
    translation = source_cfg.get("translation", "google")
    repo = source_cfg.get("hf_path", "unicamp-dl/mmarco")
    run_suffix = source_cfg.get("run_suffix", f"{lang}-msmarco")
    negatives_per = int(source_cfg.get("negatives_per_triplet", 1))
    base = f"hf://datasets/{repo}/data/{translation}"
    tsv_kwargs = {"sep": "\t", "column_names": ["id", "text"]}
    queries_relpath = source_cfg.get(
        "queries_relpath",
        f"queries/dev/{lang}_queries.dev.small.tsv",
    )

    queries_ds = _load_hf_csv_stream(
        data_files=f"{base}/{queries_relpath}",
        **tsv_kwargs,
    )

    query_by_id: dict[str, str] = {}
    for row in queries_ds:
        query_by_id[str(row["id"])] = str(row["text"])

    run_path = hf_hub_download(
        repo,
        f"data/{translation}/runs/run.bm25_{run_suffix}.txt",
        repo_type="dataset",
    )
    ranked: dict[str, list[tuple[int, str]]] = defaultdict(list)
    needed_doc_ids: set[str] = set()
    with open(run_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, doc_id, rank = parts
            ranked[qid].append((int(rank), doc_id))
            needed_doc_ids.add(doc_id)

    corpus_ds = _load_hf_csv_stream(
        data_files=f"{base}/collections/{lang}_collection.tsv",
        **tsv_kwargs,
    )
    corpus_by_id: dict[str, str] = {}
    for row in corpus_ds:
        doc_id = str(row["id"])
        if doc_id not in needed_doc_ids:
            continue
        corpus_by_id[doc_id] = str(row["text"])
        if len(corpus_by_id) >= len(needed_doc_ids):
            break

    for qid, doc_ranks in ranked.items():
        query = query_by_id.get(qid)
        if not query:
            continue
        doc_ids = [doc_id for _, doc_id in sorted(doc_ranks, key=lambda item: item[0])]
        if len(doc_ids) < 2:
            continue
        positive = corpus_by_id.get(doc_ids[0])
        if not positive:
            continue
        negatives = [corpus_by_id[doc_id] for doc_id in doc_ids[1:] if doc_id in corpus_by_id]
        if not negatives:
            continue
        yield query, positive, negatives[:negatives_per]


def _load_tsv_id_text_map(
    *,
    data_files: str,
    needed_ids: set[str],
) -> dict[str, str]:
    """Stream a unicamp id\\ttext TSV and keep rows whose id is in ``needed_ids``."""
    out: dict[str, str] = {}
    if not needed_ids:
        return out
    dataset = _load_hf_csv_stream(
        data_files=data_files,
        sep="\t",
        column_names=["id", "text"],
    )
    for row in dataset:
        doc_id = str(row["id"])
        if doc_id not in needed_ids:
            continue
        out[doc_id] = str(row["text"])
        if len(out) >= len(needed_ids):
            break
    return out


def iter_unicamp_mmarco_train_id_triplets(
    source_cfg: dict[str, Any],
) -> Iterator[tuple[str, str, list[str]]]:
    """Yield triplets from unicamp MS MARCO train ID triples + translated texts.

    Uses ``data/triples.train.ids.small.tsv`` (qid, positive_pid, negative_pid) with
    ``queries/train/{lang}_queries.train.tsv`` and ``collections/{lang}_collection.tsv``.
    """
    from huggingface_hub import hf_hub_download

    lang = source_cfg["config"]
    translation = source_cfg.get("translation", "google")
    repo = source_cfg.get("hf_path", "unicamp-dl/mmarco")
    negatives_per = int(source_cfg.get("negatives_per_triplet", 1))
    max_triples = int(source_cfg.get("max_id_triples", 2_000_000))
    base = f"hf://datasets/{repo}/data/{translation}"
    triples_relpath = source_cfg.get("triples_relpath", "data/triples.train.ids.small.tsv")
    queries_relpath = source_cfg.get(
        "queries_relpath",
        f"queries/train/{lang}_queries.train.tsv",
    )
    collection_relpath = source_cfg.get(
        "collection_relpath",
        f"collections/{lang}_collection.tsv",
    )

    triples_path = hf_hub_download(repo, triples_relpath, repo_type="dataset")
    id_triples: list[tuple[str, str, str]] = []
    needed_query_ids: set[str] = set()
    needed_doc_ids: set[str] = set()
    with open(triples_path, encoding="utf-8") as handle:
        for line in handle:
            if len(id_triples) >= max_triples:
                break
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pos_id, neg_id = str(parts[0]), str(parts[1]), str(parts[2])
            id_triples.append((qid, pos_id, neg_id))
            needed_query_ids.add(qid)
            needed_doc_ids.add(pos_id)
            needed_doc_ids.add(neg_id)

    query_by_id = _load_tsv_id_text_map(
        data_files=f"{base}/{queries_relpath}",
        needed_ids=needed_query_ids,
    )
    corpus_by_id = _load_tsv_id_text_map(
        data_files=f"{base}/{collection_relpath}",
        needed_ids=needed_doc_ids,
    )

    for qid, pos_id, neg_id in id_triples:
        query = query_by_id.get(qid)
        positive = corpus_by_id.get(pos_id)
        negative = corpus_by_id.get(neg_id)
        if not query or not positive or not negative:
            continue
        yield query, positive, [negative][:negatives_per]


def iter_mrtydi_hard_negative_triplets(
    source_cfg: dict[str, Any],
) -> Iterator[tuple[str, str, list[str]]]:
    """Yield triplets from crystina-z/mrtydi-mContriever-mmarco-HN style datasets."""
    dataset = _load_hf_dataset(source_cfg)
    query_col = source_cfg.get("query_column", "query")
    positives_col = source_cfg.get("positives_column", "positive_passages")
    negatives_col = source_cfg.get("negatives_column", "negative_passages")
    negatives_per = int(source_cfg.get("negatives_per_triplet", 8))

    def _passage_text(passage: Any) -> str:
        if isinstance(passage, dict):
            title = passage.get("title") or ""
            body = passage.get("text") or passage.get("passage") or ""
            if title:
                return f"{title}\n{body}".strip()
            return str(body).strip()
        return str(passage).strip()

    for row in dataset:
        query = row.get(query_col)
        positives = row.get(positives_col) or []
        negatives = row.get(negatives_col) or []
        if not query or not positives or not negatives:
            continue
        positive_text = _passage_text(positives[0])
        negative_texts = [_passage_text(item) for item in negatives if _passage_text(item)]
        if not positive_text or not negative_texts:
            continue
        yield str(query), positive_text, negative_texts[:negatives_per]


def iter_msmarco_triplets(source_cfg: dict[str, Any]) -> Iterator[tuple[str, str, list[str]]]:
    """Yield triplets from query/positive/negative column datasets.

    Supports:
    - single ``negative`` column
    - multi-neg columns via ``negative_columns: [negative_1, ...]`` (hard-negatives-7)
    - optional ``group_negatives_by_query: true`` to pack 1-neg rows into multi-neg triplets
    """
    dataset = _load_hf_dataset(source_cfg)
    query_col = source_cfg.get("query_column", "query")
    positive_col = source_cfg.get("positive_column", "positive")
    negative_col = source_cfg.get("negative_column", "negative")
    negative_columns = source_cfg.get("negative_columns")
    negatives_per = int(source_cfg.get("negatives_per_triplet", 1))
    group_by_query = bool(source_cfg.get("group_negatives_by_query", False))

    def _row_negatives(row: dict[str, Any]) -> list[str]:
        negatives: list[str] = []
        if negative_columns:
            for col in negative_columns:
                value = row.get(col)
                if value:
                    negatives.append(str(value))
        else:
            value = row.get(negative_col)
            if value:
                negatives.append(str(value))
        return negatives

    if not group_by_query:
        for row in dataset:
            query = row.get(query_col)
            positive = row.get(positive_col)
            if not query or not positive:
                continue
            negatives = _row_negatives(row)
            if not negatives:
                continue
            yield str(query), str(positive), negatives[:negatives_per]
        return

    yield from _group_triplets_by_query(
        (
            (str(row[query_col]), str(row[positive_col]), _row_negatives(row))
            for row in dataset
            if row.get(query_col) and row.get(positive_col)
        ),
        negatives_per=negatives_per,
    )


def _group_triplets_by_query(
    triplets: Iterator[tuple[str, str, list[str]]],
    *,
    negatives_per: int,
) -> Iterator[tuple[str, str, list[str]]]:
    """Pack 1-neg (or sparse) rows into multi-neg triplets keyed by query text.

    Yields a query as soon as it reaches ``negatives_per`` so callers can stop
    early at a triplet target without buffering the full source.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for query, positive, negatives in triplets:
        bucket = buckets.get(query)
        if bucket is None:
            bucket = {"positive": positive, "negatives": [], "seen": set()}
            buckets[query] = bucket
        for neg in negatives:
            if not neg or neg in bucket["seen"] or neg == bucket["positive"]:
                continue
            bucket["negatives"].append(neg)
            bucket["seen"].add(neg)
            if len(bucket["negatives"]) >= negatives_per:
                break
        if len(bucket["negatives"]) >= negatives_per:
            yield query, bucket["positive"], bucket["negatives"][:negatives_per]
            del buckets[query]

    for query, bucket in buckets.items():
        if not bucket["negatives"]:
            continue
        yield query, bucket["positive"], bucket["negatives"][:negatives_per]


def iter_grouped_triplet_stream(
    base_iter: Iterator[tuple[str, str, list[str]]],
    *,
    negatives_per: int,
    group_by_query: bool,
) -> Iterator[tuple[str, str, list[str]]]:
    """Optionally wrap a triplet iterator with query-based negative packing."""
    if not group_by_query:
        for query, positive, negatives in base_iter:
            if not negatives:
                continue
            yield query, positive, negatives[:negatives_per]
        return
    yield from _group_triplets_by_query(base_iter, negatives_per=negatives_per)

def iter_maupqa_triplets(source_cfg: dict[str, Any]) -> Iterator[tuple[str, str, list[str]]]:
    """Yield (query, positive, negatives) from ipipan/maupqa question-passage pairs."""
    from collections import defaultdict

    loader = source_cfg.get("loader")
    if loader == "maupqa_csv" or source_cfg.get("hf_path") == "ipipan/maupqa":
        subsets = source_cfg.get(
            "csv_subsets",
            ["msmarco", "nq", "poquad", "mqa", "mkqa"],
        )
        repo = source_cfg.get("hf_path", "ipipan/maupqa")
        data_files = [f"hf://datasets/{repo}/data/{subset}/train-v2.0.0.csv" for subset in subsets]
        dataset = _load_hf_csv_stream(data_files=data_files)
    else:
        dataset = _load_hf_dataset(source_cfg)
    query_col = source_cfg.get("query_column", "question")
    negatives_per = int(source_cfg.get("negatives_per_triplet", 3))
    title_col = source_cfg.get("title_column", "passage_title")
    text_col = source_cfg.get("text_column", "passage_text")
    relevant_col = source_cfg.get("relevant_column", "relevant")
    group_col = source_cfg.get("group_column", "question_id")

    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"question": None, "positive": [], "negative": []})
    for row in dataset:
        qid = str(row[group_col])
        question = row.get(query_col)
        if not question:
            continue
        title = row.get(title_col) or ""
        body = row.get(text_col) or ""
        passage = f"{title}\n{body}".strip() if title else str(body)
        if not passage:
            continue
        bucket = grouped[qid]
        bucket["question"] = str(question)
        if bool(row.get(relevant_col, False)):
            bucket["positive"].append(passage)
        else:
            bucket["negative"].append(passage)

    for bucket in grouped.values():
        if not bucket["positive"] or not bucket["negative"]:
            continue
        yield (
            bucket["question"],
            bucket["positive"][0],
            bucket["negative"][:negatives_per],
        )


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
        loader = source_cfg.get("loader")
        if name.startswith("miracl"):
            triplet_iter = iter_miracl_triplets(source_cfg)
        elif loader == "unicamp_mmarco_train_ids":
            triplet_iter = iter_unicamp_mmarco_train_id_triplets(source_cfg)
        elif loader == "unicamp_mmarco":
            triplet_iter = iter_unicamp_mmarco_triplets(source_cfg)
        elif loader == "mrtydi_hard_negatives" or name.startswith("mrtydi"):
            triplet_iter = iter_mrtydi_hard_negative_triplets(source_cfg)
        elif loader == "maupqa_csv" or name == "maupqa":
            triplet_iter = iter_maupqa_triplets(source_cfg)
        elif "msmarco" in name or "mmarco" in name:
            triplet_iter = iter_msmarco_triplets(source_cfg)
        else:
            raise ValueError(f"Unknown retrieval source '{name}' for language '{lang}'")

        # Multi-neg packing for 1-neg sources (unicamp train IDs, plain mMARCO, etc.).
        # msmarco path already handles group_negatives_by_query internally.
        if loader in {"unicamp_mmarco_train_ids", "unicamp_mmarco"} and bool(
            source_cfg.get("group_negatives_by_query", False)
        ):
            triplet_iter = iter_grouped_triplet_stream(
                triplet_iter,
                negatives_per=int(source_cfg.get("negatives_per_triplet", 7)),
                group_by_query=True,
            )

        remaining_global = max(target_triplets - triplet_idx, 0)
        source_limit = int(source_cfg.get("target_triplets", remaining_global))
        source_limit = min(source_limit, remaining_global)
        source_added = 0

        for query, positive, negatives in triplet_iter:
            if triplet_idx >= target_triplets or source_added >= source_limit:
                break
            # Dedupe exact triplet content (not query alone) so multi-neg / multi-pos
            # MS MARCO rows can contribute distinct training triplets up to target.
            triplet_key = "\n".join([query, positive, *negatives])
            if not dedupe.add_if_new(triplet_key):
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
) -> tuple[int, int]:
    """Build retrieval parquet corpus.

    Returns:
        (total_rows, triplet_count)
    """
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
    return total_rows, triplet_count


def get_retrieval_lang_configs(cfg: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    pilot = cfg.get("pilot", {})
    languages = list(cfg.get("languages", ["en", "ko"]))

    if pilot.get("enabled"):
        lang_cfgs: dict[str, dict[str, Any]] = {}
        pilot_triplets = int(pilot.get("triplets_per_lang", 10_000))
        for lang in languages:
            if lang in cfg.get("pilot_sources", {}):
                lang_cfgs[lang] = dict(cfg["pilot_sources"][lang])
            elif lang in cfg:
                lang_cfg = dict(cfg[lang])
                lang_cfg["target_triplets"] = min(int(lang_cfg.get("target_triplets", pilot_triplets)), pilot_triplets)
                lang_cfgs[lang] = lang_cfg
        return languages, lang_cfgs

    lang_cfgs = {lang: cfg[lang] for lang in languages if lang in cfg}
    return languages, lang_cfgs
