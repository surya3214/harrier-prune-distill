"""Unit tests for retrieval triplet loaders (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harrier_distill.retrieval import (
    _group_triplets_by_query,
    collect_retrieval_rows,
    expand_triplet_rows,
    iter_mrtydi_hard_negative_triplets,
    iter_msmarco_triplets,
    iter_unicamp_mmarco_train_id_triplets,
)


class ExpandTripletTests(unittest.TestCase):
    def test_expand_includes_query_pos_and_negs(self) -> None:
        rows = expand_triplet_rows(
            lang="en",
            source="test",
            query="what is paris",
            positive="Paris is the capital of France.",
            negatives=["Lyon is a city.", "Marseille is a port."],
            min_chars=5,
            triplet_idx=0,
        )
        roles = [row["role"] for row in rows]
        self.assertEqual(roles.count("query"), 1)
        self.assertEqual(roles.count("doc"), 3)
        self.assertEqual(len({row["triplet_id"] for row in rows}), 1)


class UnicampTrainIdLoaderTests(unittest.TestCase):
    def test_resolves_id_triples_from_local_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            triples = root / "triples.tsv"
            triples.write_text(
                "\n".join(
                    [
                        "q1\td1\td2",
                        "q2\td3\td4",
                        "q_missing\td1\td2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            query_map = {"q1": "query one text here", "q2": "query two text here"}
            doc_map = {
                "d1": "positive one passage text",
                "d2": "negative one passage text",
                "d3": "positive two passage text",
                "d4": "negative two passage text",
            }

            def fake_download(repo: str, relpath: str, repo_type: str = "dataset") -> str:
                self.assertEqual(repo, "unicamp-dl/mmarco")
                self.assertEqual(relpath, "data/triples.train.ids.small.tsv")
                return str(triples)

            def fake_load_map(*, data_files: str, needed_ids: set[str]) -> dict[str, str]:
                if "queries" in data_files:
                    return {k: v for k, v in query_map.items() if k in needed_ids}
                return {k: v for k, v in doc_map.items() if k in needed_ids}

            source_cfg = {
                "config": "hindi",
                "hf_path": "unicamp-dl/mmarco",
                "translation": "google",
                "negatives_per_triplet": 1,
                "max_id_triples": 10,
            }
            with (
                patch(
                    "huggingface_hub.hf_hub_download",
                    side_effect=fake_download,
                ),
                patch(
                    "harrier_distill.retrieval._load_tsv_id_text_map",
                    side_effect=fake_load_map,
                ),
            ):
                triplets = list(iter_unicamp_mmarco_train_id_triplets(source_cfg))

            self.assertEqual(len(triplets), 2)
            self.assertEqual(triplets[0][0], "query one text here")
            self.assertEqual(triplets[0][1], "positive one passage text")
            self.assertEqual(triplets[0][2], ["negative one passage text"])


class MrTyDiLoaderTests(unittest.TestCase):
    def test_yields_from_passage_lists(self) -> None:
        rows = [
            {
                "query": "who founded Seoul",
                "positive_passages": [{"title": "Seoul", "text": "Seoul is the capital of South Korea."}],
                "negative_passages": [
                    {"title": "Busan", "text": "Busan is a port city."},
                    {"title": "Incheon", "text": "Incheon has an airport."},
                ],
            },
            {
                "query": "empty",
                "positive_passages": [],
                "negative_passages": [{"text": "neg"}],
            },
        ]
        source_cfg = {
            "hf_path": "crystina-z/mrtydi-mContriever-mmarco-HN",
            "config": "korean",
            "split": "train",
            "streaming": True,
            "negatives_per_triplet": 8,
        }
        with patch("harrier_distill.retrieval._load_hf_dataset", return_value=rows):
            triplets = list(iter_mrtydi_hard_negative_triplets(source_cfg))

        self.assertEqual(len(triplets), 1)
        query, positive, negatives = triplets[0]
        self.assertEqual(query, "who founded Seoul")
        self.assertIn("Seoul is the capital", positive)
        self.assertEqual(len(negatives), 2)


class MultiNegColumnLoaderTests(unittest.TestCase):
    def test_reads_negative_1_through_7(self) -> None:
        rows = [
            {
                "query": "capital of france query text",
                "positive": "Paris is the capital of France passage.",
                "negative_1": "Lyon is in France passage text.",
                "negative_2": "Marseille is a port passage text.",
                "negative_3": "Nice is on the coast passage text.",
                "negative_4": "Bordeaux is wine country passage.",
                "negative_5": "Toulouse is in the south passage.",
                "negative_6": "Nantes is in the west passage.",
                "negative_7": "Lille is in the north passage.",
            }
        ]
        source_cfg = {
            "hf_path": "hotchpotch/mmarco-hard-negatives-reranker-filtered",
            "config": "french-hard-negatives-7",
            "split": "train",
            "streaming": True,
            "query_column": "query",
            "positive_column": "positive",
            "negative_columns": [f"negative_{i}" for i in range(1, 8)],
            "negatives_per_triplet": 7,
        }
        with patch("harrier_distill.retrieval._load_hf_dataset", return_value=rows):
            triplets = list(iter_msmarco_triplets(source_cfg))
        self.assertEqual(len(triplets), 1)
        _, positive, negatives = triplets[0]
        self.assertIn("Paris", positive)
        self.assertEqual(len(negatives), 7)


class GroupNegativesByQueryTests(unittest.TestCase):
    def test_packs_until_negatives_per_and_streams_early(self) -> None:
        rows = [
            ("q1 text enough chars", "pos1 text enough", ["neg1a text enough"]),
            ("q1 text enough chars", "pos1 text enough", ["neg1b text enough"]),
            ("q1 text enough chars", "pos1 text enough", ["neg1c text enough"]),
            ("q2 text enough chars", "pos2 text enough", ["neg2a text enough"]),
        ]
        out = list(_group_triplets_by_query(iter(rows), negatives_per=3))
        self.assertEqual(len(out), 2)
        q1 = next(item for item in out if item[0].startswith("q1"))
        self.assertEqual(len(q1[2]), 3)
        q2 = next(item for item in out if item[0].startswith("q2"))
        self.assertEqual(len(q2[2]), 1)

    def test_msmarco_group_flag_packs_rows(self) -> None:
        rows = [
            {"query": "same query text here", "positive": "shared positive passage", "negative": "neg one passage"},
            {"query": "same query text here", "positive": "shared positive passage", "negative": "neg two passage"},
            {"query": "same query text here", "positive": "shared positive passage", "negative": "neg three passage"},
        ]
        source_cfg = {
            "hf_path": "chieunq/mMARCO_vietnamese",
            "split": "train",
            "streaming": True,
            "negatives_per_triplet": 3,
            "group_negatives_by_query": True,
        }
        with patch("harrier_distill.retrieval._load_hf_dataset", return_value=rows):
            triplets = list(iter_msmarco_triplets(source_cfg))
        self.assertEqual(len(triplets), 1)
        self.assertEqual(len(triplets[0][2]), 3)


class CollectRetrievalRowsTests(unittest.TestCase):
    def test_allows_same_query_with_different_docs(self) -> None:
        """Triplet-content dedupe should keep distinct (q, pos, neg) rows."""
        fake_triplets = [
            ("same query text for retrieval", "pos A document text", ["neg A document text"]),
            ("same query text for retrieval", "pos B document text", ["neg B document text"]),
        ]
        lang_cfg = {
            "target_triplets": 10,
            "sources": [
                {
                    "name": "mmarco_triplets",
                    "hf_path": "dummy",
                    "config": "english-triplet-10",
                }
            ],
        }
        with patch(
            "harrier_distill.retrieval.iter_msmarco_triplets",
            return_value=iter(fake_triplets),
        ):
            rows = collect_retrieval_rows(
                lang="en",
                lang_cfg=lang_cfg,
                min_chars=5,
                dedupe_cfg={"normalize_whitespace": True, "lowercase_for_dedupe": False},
            )
        triplet_ids = {row["triplet_id"] for row in rows if "triplet_id" in row}
        self.assertEqual(len(triplet_ids), 2)

    def test_config_targets_are_350k(self) -> None:
        from harrier_distill.config import load_retrieval_datasets_config
        from harrier_distill.retrieval import get_retrieval_lang_configs

        cfg = load_retrieval_datasets_config("configs/retrieval_datasets.yaml")
        langs, lang_cfgs = get_retrieval_lang_configs(cfg)
        self.assertEqual(len(langs), 16)
        for lang in langs:
            self.assertEqual(lang_cfgs[lang]["target_triplets"], 350_000)


if __name__ == "__main__":
    unittest.main()
