"""Unit tests for retrieval triplet loaders (no network)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harrier_distill.retrieval import (
    collect_retrieval_rows,
    expand_triplet_rows,
    iter_mrtydi_hard_negative_triplets,
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


if __name__ == "__main__":
    unittest.main()
