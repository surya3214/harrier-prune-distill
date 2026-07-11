#!/usr/bin/env python3
"""Machine-readable retrieval pipeline resume status for shell orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from harrier_distill.config import (
    get_last_completed_retrieval_lang,
    get_next_incomplete_retrieval_lang,
    get_training_order,
    is_retrieval_embedding_complete,
    is_retrieval_lang_complete,
    load_distill_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "distill.yaml"))
    parser.add_argument(
        "--mode",
        choices=("next", "lang", "summary"),
        required=True,
        help="next=START/ALL_DONE; lang=EMBED_/TRAIN_ lines; summary=human one-liner",
    )
    parser.add_argument("--lang", default=None, help="Required for --mode lang")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_distill_config(args.config)

    if args.mode == "next":
        nxt = get_next_incomplete_retrieval_lang(cfg)
        if nxt is None:
            print("ALL_DONE")
        else:
            print(f"START {nxt}")
        return

    if args.mode == "summary":
        order = get_training_order()
        last = get_last_completed_retrieval_lang(cfg)
        nxt = get_next_incomplete_retrieval_lang(cfg)
        done = 0
        for lang in order:
            if is_retrieval_lang_complete(cfg, lang):
                done += 1
            else:
                break
        if nxt is None:
            print(f"Retrieval resume: all {len(order)} languages complete")
        elif last is None:
            print(f"Retrieval resume: starting from {nxt} (0/{len(order)} complete)")
        else:
            print(
                f"Retrieval resume: last complete={last}; "
                f"continuing from {nxt} ({done}/{len(order)} complete)"
            )
        return

    if not args.lang:
        raise SystemExit("--lang is required for --mode lang")
    if args.lang not in get_training_order():
        raise SystemExit(f"Unknown language: {args.lang}")

    embed = "EMBED_DONE" if is_retrieval_embedding_complete(cfg, args.lang) else "EMBED_NEEDED"
    train = "TRAIN_DONE" if is_retrieval_lang_complete(cfg, args.lang) else "TRAIN_NEEDED"
    print(embed)
    print(train)


if __name__ == "__main__":
    main()
