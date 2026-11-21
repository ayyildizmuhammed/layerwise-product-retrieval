"""Shared command-line helpers."""

from __future__ import annotations

import argparse
import logging

import torch

from lpr.data.manifest import EVALUATION_ROLES, check_disjoint_groups, load_split
from lpr.retrieval.embed import default_device


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split-dir", required=True,
                        help="directory with gallery.csv, test_known.csv and optional "
                             "val_known.csv, val_unknown.csv and test_unknown.csv")
    parser.add_argument("--image-root", default=None,
                        help="root for relative image paths in the split CSVs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--references", type=int, nargs="+", default=[10, 1],
                        help="reference images per SKU (nested galleries)")
    parser.add_argument("--gallery-seed", type=int, default=42)
    parser.add_argument("--rule", choices=["score", "score_margin"], default="score",
                        help="rejection threshold rule (score_margin is the RP2K rule)")
    parser.add_argument("--target-far", type=float, default=5.0,
                        help="validation-unknown FAR target in percent")
    parser.add_argument("--prototypes", action="store_true",
                        help="also evaluate prototype galleries (mean embedding per SKU)")
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto)")


def load_evaluation_split(args: argparse.Namespace) -> dict:
    split = load_split(args.split_dir, EVALUATION_ROLES, args.image_root)
    for role in ("gallery", "test_known"):
        if role not in split:
            raise SystemExit(f"{args.split_dir} has no {role}.csv")
    check_disjoint_groups(split)
    return split


def resolve_device(name: str | None) -> torch.device:
    return torch.device(name) if name else default_device()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
