"""Evaluate fine-tuned windows, and optionally compare representations with paired tests.

Every checkpoint yields two representations: the trained readout ``r_l`` and the projected
embedding ``z_l``. Raw PE and frozen readouts can be added for comparison.

Example::

    lpr-evaluate --checkpoints runs/l3_l5/best.pt runs/l6_l8/best.pt runs/l9_l11/best.pt \\
        --split-dir data/splits/internal --image-root data/crops --output-dir runs/eval \\
        --compare pe_l9_l11_trained_readout pe_l3_l5_trained_readout
"""

from __future__ import annotations

import argparse

from lpr.cli.common import (
    add_evaluation_arguments,
    load_evaluation_split,
    resolve_device,
    setup_logging,
)
from lpr.evaluation.runner import evaluate_all
from lpr.models.pe import PerceptionEncoder
from lpr.models.representations import PERawFinal, WindowRepresentations
from lpr.training.trainer import load_window


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_evaluation_arguments(parser)
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="best.pt files written by lpr-finetune")
    parser.add_argument("--include-raw", action="store_true", help="also evaluate raw PE")
    parser.add_argument("--compare", nargs=2, action="append", default=[],
                        metavar=("A", "B"), help="representation pair for paired tests")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging()
    split = load_evaluation_split(args)
    representations = []
    if args.include_raw:
        representations.append(PERawFinal(PerceptionEncoder()))
    for checkpoint in args.checkpoints:
        # Each window gets its own encoder copy because fine-tuning changed its blocks.
        representations.append(WindowRepresentations(load_window(checkpoint, PerceptionEncoder())))
    if not representations:
        raise SystemExit("Nothing to evaluate: pass --checkpoints and/or --include-raw")
    evaluate_all(
        representations, split, args.output_dir, references=args.references,
        gallery_seed=args.gallery_seed, rule=args.rule, target_far=args.target_far,
        pairs=[tuple(pair) for pair in args.compare],
        prototypes=args.prototypes, bootstrap_repeats=args.bootstrap_repeats,
        resampling_unit=args.resampling_unit,
        device=resolve_device(args.device),
        batch_size=args.batch_size, num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
