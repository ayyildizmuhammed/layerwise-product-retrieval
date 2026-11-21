"""Frozen layer-wise sweep of the pretrained PE and the raw baselines.

Evaluates the 24 frozen readouts (layers 1-12, attention and mean pooling), the raw PE final
embedding and, optionally, the raw CLIP B/16 and SigLIP2 B/16 baselines on one split.

Example::

    lpr-frozen-sweep --split-dir data/splits/internal --image-root data/crops \\
        --output-dir runs/frozen_sweep --baselines clip_b16 siglip2_b16
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
from lpr.models.pe import POOLINGS, PerceptionEncoder
from lpr.models.representations import PEFrozenReadouts, PERawFinal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_evaluation_arguments(parser)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="layers to read out (default: all)")
    parser.add_argument("--poolings", nargs="+", choices=POOLINGS, default=list(POOLINGS))
    parser.add_argument("--baselines", nargs="*", default=[],
                        help="raw baselines to add: clip_b16, siglip2_b16")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging()
    split = load_evaluation_split(args)
    encoder = PerceptionEncoder()
    representations = [
        PERawFinal(encoder),
        PEFrozenReadouts(encoder, layers=args.layers, poolings=args.poolings),
    ]
    if args.baselines:
        from lpr.models.baselines import OpenClipBaseline

        representations += [OpenClipBaseline(name) for name in args.baselines]
    evaluate_all(
        representations, split, args.output_dir, references=args.references,
        gallery_seed=args.gallery_seed, rule=args.rule, target_far=args.target_far,
        prototypes=args.prototypes, bootstrap_repeats=args.bootstrap_repeats,
        device=resolve_device(args.device),
        batch_size=args.batch_size, num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
