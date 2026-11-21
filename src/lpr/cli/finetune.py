"""Fine-tune one depth window with iterative hard-negative mining.

Example::

    lpr-finetune --config configs/finetune_l9_l11.yaml --split-dir data/splits/finetune \\
        --image-root data/crops --output-dir runs/l9_l11_seed54

The split directory holds ``train.csv`` and, optionally, ``val.csv``. Without ``val.csv`` a
per-SKU validation split of ``--val-ratio`` of the images is drawn from ``train.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lpr.cli.common import resolve_device, setup_logging
from lpr.data.manifest import TRAINING_ROLES, load_split, write_manifest
from lpr.models.pe import PerceptionEncoder
from lpr.models.window import WindowModel
from lpr.training.trainer import TrainConfig, Trainer, split_train_val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="YAML file with TrainConfig fields")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="override the training seed")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging()
    values = yaml.safe_load(Path(args.config).read_text()) or {}
    if args.seed is not None:
        values["seed"] = args.seed
    config = TrainConfig.from_dict(values)

    split = load_split(args.split_dir, TRAINING_ROLES, args.image_root)
    if "train" not in split:
        raise SystemExit(f"{args.split_dir} has no train.csv")
    train, val = split["train"], split.get("val")
    if val is None:
        train, val = split_train_val(train, args.val_ratio, config.seed)
        output = Path(args.output_dir)
        write_manifest(train, output / "splits" / "train.csv")
        write_manifest(val, output / "splits" / "val.csv")

    model = WindowModel(PerceptionEncoder(), config.stop_layer, config.window_size)
    trainer = Trainer(model, train, val, args.output_dir, config, resolve_device(args.device))
    summary = trainer.fit()
    print(summary)


if __name__ == "__main__":
    main()
