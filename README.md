# Layer-Wise Product Retrieval

Reference implementation of the method in

> **Layer-Wise Representation Analysis of a Vision Transformer for Fine-Grained Product Retrieval:
> In-Domain Performance, Cross-Domain Generalisation, and Unknown-Product Rejection**

The study asks whether intermediate layers of a vision transformer give a better basis for
exact-SKU product retrieval than its final embedding. It reads out every layer of the Perception
Encoder `PE-Core-B16-224`, fine-tunes equal-sized windows of blocks at different depths with
iterative hard-negative mining, and evaluates retrieval, cross-domain transfer and the rejection
of products that are absent from the catalogue.

This repository provides the code for every step:

| Step | Command |
|---|---|
| Frozen layer-wise readouts of the pretrained PE (layers 1-12, attention and mean pooling), raw PE, CLIP and SigLIP2 baselines | `lpr-frozen-sweep` |
| Depth-specific fine-tuning of the L3-5, L6-8 and L9-11 windows with batch-hard triplet loss and iterative hard-negative mining | `lpr-finetune` |
| SKU-level retrieval, unknown-product rejection, paired statistics | `lpr-evaluate` |

[`docs/method.md`](docs/method.md) gives an overview of the method and where each part lives in
the code.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/ayyildizmuhammed/layerwise-product-retrieval.git
cd layerwise-product-retrieval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[baselines,dev]"
```

### Perception Encoder

The encoder is Meta's [Perception Encoder](https://github.com/facebookresearch/perception_models)
(Bolya et al., 2025). Its code is not copied into this repository; install it from the official
repository, pinned to the commit used in the experiments:

```bash
pip install --no-deps "git+https://github.com/facebookresearch/perception_models@3e352cca660658d4b5c90f42a7808b11469e4c66"
```

`--no-deps` matters: the `perception_models` requirements pin a large stack for its language and
video models. The vision encoder only needs `torch`, `timm`, `einops` and `huggingface_hub`, which
this package already installs. The pretrained weights (`facebook/PE-Core-B16-224`) are downloaded
from Hugging Face on first use. The PE code and weights are subject to their own licence
(`LICENSE.PE` in the official repository).

The CLIP B/16 and SigLIP2 B/16 baselines use [`open_clip`](https://github.com/mlfoundations/open_clip)
(the `baselines` extra).

## Datasets

| Dataset | Split files | Images |
|---|---|---|
| SKU110K evaluation set: 360 SKUs; 259 catalogue SKUs, 10 validation-unknown and 91 test-unknown SKUs | [`data/sku110k`](data/sku110k) | rebuilt from SKU110K with [`scripts/rebuild_sku110k_crops.py`](scripts/rebuild_sku110k_crops.py) |
| RP2K, official split | [`data/rp2k`](data/rp2k) | download from the authors |
| Products-6K, candidate-to-query protocol | [`data/products6k`](data/products6k) | download from the authors |

[`docs/data.md`](docs/data.md) describes each dataset, how the SKU110K crops are rebuilt and
verified, and how the split was built. [`docs/reproduce.md`](docs/reproduce.md) lists the commands
from a fresh clone to the evaluation outputs.

## Data format

Every command reads a *split directory* with one CSV per role. Each CSV has the columns

| Column | Meaning |
|---|---|
| `path` | image path, relative to `--image-root` or absolute |
| `sku_id` | SKU label |
| `family_id` | product family (optional; enables family metrics and same/cross-family unknown segments) |
| `source_group` | source photograph of the crop (optional; resampling unit of the statistics and checked to be disjoint across roles) |

Fine-tuning uses `train.csv` and, optionally, `val.csv`. Evaluation uses `gallery.csv` and
`test_known.csv`, plus `val_known.csv`, `val_unknown.csv` and `test_unknown.csv` for rejection.
Unknown roles hold SKUs that are absent from the gallery.

## Usage

**Frozen layer-wise sweep** of the pretrained PE with the raw baselines:

```bash
lpr-frozen-sweep --split-dir SPLIT --image-root IMAGES --output-dir runs/frozen \
    --baselines clip_b16 siglip2_b16
```

**Fine-tune one window** (repeat with `finetune_l3_l5.yaml` and `finetune_l6_l8.yaml`; the
seeds used are 17, 42 and 54):

```bash
lpr-finetune --config configs/finetune_l9_l11.yaml --split-dir FINETUNE_SPLIT \
    --image-root IMAGES --output-dir runs/l9_l11_seed54 --seed 54
```

**Evaluate** the fine-tuned windows. Each checkpoint gives the trained readout `r_l` and the
projected embedding `z_l`; `--compare` adds paired tests:

```bash
lpr-evaluate --checkpoints runs/l3_l5_seed54/best.pt runs/l6_l8_seed54/best.pt \
    runs/l9_l11_seed54/best.pt --include-raw \
    --split-dir SPLIT --image-root IMAGES --output-dir runs/eval \
    --compare pe_l9_l11_trained_readout pe_l3_l5_trained_readout
```

Common options: `--references 10 1` (nested galleries with up to 10 and 1 reference images per
SKU), `--gallery-seed`, `--rule score|score_margin` (rejection rule), `--target-far 5`,
`--prototypes`, `--bootstrap-repeats 10000`, `--resampling-unit source_group|sku|image`,
`--device cuda|mps|cpu`.

### Outputs

Each evaluation writes to `--output-dir`:

* `summary.csv` - one row per representation, gallery type and reference count: R@1, R@5, R@10,
  mAP@10, macro-SKU R@1, family confusion rates, the selected threshold, FAR overall and per
  unknown segment, and known coverage;
* `results.json` - all metrics, thresholds, rejection operating points and AUROC values;
* `per_query/*.csv` - per-query predictions, correctness and top-1/top-2 scores;
* `paired.csv` - paired differences with source-group bootstrap intervals and randomisation
  p-values, when `--compare` is given.

## Hardware

The experiments ran on an NVIDIA Jetson AGX Orin (CUDA). Evaluation runs on any
device. For fine-tuning, use CUDA or CPU: batches with mined examples vary in size, and Apple MPS
recompiles its kernels for every new shape, which makes training very slow there.

## Tests

```bash
pytest               # unit and end-to-end tests on synthetic data; no model weights needed
ruff check src tests
```

## Citation

If you use this code, please cite the manuscript (citation details will be added on publication)
and the Perception Encoder:

```bibtex
@inproceedings{bolya2025perception,
  title     = {Perception Encoder: The Best Visual Embeddings Are Not at the Output of the Network},
  author    = {Bolya, Daniel and Huang, Po-Yao and Sun, Peize and Cho, Jang Hyun and Madotto, Andrea and Wei, Chen and Ma, Tengyu and Zhi, Jiale and Rajasegaran, Jathushan and Rasheed, Hanoona and Wang, Junke and Monteiro, Marco and Xu, Hu and Dong, Shiyu and Ravi, Nikhila and Li, Daniel and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {38},
  year      = {2025}
}
```

## Licence

Code: MIT (see [`LICENSE`](LICENSE)). Annotations and split files in `data/`: CC BY-NC 4.0. The
Perception Encoder, CLIP and SigLIP2 models and the SKU110K, RP2K and Products-6K images are
subject to their own licences.
