# Data

The repository ships annotations and split files only. No image is redistributed: the SKU110K
crops are rebuilt from the official SKU110K images by a script, and RP2K and Products-6K are
downloaded from their authors.

Every split directory follows the format described in the [README](../README.md#data-format):
one CSV per role with the columns `path`, `sku_id`, `family_id` and `source_group`.

## SKU110K evaluation set

An evaluation set of 360 SKUs from 67 brands, with product crops cut from the shelf photographs
of SKU110K (Goldman et al., 2019) and labelled at SKU level.

SKU110K marks every product on a shelf with a box, but all boxes share one class. Every crop of
this set was assigned to an SKU manually; visually similar crops of a different size, flavour,
formulation or package are separate SKUs. A family groups the SKUs of one brand; its members are
siblings that usually share the package design and differ in a small cue.

### Files

| File | Content |
|---|---|
| [`data/sku110k/annotations.csv`](../data/sku110k/annotations.csv) | one row per crop: `crop_id`, `filepath`, `sku_id`, `brand_id`, `sku110k_image`, `box_index`, `width`, `height`, `sha256` |
| [`data/sku110k/splits/`](../data/sku110k/splits) | `gallery.csv`, `val_known.csv`, `test_known.csv`, `val_unknown.csv`, `test_unknown.csv` and `split_summary.json` |

`box_index` is the row number of the crop's box among the rows of its image in the official
SKU110K annotation files; `sha256` is the hash of the rebuilt crop.

### Rebuilding the crops

1. Download SKU110K from its authors (about 12 GB), following the
   [official repository](https://github.com/eg4000/SKU110K_CVPR19), and extract it. The folder
   must contain `images/` and `annotations/`. SKU110K is provided by its owners for academic and
   non-commercial use only; read its terms before downloading.
2. Rebuild and verify the crops:

```bash
python scripts/rebuild_sku110k_crops.py --sku110k-root SKU110K_fixed --output-dir data/crops
python scripts/verify_sku110k_crops.py --crops-dir data/crops
```

Each box is expanded by 5 % of its size on every side, rounded outward to whole pixels, clamped
to the image and saved as JPEG with quality 95. The verification compares every file with its
published SHA-256.

### Split

The split was built with `scripts/build_sku110k_splits.py` (seed 42) from the rules in
[`lpr/data/splits.py`](../src/lpr/data/splits.py). The source group of a crop is its SKU110K
photograph, and every photograph belongs to exactly one role.

| Role | SKUs |
|---|---:|
| Gallery (catalogue) | 259 |
| Validation, known | 116 |
| Test query, known | 259 |
| Validation, unknown | 10 |
| Test query, unknown | 91 |

Unknown SKUs never enter the gallery. Of the 91 test-unknown SKUs, 28 are *same-family* unknowns
(another SKU of their brand is in the catalogue) and 63 *cross-family* unknowns. Known crops that
share a photograph with an unknown SKU are removed, and queries byte-identical to a gallery
image are removed. With `--references 10` the gallery keeps up to ten crops
per SKU.

## RP2K

RP2K (Peng et al., 2020) is evaluated with its official split: official training images form the
gallery and official test images supply the validation and test queries. The evaluated cohort
holds 450 of the 1,819 SKUs with at least ten training and two test images, drawn in a fixed
order with seed 42; 135 of them are held out as unknown products.

| Role | SKUs |
|---|---:|
| Gallery (ten images per SKU) | 315 |
| Validation, known | 313 |
| Test query, known | 313 |
| Validation, unknown | 11 |
| Test query, unknown | 124 |

Download RP2K from its authors and pass its root as `--image-root`; the paths in
[`data/rp2k/splits/`](../data/rp2k/splits) are relative to that root (`train/...`, `test/...`).
RP2K has no shelf-scene provenance and no family taxonomy, so `source_group` and `family_id` are
empty and only overall FAR is reported. Use `--rule score_margin` for RP2K.

## Products-6K

Products-6K (Georgiadis et al., 2021) pairs studio-style candidate images with photographs of
products held by hand in a supermarket. The official candidate-to-query protocol is used: the
candidate images of 512 SKUs form the gallery and supermarket photographs of 104 of these SKUs
are the queries; the other gallery SKUs act as distractors. There is no unknown split.

Download the public release and pass its root as `--image-root`; the paths in
[`data/products6k/splits/`](../data/products6k/splits) are relative to that root
(`Candidate_Set/Candidate_Images_Selected_500/...`, `Query_Set/Query_Images/...`).

## Licences

The SKU110K annotations, taxonomy and split files in `data/` are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). SKU110K, RP2K and Products-6K
images remain subject to the terms of their owners and are not part of this repository.
