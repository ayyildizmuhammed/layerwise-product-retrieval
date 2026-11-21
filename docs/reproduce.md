# Reproducing the evaluation

Step-by-step commands, from a fresh clone to the evaluation outputs. Install the package and the
Perception Encoder first ([README](../README.md#installation)).

## 1. Prepare the data

**SKU110K evaluation set** - rebuild and verify the crops ([details](data.md#sku110k-evaluation-set)):

```bash
python scripts/rebuild_sku110k_crops.py --sku110k-root /path/to/SKU110K_fixed --output-dir data/crops
python scripts/verify_sku110k_crops.py --crops-dir data/crops
```

The split is already in `data/sku110k/splits`. To rebuild it from the annotations:

```bash
python scripts/build_sku110k_splits.py --output-dir data/sku110k/splits
```

**RP2K and Products-6K** - download both datasets from their authors; the split files in
`data/rp2k/splits` and `data/products6k/splits` refer to paths inside the downloaded folders.

## 2. Frozen layer-wise readouts and raw baselines

All twelve PE layers with attention and mean pooling, the raw PE final embedding and the raw
CLIP B/16 and SigLIP2 B/16 baselines, with ten and one reference image per SKU:

```bash
lpr-frozen-sweep --split-dir data/sku110k/splits --image-root data/crops \
    --output-dir runs/sku110k_frozen --baselines clip_b16 siglip2_b16 --prototypes

lpr-frozen-sweep --split-dir data/rp2k/splits --image-root /path/to/RP2K \
    --output-dir runs/rp2k_frozen --rule score_margin

lpr-frozen-sweep --split-dir data/products6k/splits --image-root /path/to/Products-6K \
    --output-dir runs/products6k_frozen
```

## 3. Depth-specific fine-tuning

Fine-tuning needs a labelled training collection in the same CSV format (`train.csv` and,
optionally, `val.csv` in one directory). Train the three windows with the three training seeds:

```bash
for window in l3_l5 l6_l8 l9_l11; do
  for seed in 17 42 54; do
    lpr-finetune --config configs/finetune_${window}.yaml --split-dir /path/to/train_split \
        --image-root /path/to/train_images --output-dir runs/${window}_seed${seed} --seed ${seed}
  done
done
```

Each run writes `best.pt` (trainable weights only), `last.pt`, `history.json` and `summary.json`.
Use a CUDA device; see the [README](../README.md#hardware).

## 4. Evaluate the fine-tuned windows

```bash
lpr-evaluate --checkpoints runs/l3_l5_seed54/best.pt runs/l6_l8_seed54/best.pt \
    runs/l9_l11_seed54/best.pt --include-raw \
    --split-dir data/sku110k/splits --image-root data/crops --output-dir runs/sku110k_seed54 \
    --compare pe_l9_l11_trained_readout pe_l3_l5_trained_readout \
    --compare pe_l9_l11_trained_readout pe_l6_l8_trained_readout
```

The same command with the RP2K or Products-6K split directory evaluates cross-domain transfer
(add `--rule score_margin` for RP2K).

## 5. Read the results

`runs/<name>/summary.csv` holds one row per representation and reference count; `results.json`
holds every metric, the rejection thresholds and operating points and the AUROC values;
`paired.csv` holds the paired comparisons with source-group bootstrap intervals and
randomisation p-values; `per_query/` holds the per-query predictions for further analysis.
