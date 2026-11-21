# Method overview

This page summarises the method and points to the code that implements each part. Defaults are
those of the configs in [`configs/`](../configs) and of `TrainConfig`.

## Retrieval

A representation maps an image to an L2-normalised vector, and the similarity of a query and a
gallery image is their cosine. Because an SKU may have several gallery images, image scores are
collapsed to one score per SKU, the maximum over the SKU's gallery images, and SKUs are ranked by
that score. Every metric is computed on this SKU-level ranking. Search is exact, without an
approximate index.

| Part | Code |
|---|---|
| Cosine scores, SKU collapse, ranking | [`lpr/retrieval/search.py`](../src/lpr/retrieval/search.py) |
| Embedding a list of images | [`lpr/retrieval/embed.py`](../src/lpr/retrieval/embed.py) |

## Layer-wise readouts

The encoder is the Perception Encoder `PE-Core-B16-224`, a 12-block vision transformer. After
any block, the token sequence is passed through the pretrained post-layer normalisation and
pooled into a 768-dimensional readout, either with the pretrained attention pool (the same pool
at every depth) or as the mean of the patch tokens. Blocks above the requested layer are not
executed, and all layers can be read out from a single forward pass. The raw final embedding is
the encoder's standard 1024-dimensional output.

| Part | Code |
|---|---|
| Encoder wrapper, readouts, raw final embedding | [`lpr/models/pe.py`](../src/lpr/models/pe.py) |
| Representations behind one interface | [`lpr/models/representations.py`](../src/lpr/models/representations.py) |
| CLIP B/16 and SigLIP2 B/16 baselines | [`lpr/models/baselines.py`](../src/lpr/models/baselines.py) |

## Depth-specific fine-tuning

A window of consecutive blocks ending at a stop layer is trained together with the post-layer
normalisation, the attention pool and a bias-free 768 x 768 linear head; all other encoder weights
are frozen. One forward pass gives two embeddings: the *trained readout* (the attention-pooled
stop-layer feature) and the *projected embedding* (the same feature after the head). The provided
configs train three windows of three blocks, L3-5, L6-8 and L9-11, with identical settings.

Training minimises a batch-hard triplet loss with cosine distance on the projected embedding.
Batches use PK sampling. Training runs in mining cycles: at the start of every cycle the current
model embeds the training set and retrieves each image's nearest neighbours; different-SKU
neighbours are ranked by similarity, the closest few form a hard bucket and the following ones a
semi-hard bucket, and up to four mined negatives per anchor are added to the batches. In the first
cycle, before the head has been trained, the pretrained stop-layer readout is used for mining.
Validation recall@1 is computed after the first epoch and every second epoch; the best checkpoint
is kept and training stops after four evaluations without improvement.

| Part | Code |
|---|---|
| Fine-tuning window, trained readout and projected embedding | [`lpr/models/window.py`](../src/lpr/models/window.py) |
| Batch-hard triplet loss | [`lpr/training/loss.py`](../src/lpr/training/loss.py) |
| Hard-negative mining | [`lpr/training/mining.py`](../src/lpr/training/mining.py) |
| PK sampler with mined examples | [`lpr/data/sampler.py`](../src/lpr/data/sampler.py) |
| Training loop, optimiser, schedule, validation, checkpoints | [`lpr/training/trainer.py`](../src/lpr/training/trainer.py) |

## Preprocessing

Every image is padded to a square with a neutral grey value, resized to the encoder's input size
and normalised with that encoder's image statistics. Training augmentation is mild: small colour,
brightness and contrast jitter, low-rate JPEG compression, small shifts, scaling and rotation and
occasional motion blur. Horizontal flips and cutout are not used because they mirror package text
or erase small identifying details. Code: [`lpr/data/transforms.py`](../src/lpr/data/transforms.py).

## Evaluation

**Reference galleries.** The gallery keeps up to ten reference images per SKU, and a
single-reference gallery is evaluated alongside it. Galleries for different reference counts are
nested, so the query set stays fixed; different gallery seeds change which images are drawn.
Prototype galleries (one mean embedding per SKU) are available with `--prototypes`.

**Metrics.** Recall@K, mAP@10 (the reciprocal rank truncated at ten, since there is one relevant
SKU per query), macro-SKU recall@1 and the family confusion rate: the share of queries, and of
errors, whose wrong top-1 SKU belongs to the same family as the true SKU.

**Unknown-product rejection.** A query is accepted when its top-1 SKU score reaches a threshold
selected on validation queries only: the lowest score whose false accept rate on the
validation-unknown queries is at most 5 %, or, with `--rule score_margin`, a score and margin
pair from a fixed grid. On the test queries the false accept rate is reported overall and for
same-family and cross-family unknowns, together with the share of known queries accepted and the
accuracy among them. The AUROC of the top-1 score and of the top-1/top-2 margin measures how well
each separates correct from incorrect known predictions.

**Statistics.** Crops from one shelf photograph are dependent, so the photograph (source group)
is the resampling unit: confidence intervals use a source-group clustered bootstrap and paired
differences a source-group paired randomisation test (sign flips per group). Without source
groups the unit is the query image; `--resampling-unit sku` resamples query SKUs instead, as used
for Products-6K. Intervals for the false accept rate resample the unknown SKUs.

| Part | Code |
|---|---|
| Split files and roles | [`lpr/data/manifest.py`](../src/lpr/data/manifest.py) |
| Source-group-disjoint split builder | [`lpr/data/splits.py`](../src/lpr/data/splits.py) |
| Nested and prototype galleries | [`lpr/evaluation/galleries.py`](../src/lpr/evaluation/galleries.py) |
| Closed-set metrics | [`lpr/evaluation/metrics.py`](../src/lpr/evaluation/metrics.py) |
| Threshold selection, FAR, coverage, AUROC | [`lpr/evaluation/rejection.py`](../src/lpr/evaluation/rejection.py) |
| Full evaluation of one representation | [`lpr/evaluation/benchmark.py`](../src/lpr/evaluation/benchmark.py) |
| Bootstrap intervals and paired tests | [`lpr/stats/resampling.py`](../src/lpr/stats/resampling.py) |
| Running several representations and writing outputs | [`lpr/evaluation/runner.py`](../src/lpr/evaluation/runner.py) |

## Settings

| Setting | Default | Config key |
|---|---|---|
| Trainable window | three blocks ending at layer 5, 8 or 11 | `stop_layer`, `window_size` |
| PK sampling | 16 SKUs x 4 images | `P`, `K` |
| Triplet margin | 0.3 | `margin` |
| Learning rate, window / head | 1e-5 / 1e-3 | `lr_encoder`, `lr_head` |
| Weight decay | 1e-4 | `weight_decay` |
| Cosine schedule, warm-up steps | 200 | `warmup_steps` |
| Mining cycles x epochs | 5 x 8 | `cycles`, `epochs_per_cycle` |
| Validation | after epoch 1 and every second epoch | `eval_every` |
| Early stopping | 4 evaluations without improvement | `patience` |
| Mined negatives / positives per anchor | 4 / 2 | `num_negatives`, `num_positives` |
| Neighbours examined per anchor in mining | 80 | `mining.top_k` |
| Hard negatives per anchor in mining | 4 | `mining.num_hard` |
| Input size, padding value | 224, 128 | `ImageTransform` |
| Training seed | 54 (`--seed` overrides) | `seed` |
| Rejection target | 5 % validation-unknown FAR | `--target-far` |
| Bootstrap and randomisation samples | 10,000 | `--bootstrap-repeats` |
