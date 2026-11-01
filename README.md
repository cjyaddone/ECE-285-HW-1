# Conditional VAE for Chest X-rays

This implementation trains a convolutional conditional variational autoencoder
on the NORMAL and PNEUMONIA images. It uses a class-balanced training sampler,
a stratified validation split, spatial and feature-wise class conditioning, KL
warm-up, and an L1 + SSIM + edge reconstruction objective.

## Setup

From the `HW1` directory, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The default data path is:

```text
dataset/chest_xray/chest_xray/
├── train/{NORMAL,PNEUMONIA}
├── val/{NORMAL,PNEUMONIA}
└── test/{NORMAL,PNEUMONIA}
```

The original `val` directory has only 16 images, so training creates a
reproducible, stratified 90/10 split from the original training set. The test
set is not used during training.

## Train

```bash
python train.py
```

Useful options include `--batch-size`, `--epochs`, `--latent-dim`, `--beta`,
and `--workers`. For example, a smaller run for checking the pipeline is:

```bash
python train.py --epochs 2 --batch-size 16 --workers 0
```

Checkpoints are saved as `outputs/spatial_film/best.pt` and
`outputs/spatial_film/last.pt`. Resume with:

```bash
python train.py --resume outputs/spatial_film/last.pt
```

The default `spatial_film` architecture supplies one-hot class maps to the
encoder and modulates every decoder stage with the class embedding. The earlier
bottleneck-only architecture remains available with `--architecture
bottleneck`, and its existing checkpoints remain compatible.

The tuned defaults use 200 epochs, a 30-epoch KL warm-up, and `beta=3e-4`.
Use separate folders when comparing this model with an earlier run:

```bash
python train.py \
  --output-dir outputs/spatial_film \
  --log-dir runs/spatial_film
```

## TensorBoard

Start TensorBoard in a second terminal:

```bash
tensorboard --logdir runs
```

The dashboard contains total, L1, SSIM, edge, and KL losses for training and
validation, along with the KL weight, learning rate, reconstructions, and
conditional random samples.

## Generate images

```bash
python generate.py --checkpoint outputs/spatial_film/best.pt --num-per-class 16
```

This writes individual class-conditional images, a combined sample grid, and
test-set reconstructions to `generated/`.

## FID and Inception Score

After training, evaluate the best checkpoint against the held-out test set:

```bash
python evaluate.py --checkpoint outputs/spatial_film/best.pt
```

The metrics are written to `outputs/spatial_film/metrics.json`. The script
follows Ignite's FID and Inception Score interfaces and converts both real and
generated images to three-channel 299 x 299 inputs. The pretrained Inception
network downloads its ImageNet weights the first time it is used.

These scores are most useful for comparing runs that use the same evaluation
script and sample count. ImageNet features are not a measure of clinical
correctness for chest X-rays.

## Homework report and KLD sweep

The reproducible three-setting KLD-weight study and submission-ready report are
under `report/`. See `report/README.md` for the exact training, evaluation,
generation, and report-build commands.

The 200-epoch sweep selected `beta=3e-4`: its held-out FID was `0.07789`,
compared with `0.11842` for `1e-4` and `0.13260` for `1e-3` (624 samples per
setting with the same evaluation pipeline). The editable report source, final
PDF, plots, sample grids, and machine-readable summary are all kept together in
that directory.
