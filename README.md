# Microscope Image Enhancer

A self-supervised denoising tool for metaphase and anaphase microscopy images. It cleans up noisy or low-quality microscope images without requiring paired clean/noisy training data a common bottleneck in real microscopy datasets, where a genuinely "clean" reference image rarely exists for the same field of view.

![Python](https://img.shields.io/badge/python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-orange)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this is

Most image denoising models are trained on pairs of (noisy, clean) images. Real microscope images almost never come with a matching clean version so this project instead uses a blind-spot self-supervised training scheme (inspired by Noise2Void): a small UNet is trained to predict the true value of randomly hidden pixels, using only their surrounding context. Since real structure is spatially predictable and noise is not, the network learns to reconstruct structure and suppress noise, with no clean reference images required at any point.

## How it works

1. Masking (training only) for every training patch, ~2% of pixels are randomly hidden and replaced with a nearby pixel's value. The network must recover the true value at exactly those hidden spots, using only the surrounding context.
2. Architecture a UNet (~1.93M parameters): 3 encoder stages that shrink the image while extracting increasingly abstract structure, a bottleneck, and 3 decoder stages that rebuild full resolution, aided by skip connections that preserve fine detail lost during shrinking.
3. Inference a trained model only understands fixed-size patches, so full-size images are processed using an overlapping sliding window, blended smoothly (Hann-window weighting) to avoid visible tile seams.
4. The app a Gradio-based drag-and-drop interface wraps the trained model for direct use, no code required.

## Dataset

| | |
|---|---|
| Total images | 1,410 (149 anaphase, 1,261 metaphase) |
| Source | MIDOG21 and TUPAC16 (public breast cancer histology benchmarks) |
| Format | Grayscale PNG, 128×128 pixels |
| Split | Stratified 80/10/10 (train/val/test), split per-class to preserve proportional representation |

> Dataset images are **not included** in this repository (see `.gitignore`). Download MIDOG21 / TUPAC16 from their official sources and place images under `Data/Raw/<class_name>/`.

## Experiment: does training patch size matter?

Two identical UNet architectures were trained under the same self-supervised scheme, differing only in training patch size:

| Model | Patch size | Examples/epoch | Mean test loss | Std | Min | Max |
|---|---|---|---|---|---|---|
| **Model A (winner)** | 64×64 | 56,350 | **0.000117** | 0.000060 | 0.000032 | 0.000431 |
| Model B | 128×128 | 9,016 | 0.000217 | 0.000117 | 0.000076 | 0.001020 |

Finding: training on randomly-positioned 64×64 sub-crops outperformed training on fixed full-size 128×128 images across every metric, likely due to substantially greater training data diversity (56,350 vs. 9,016 effective examples), despite each individual example containing less whole-image context. Model A is used in the deployed app.

Both models were trained for 20 epochs (Adam, lr=1e-4, batch size 16, CPU) with the same stratified test set (143 images, never seen during training or validation) used for final evaluation.

## Project structure

```
microscope_sr/
├── Data/Raw/<class>/       # your microscope images (not included — see Dataset section)
├── checkpoints/            # Model A (64x64) — weights, logs, plots
├── checkpoints_128/        # Model B (128x128) — weights, logs, plots
├── src/
│   ├── dataset.py           # image loading, blind-spot masking, train/val/test split
│   ├── model.py               # UNet architecture
│   ├── train.py                 # trains Model A (64x64)
│   ├── train_128.py               # trains Model B (128x128)
│   ├── evaluate.py                  # test-set evaluation (mean/std/min/max loss)
│   ├── plot_training.py               # generates loss curve + bar chart from training logs
│   ├── inference.py                     # sliding-window enhancement for full-size images
│   └── app.py                             # Gradio drag-and-drop app
└── requirements.txt
```

## Setup

```bash
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

## Usage

Train:
```bash
cd src
python train.py
```

Evaluate on the test set:
```bash
python evaluate.py
```

Generate loss plots:
```bash
python plot_training.py
```

Run the app:
```bash
python app.py
```
Then open the local URL printed in the terminal (typically `http://127.0.0.1:7860`).

## Results at a glance

Training and validation loss both decreased steadily and plateaued by epoch ~17, with validation loss tracking closely to training loss throughout — indicating the model generalizes well rather than overfitting. See `checkpoints/loss_curve.png` and `checkpoints/val_loss_bar_chart.png` for the full training curves, and `checkpoints/test_sample_*.png` for qualitative before/after examples on held-out test images.

## Limitations

- Trained on CPU; only 20 epochs per model due to compute constraints.
- Dataset drawn from public benchmark images rather than images captured directly from the author's own microscope.
- Single architecture (UNet) evaluated; no comparison against other self-supervised methods (e.g. CARE, Noise2Noise) or supervised baselines.
- Improvement is measurable and statistically consistent, but visually subtle at the current training budget.

## Future work

- Longer training / GPU acceleration for a stronger denoising effect.
- Tuning `mask_ratio` and architecture depth.
- Comparison against CARE and supervised methods, if paired clean/noisy data becomes available.

## Acknowledgements

- Self-supervised masking scheme inspired by Noise2Void (Krull et al., 2019).
- Architecture based on **UNet** (Ronneberger et al., 2015).
- Dataset images sourced from the **MIDOG 2021** and **TUPAC16** challenges.
