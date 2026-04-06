# DLMI HW1 - Medical Image Synthesis (BraTS 2021)

This project implements a complete baseline pipeline for DLMI HW1:

1. Download BraTS 2021 Task 1 from KaggleHub.
2. Preprocess NIfTI 3D volumes into filtered paired 2D slices.
3. Train a CycleGAN baseline for cross-modality synthesis.
4. Evaluate generated images using PSNR and SSIM.

The default synthesis task is `T1 -> T2`.

## 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

If you prefer requirements file:

```bash
pip install -r requirements.txt
```

## 2) Download dataset (exact command requested)

You can run this exact Python code:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("dschettler8845/brats-2021-task1")

print("Path to dataset files:", path)
```

Or use the provided script:

```bash
python scripts/download_dataset.py
```

By default it creates a symlink at `data/raw/brats_2021_task1` to the downloaded cache path.

Quick sanity check before full preprocessing:

```bash
python scripts/preprocess_brats.py \
  --raw-root data/raw \
  --source-modality t1 \
  --target-modality t2 \
  --sanity-check
```

## 3) Preprocess BraTS to 2D paired slices

```bash
python scripts/preprocess_brats.py \
  --raw-root data/raw \
  --processed-root data/processed \
  --source-modality t1 \
  --target-modality t2 \
  --split-ratio 0.8 \
  --min-foreground-ratio 0.1 \
  --axis 2
```

What this does:

- Reads `.nii.gz` files via `nibabel`.
- If only BraTS `.tar` archives are present, it auto-extracts them into `data/raw/extracted` first.
- Normalizes each volume using Z-score followed by Min-Max to `[-1, 1]`.
- Slices 3D volumes into 2D images.
- Removes mostly black slices (foreground ratio threshold).
- Stores paired slices in:
  - `data/processed/train/A`, `data/processed/train/B`
  - `data/processed/val/A`, `data/processed/val/B`

## 4) Train CycleGAN baseline

```bash
python scripts/train_cyclegan.py --config configs/cyclegan_brats.yaml --output-dir checkpoints
```

The training objective is:

$$
\mathcal{L}_{total} = \mathcal{L}_{GAN} + \lambda_{cyc}\mathcal{L}_{cycle} + \lambda_{id}\mathcal{L}_{identity}
$$

## 5) Evaluate (PSNR/SSIM)

```bash
python scripts/evaluate_cyclegan.py \
  --config configs/cyclegan_brats.yaml \
  --checkpoint checkpoints/cyclegan_epoch_100.pt
```

## Suggested report structure

Follow the HW template in `docs/report_template.md`:

- Introduction
- Dataset and preprocessing
- Methodology
- Experiments
- Qualitative results
- Quantitative results (PSNR/SSIM)
- Conclusion and future work

## Notes aligned with HW instruction

- Dataset: BraTS 2021 multi-modal MRI.
- Baseline model: CycleGAN (implemented).
- Recommended extension: Latent Diffusion + MONAI Generative Models.
- Clinical consistency extension: evaluate downstream segmentation Dice on synthesized images.

## 6) Specific Notes for AArch64 (ARM) with A100 GPU
If you are running this code on an `aarch64` system equipped with NVIDIA GPUs (such as an A100 instance), pip default indices will fetch CPU-only Torch wheels. You must install the `cu124` index versions manually stringing the wheels:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```
You can verify it works by running:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 7) 1-Click Pipeline Execution
For a fast, hands-off execution generating a quick `eval.log` report for your HW1, use the enclosed shell script which automatically runs 20 shortened training epochs and follows up with evaluation.
```bash
./run_pipeline.sh
```
