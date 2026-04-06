# DLMI HW1 Report Template

## 1. Introduction
- Why medical image synthesis matters (data scarcity, modality transfer, privacy).
- Task definition: source modality and target modality.

## 2. Dataset
- BraTS 2021 summary.
- Four MRI modalities (`T1`, `T1ce`, `T2`, `FLAIR`).
- Number of patients and splits.

## 3. Preprocessing
- NIfTI loading and shape handling.
- Intensity normalization method.
- 2D slicing strategy and axis.
- Background filtering threshold.

## 4. Methodology
- CycleGAN architecture (Generator/Discriminator).
- Losses:
  - Adversarial loss
  - Cycle consistency loss
  - Identity loss
- Training configuration (batch size, epochs, AMP).

## 5. Experiments
- Hardware settings.
- Hyperparameters and ablations.
- Training curves or representative epochs.

## 6. Qualitative Results
- Real source vs generated target vs real target.
- Failure cases (hallucination or missing structure).

## 7. Quantitative Results
- PSNR table.
- SSIM table.
- Optional FID and downstream segmentation Dice.

## 8. Conclusion and Future Work
- Main findings.
- Limitations and future direction (LDM, MONAI diffusion, clinical validation).
