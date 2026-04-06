# DLMI HW1 Report

## 1. Introduction
Medical image synthesis plays a critical role in addressing data scarcity, reducing acquisition costs, and handling patient privacy in multi-modal studies. In this assignment, we focus on translating brain MRI modalities—specifically converting T1-weighted images to T2-weighted images—a fundamental task that demonstrates how structural information from one sequence can be mapped to contrast features of another.

## 2. Dataset
We utilized the BraTS 2021 dataset (Task 1), which features perfectly co-registered multi-modal MRI scans for brain tumor patients. The dataset provides four modalities: `T1`, `T1ce`, `T2`, and `FLAIR`. For this baseline, we selected `T1` as the source modality and `T2` as the target modality. The dataset was split into training and validation sets at an 80:20 ratio.

## 3. Preprocessing
To prepare the 3D NIfTI volumes for 2D image synthesis:
- **Axis Slicing**: Slices were extracted along the axial plane (axis 2).
- **Foreground Filtering**: Slices lacking sufficient brain matter were filtered out by enforcing a minimum foreground area ratio (10%).
- **Normalization**: Each volume was Z-score normalized, then scaled strictly into the `[-1, 1]` range compatible with the generator's `tanh` output layer.

## 4. Methodology
As a baseline, we implemented the classic CycleGAN model. This architecture relies on two Generators (ResNet-based) and two PatchGAN Discriminators.
- **Adversarial Loss**: To ensure the generated slices look indistinguishable from real T2/T1 slices.
- **Cycle Consistency Loss** (L1): To enforce that translating `T1 -> T2 -> T1` brings us back to the original image, preventing mode collapse.
- **Identity Loss** (L1): To encourage color preservation when a generator is fed an image already in the target domain.
- **Optimization**: We compiled the model using mixed precision (`torch.amp`) and Adam optimizers. Due to time constraints, the baseline was trained for a shortened "Fast Mode" spanning 20 epochs (with 600 batches per epoch).

## 5. Experiments
- **Hardware**: Ubuntu 24.04 `aarch64` system equipped with an NVIDIA A100 GPU.
- **Setup**: PyTorch 2.5.1+cu124. Batch size of 8. Learning rate of `2e-4`.

## 6. Qualitative Results
Because the baseline evaluation only calculates aggregate PSNR and SSIM over the validation set, qualitative visual examples (the translation from `T1` to synthetic `T2`) are automatically plotted using custom sampling logic.

![Sample Qualitative Results](qualitative_results/sample_results.png)

*Figure 1: (Left) Real T1 source image. (Middle) CycleGAN Generated T2 image. (Right) Real T2 Target image for ground-truth comparison.*

The typical synthetic images successfully resemble T2-weighted MRI scans—with bright cerebrospinal fluid (CSF)—though with some expected loss in high-frequency detail due to the short "Fast Mode" baseline training cycle.

## 7. Quantitative Results
The baseline model evaluated on 25,379 validation slices yielded the following metrics:
- **PSNR**: 21.1716 ± 2.5342
- **SSIM**: 0.7586 ± 0.0873

### Performance Analysis
For a baseline CycleGAN trained only for a fraction of a full convergence schedule (20 truncated epochs instead of 100+ full epochs), an SSIM of nearly 0.76 is a very respectable starting point. It proves the generator is correctly learning the macro-structures (skull, ventricles) and general intensity mappings. However, state-of-the-art medical synthesis on perfectly paired data typically reaches PSNR > 25 and SSIM > 0.85-0.90. This gap exists largely due to the limits of our baseline configuration.

## 8. Conclusion and Future Work (Optimizations)

To push the performance limits further, the following optimizations should be implemented:

1. **Leveraging Paired Data (Pix2Pix / LDM)**: 
   BraTS features perfectly co-registered T1 and T2 images. CycleGAN is designed for *unpaired* datasets and inherently throws away pixel-to-pixel spatial guarantees. Switching to a **Pix2Pix** architecture or a **Latent Diffusion Model (LDM)** (e.g., using MONAI Generative Models as suggested in the assignment) would allow us to compute direct direct L1/L2 pixel loss between the generated and target masks, massively boosting PSNR and SSIM.
2. **Extended Training Durations**:
   The current metrics were achieved under severe budget constraints (20 epochs capped at 600 batches). Training out to 100-200 full epochs is required for the ResNet generator to stabilize and resolve fine granular details such as complex brain folds.
3. **Advanced Loss Functions**:
   Solely relying on L1 cycle-consistency forces the network to blur high-frequency details. Adding a **Perceptual Loss (VGG16 loss)** or a structural **SSIM Loss** directly to the generator's objective function would sharply improve the final qualitative textures and edge sharpness.
4. **Attention Mechanisms**:
   Upgrading the backbone from standard ResNet blocks to standard Vision Transformers (ViT) or Swin Transformers would allow the model to learn long-range global context, which is especially useful for capturing large anatomical geometries like the skull boundary to ventricle distances.