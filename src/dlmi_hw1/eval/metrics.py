from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def denorm_to_unit(x: np.ndarray) -> np.ndarray:
    return ((x + 1.0) / 2.0).clip(0.0, 1.0)


def compute_psnr_ssim(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pred_u = denorm_to_unit(pred)
    tgt_u = denorm_to_unit(target)

    psnr = float(peak_signal_noise_ratio(tgt_u, pred_u, data_range=1.0))
    ssim = float(structural_similarity(tgt_u, pred_u, data_range=1.0))
    return psnr, ssim
