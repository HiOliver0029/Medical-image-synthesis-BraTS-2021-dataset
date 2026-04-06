from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dlmi_hw1.data.datasets import PairedSliceDataset
from dlmi_hw1.eval.metrics import compute_psnr_ssim
from dlmi_hw1.models.cyclegan import Generator


def evaluate(config_path: Path, checkpoint_path: Path) -> None:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PairedSliceDataset(Path(data_cfg["processed_root"]), split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    gen_cfg = {
        "in_channels": model_cfg["in_channels"],
        "out_channels": model_cfg["out_channels"],
        "ngf": model_cfg["ngf"],
        "n_res_blocks": model_cfg["n_res_blocks"],
    }
    g_ab = Generator(**gen_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    g_ab.load_state_dict(ckpt["g_ab_state"])
    g_ab.eval()

    all_psnr = []
    all_ssim = []

    with torch.no_grad():
        for real_a, real_b in tqdm(loader, desc="Evaluate"):
            real_a = real_a.to(device)
            pred_b = g_ab(real_a).cpu().numpy()[0, 0]
            tgt_b = real_b.numpy()[0, 0]

            psnr, ssim = compute_psnr_ssim(pred_b, tgt_b)
            all_psnr.append(psnr)
            all_ssim.append(ssim)

    print(f"Validation slices: {len(all_psnr)}")
    print(f"PSNR: {np.mean(all_psnr):.4f} +/- {np.std(all_psnr):.4f}")
    print(f"SSIM: {np.mean(all_ssim):.4f} +/- {np.std(all_ssim):.4f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate CycleGAN using PSNR and SSIM")
    parser.add_argument("--config", type=Path, default=Path("configs/cyclegan_brats.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    evaluate(config_path=args.config, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
