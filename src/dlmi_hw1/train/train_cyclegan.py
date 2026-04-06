from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import yaml
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from dlmi_hw1.data.datasets import PairedSliceDataset
from dlmi_hw1.models.cyclegan import Generator, PatchDiscriminator
from dlmi_hw1.models.losses import GANLoss
from dlmi_hw1.seed import set_seed


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(out_dir: Path, epoch: int, models: Dict[str, nn.Module], optimizers: Dict[str, Adam]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        **{f"{name}_state": model.state_dict() for name, model in models.items()},
        **{f"opt_{name}_state": opt.state_dict() for name, opt in optimizers.items()},
    }
    torch.save(ckpt, out_dir / f"cyclegan_epoch_{epoch:03d}.pt")


def train(config_path: Path, output_dir: Path) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    gen_cfg = {
        "in_channels": model_cfg["in_channels"],
        "out_channels": model_cfg["out_channels"],
        "ngf": model_cfg["ngf"],
        "n_res_blocks": model_cfg["n_res_blocks"],
    }
    disc_cfg = {
        "in_channels": model_cfg["in_channels"],
        "ndf": model_cfg["ndf"],
    }

    dataset = PairedSliceDataset(Path(data_cfg["processed_root"]), split="train")
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    g_ab = Generator(**gen_cfg).to(device)
    g_ba = Generator(**gen_cfg).to(device)
    d_a = PatchDiscriminator(**disc_cfg).to(device)
    d_b = PatchDiscriminator(**disc_cfg).to(device)

    criterion_gan = GANLoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    opt_g = Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=train_cfg["lr"], betas=(0.5, 0.999))
    opt_d_a = Adam(d_a.parameters(), lr=train_cfg["lr"], betas=(0.5, 0.999))
    opt_d_b = Adam(d_b.parameters(), lr=train_cfg["lr"], betas=(0.5, 0.999))

    scaler = GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and torch.cuda.is_available())

    for epoch in range(1, train_cfg["epochs"] + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{train_cfg['epochs']}", total=train_cfg.get("max_batches_per_epoch", len(loader)))
        for i, (real_a, real_b) in enumerate(pbar):
            if i >= train_cfg.get("max_batches_per_epoch", len(loader)):
                break
            real_a = real_a.to(device)
            real_b = real_b.to(device)

            opt_g.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=scaler.is_enabled()):
                same_b = g_ab(real_b)
                same_a = g_ba(real_a)
                loss_idt = (
                    criterion_identity(same_b, real_b) + criterion_identity(same_a, real_a)
                ) * train_cfg["lambda_identity"]

                fake_b = g_ab(real_a)
                fake_a = g_ba(real_b)

                loss_gan = criterion_gan(d_b(fake_b), True) + criterion_gan(d_a(fake_a), True)

                rec_a = g_ba(fake_b)
                rec_b = g_ab(fake_a)
                loss_cycle = (
                    criterion_cycle(rec_a, real_a) + criterion_cycle(rec_b, real_b)
                ) * train_cfg["lambda_cycle"]

                loss_g = loss_gan + loss_cycle + loss_idt

            scaler.scale(loss_g).backward()
            scaler.step(opt_g)

            opt_d_a.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=scaler.is_enabled()):
                loss_d_a = 0.5 * (
                    criterion_gan(d_a(real_a), True)
                    + criterion_gan(d_a(fake_a.detach()), False)
                )
            scaler.scale(loss_d_a).backward()
            scaler.step(opt_d_a)

            opt_d_b.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=scaler.is_enabled()):
                loss_d_b = 0.5 * (
                    criterion_gan(d_b(real_b), True)
                    + criterion_gan(d_b(fake_b.detach()), False)
                )
            scaler.scale(loss_d_b).backward()
            scaler.step(opt_d_b)

            scaler.update()

            pbar.set_postfix(
                loss_g=float(loss_g.detach().cpu()),
                loss_d_a=float(loss_d_a.detach().cpu()),
                loss_d_b=float(loss_d_b.detach().cpu()),
            )

        if epoch % train_cfg["save_every"] == 0 or epoch == train_cfg["epochs"]:
            save_checkpoint(
                output_dir,
                epoch,
                models={"g_ab": g_ab, "g_ba": g_ba, "d_a": d_a, "d_b": d_b},
                optimizers={"g": opt_g, "d_a": opt_d_a, "d_b": opt_d_b},
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train CycleGAN on BraTS 2D slices")
    parser.add_argument("--config", type=Path, default=Path("configs/cyclegan_brats.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    train(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
