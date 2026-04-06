from __future__ import annotations

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)
