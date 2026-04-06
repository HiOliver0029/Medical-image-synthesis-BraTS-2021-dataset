from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, ngf: int = 64, n_res_blocks: int = 6) -> None:
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        curr = ngf
        for _ in range(2):
            model.extend(
                [
                    nn.Conv2d(curr, curr * 2, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(curr * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            curr *= 2

        for _ in range(n_res_blocks):
            model.append(ResidualBlock(curr))

        for _ in range(2):
            model.extend(
                [
                    nn.ConvTranspose2d(curr, curr // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(curr // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            curr //= 2

        model.extend(
            [
                nn.ReflectionPad2d(3),
                nn.Conv2d(curr, out_channels, kernel_size=7),
                nn.Tanh(),
            ]
        )

        self.net = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 1, ndf: int = 64) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
