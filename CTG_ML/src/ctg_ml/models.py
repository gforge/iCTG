from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


class TCNBinaryClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: list[int] | tuple[int, ...] = (32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        for i, ch in enumerate(channels):
            layers.append(
                TemporalBlock(
                    in_channels=prev,
                    out_channels=ch,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            prev = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(prev, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels=2, time)
        return self.head(self.tcn(x)).squeeze(-1)
