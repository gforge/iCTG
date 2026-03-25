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


class TCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int] | tuple[int, ...],
        kernel_size: int,
        dropout: float,
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
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.tcn(x))


class MultimodalMultitaskTCN(nn.Module):
    def __init__(
        self,
        sequence_in_channels: int,
        tabular_in_features: int,
        tcn_channels: list[int] | tuple[int, ...],
        kernel_size: int,
        dropout: float,
        tabular_hidden_dim: int,
        fusion_hidden_dim: int,
        num_apgar_outputs: int,
        categorical_output_dims: list[int] | tuple[int, ...],
        num_regression_outputs: int,
        num_binary_outputs: int,
    ) -> None:
        super().__init__()
        self.sequence_encoder = TCNEncoder(
            in_channels=sequence_in_channels,
            channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_in_features, tabular_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        fusion_in = self.sequence_encoder.out_dim + tabular_hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.apgar_head = nn.Linear(fusion_hidden_dim, num_apgar_outputs * 11)
        self.categorical_heads = nn.ModuleList([nn.Linear(fusion_hidden_dim, int(dim)) for dim in categorical_output_dims])
        self.regression_head = nn.Linear(fusion_hidden_dim, num_regression_outputs)
        self.binary_head = nn.Linear(fusion_hidden_dim, num_binary_outputs)
        self.num_apgar_outputs = num_apgar_outputs

    def forward(
        self, x_seq: torch.Tensor, x_tab: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        seq_embed = self.sequence_encoder(x_seq)
        tab_embed = self.tabular_encoder(x_tab)
        fused = self.fusion(torch.cat([seq_embed, tab_embed], dim=1))
        apgar_logits = self.apgar_head(fused).view(-1, self.num_apgar_outputs, 11)
        categorical_logits = [head(fused) for head in self.categorical_heads]
        return apgar_logits, categorical_logits, self.regression_head(fused), self.binary_head(fused)
