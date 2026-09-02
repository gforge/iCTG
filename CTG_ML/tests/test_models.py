from __future__ import annotations

import torch

from ctg_ml.models import MultimodalMultitaskTCN


def test_multimodal_multitask_tcn_output_shapes() -> None:
    torch.manual_seed(0)
    batch, seq_channels, n_steps, tab_features = 4, 3, 32, 5
    num_apgar, categorical_dims, num_regression, num_binary = 2, [3, 4], 2, 3

    model = MultimodalMultitaskTCN(
        sequence_in_channels=seq_channels,
        tabular_in_features=tab_features,
        tcn_channels=[4, 4],
        kernel_size=3,
        dropout=0.0,
        tabular_hidden_dim=6,
        fusion_hidden_dim=8,
        num_apgar_outputs=num_apgar,
        categorical_output_dims=categorical_dims,
        num_regression_outputs=num_regression,
        num_binary_outputs=num_binary,
    )
    model.eval()

    x_seq = torch.randn(batch, seq_channels, n_steps)
    x_tab = torch.randn(batch, tab_features)
    with torch.no_grad():
        apgar_logits, categorical_logits, regression_out, binary_logits = model(x_seq, x_tab)

    assert apgar_logits.shape == (batch, num_apgar, 11)
    assert [t.shape for t in categorical_logits] == [(batch, dim) for dim in categorical_dims]
    assert regression_out.shape == (batch, num_regression)
    assert binary_logits.shape == (batch, num_binary)
    assert torch.isfinite(apgar_logits).all()
    assert torch.isfinite(binary_logits).all()
