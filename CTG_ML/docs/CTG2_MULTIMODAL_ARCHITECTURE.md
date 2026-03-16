# CTG2 Multimodal Architecture

This file describes the current CTG2 model from `configs/ctg2_multimodal.toml`, `src/ctg_ml/models.py`, and the CTG2 preprocessing scripts.

## Current Config Snapshot

- CTG sequence length: `3600` timesteps (`60 min * 60 s * 1 Hz`)
- CTG channels: `6`
  - `FHR`
  - `toco`
  - `Hr1_SignalQuality==G`
  - `Hr1_SignalQuality==Y`
  - `Hr1_SignalQuality==R`
  - `padding_mask`
- Registry/tabular input width: `72` features after encoding
- TCN channels: `[32, 32, 64, 64, 64, 128, 128, 128]`
- Kernel size: `7`
- Dropout: `0.1`
- Tabular hidden size: `64`
- Fusion hidden size: `128`
- Regression outputs: `3`
  - `apgar1`
  - `apgar5`
  - `apgar10`
- Binary outputs: `4`
  - `shoulder_dystocia`
  - `treatment_for_hypoglycemia`
  - `neonatal_sepsis_or_pneumonia`
  - `respiratorbehandling`

## Compact Flow

```mermaid
flowchart LR
    A["CTG sequence<br/>(B, 6, 3600)"] --> B0["TCN Block 0<br/>6 -> 32<br/>dilation=1"]
    B0 --> B1["Block 1<br/>32 -> 32<br/>dilation=2"]
    B1 --> B2["Block 2<br/>32 -> 64<br/>dilation=4"]
    B2 --> B3["Block 3<br/>64 -> 64<br/>dilation=8"]
    B3 --> B4["Block 4<br/>64 -> 64<br/>dilation=16"]
    B4 --> B5["Block 5<br/>64 -> 128<br/>dilation=32"]
    B5 --> B6["Block 6<br/>128 -> 128<br/>dilation=64"]
    B6 --> B7["Block 7<br/>128 -> 128<br/>dilation=128"]
    B7 --> P["AdaptiveAvgPool1d(1)<br/>(B, 128, 1)"]
    P --> F["Flatten<br/>(B, 128)"]

    T["Registry inputs<br/>(B, 72)"] --> TE["Tabular MLP<br/>Linear 72 -> 64<br/>ReLU + Dropout"]

    F --> C["Concatenate<br/>(B, 128 + 64 = 192)"]
    TE --> C

    C --> FU["Fusion MLP<br/>Linear 192 -> 128<br/>ReLU + Dropout"]
    FU --> RH["Regression head<br/>Linear 128 -> 3"]
    FU --> BH["Binary head<br/>Linear 128 -> 4"]

    RH --> RO["Regression outputs<br/>apgar1, apgar5, apgar10"]
    BH --> BO["Binary logits<br/>4 neonatal outcomes"]
```

## Block Structure

Each TCN block has the same internal structure:

```mermaid
flowchart TD
    X["Input x<br/>(B, Cin, 3600)"] --> C1["Conv1d #1<br/>Cin -> Cout<br/>kernel=7, dilation=d"]
    C1 --> CH1["Chomp1d<br/>keep time length = 3600"]
    CH1 --> R1["ReLU"]
    R1 --> D1["Dropout(0.1)"]
    D1 --> C2["Conv1d #2<br/>Cout -> Cout<br/>kernel=7, dilation=d"]
    C2 --> CH2["Chomp1d"]
    CH2 --> R2["ReLU"]
    R2 --> D2["Dropout(0.1)"]

    X --> RES["Residual path<br/>Identity or 1x1 Conv"]
    D2 --> ADD["Add main path + residual"]
    RES --> ADD
    ADD --> OUT["ReLU<br/>Output (B, Cout, 3600)"]
```

## Exact Shapes

| Stage | Shape |
|---|---|
| CTG input | `(B, 6, 3600)` |
| After Block 0 | `(B, 32, 3600)` |
| After Block 1 | `(B, 32, 3600)` |
| After Block 2 | `(B, 64, 3600)` |
| After Block 3 | `(B, 64, 3600)` |
| After Block 4 | `(B, 64, 3600)` |
| After Block 5 | `(B, 128, 3600)` |
| After Block 6 | `(B, 128, 3600)` |
| After Block 7 | `(B, 128, 3600)` |
| Pooled CTG embedding | `(B, 128)` |
| Tabular input | `(B, 72)` |
| Tabular embedding | `(B, 64)` |
| Concatenated fusion vector | `(B, 192)` |
| Fusion hidden vector | `(B, 128)` |
| Regression outputs | `(B, 3)` |
| Binary logits | `(B, 4)` |

## What Each Part Does

### CTG branch

- Reads the full 60-minute CTG trace.
- Uses dilated Conv1D residual blocks to detect patterns at increasing time scales.
- Pools over time to create one fixed-length CTG summary vector per pregnancy.

### Tabular branch

- Takes encoded registry inputs after preprocessing.
- Numeric fields are median-imputed and standardized, with missing flags.
- Boolean fields become `0/1` plus missing flags.
- Categorical fields become one-hot features plus missing and `other` buckets.
- A single dense layer maps the 72 registry features into a 64-dimensional embedding.

### Fusion stage

- Concatenates the CTG embedding and registry embedding.
- Learns interactions between the two modalities in a dense hidden layer.

### Output heads

- Regression head predicts exact Apgar scores.
- Binary head predicts 4 separate neonatal outcomes in parallel.
- Training uses masked losses so missing output labels do not contribute to loss.

## Losses

- Regression: masked smooth L1 loss on `apgar1`, `apgar5`, `apgar10`
- Binary: masked binary cross-entropy with per-output `pos_weight`
- Total loss:

```text
total_loss = regression_weight * regression_loss + binary_weight * binary_loss
```

Current weights from config:

- `regression_weight = 1.0`
- `binary_weight = 1.0`

## Important Project-Specific Choices

These are not required by TCNs in general. They are current choices for this project:

- registry inputs are fused **after** temporal pooling, not before
- one shared TCN encoder is used for all tasks
- one shared tabular encoder is used for all tasks
- all outputs share the same fusion representation
- Apgar scores are treated as regression targets, not ordinal or binary targets
- outputs with missing labels are masked rather than forced to negative
- `Hr1_SignalQuality` is encoded as one-hot channels instead of a learned embedding

## Minimal Mental Model

```text
CTG over 60 min -> TCN encoder -> CTG summary
Registry row -> tabular encoder -> registry summary
CTG summary + registry summary -> fusion layer
fusion layer -> regression head + binary head
```
