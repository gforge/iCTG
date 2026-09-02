from __future__ import annotations

import numpy as np
import pandas as pd

from ctg_ml.multimodal_config import MultimodalRegistryConfig
from ctg_ml.multimodal_registry import fit_tabular_encoder, transform_tabular_inputs

REGISTRY_CFG = MultimodalRegistryConfig(
    input_numeric=["maternal_age"],
    input_boolean=["is_smoker"],
    input_categorical=["fodelseland"],
    input_excluded_due_to_leakage=[],
    categorical_other_min_frequency=1,
    country_top_k=2,
    apgar_outputs=[],
    categorical_outputs=[],
    continuous_outputs=[],
    binary_outputs=[],
    binary_outputs_missing_as_false=[],
)


def _train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "maternal_age": [20.0, 30.0, 40.0, np.nan],
            "is_smoker": [True, False, True, None],
            "fodelseland": ["Sweden", "Sweden", "Norway", None],
        }
    )


def _val_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "maternal_age": [np.nan, 50.0],
            "is_smoker": [None, False],
            "fodelseland": ["Finland", None],
        }
    )


def test_encoder_is_fit_on_train_only() -> None:
    encoder = fit_tabular_encoder(_train_df(), REGISTRY_CFG)

    assert encoder.numeric_medians == {"maternal_age": 30.0}
    assert encoder.categorical_levels == {"fodelseland": ["Norway", "Sweden"]}
    assert encoder.feature_names == [
        "maternal_age",
        "maternal_age__missing",
        "is_smoker",
        "is_smoker__missing",
        "fodelseland__missing",
        "fodelseland==Norway",
        "fodelseland==Sweden",
        "fodelseland__other",
    ]


def test_transform_applies_train_statistics_to_val_and_flags_missing() -> None:
    encoder = fit_tabular_encoder(_train_df(), REGISTRY_CFG)

    X_train = transform_tabular_inputs(_train_df(), encoder)
    X_val = transform_tabular_inputs(_val_df(), encoder)

    assert X_train.shape == (4, len(encoder.feature_names))
    assert X_val.shape == (2, len(encoder.feature_names))
    assert X_val.dtype == np.float32

    # Train row with missing age/smoker/country: median fill + missing flags set.
    np.testing.assert_array_equal(X_train[3], [30.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(X_train[0], [20.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    # Val is filled with the *train* median (30), not the val median (50), and an unseen
    # country lands in the __other bucket.
    np.testing.assert_array_equal(X_val[0], [30.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(X_val[1], [50.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # Transforming val must not refit the encoder.
    assert encoder.numeric_medians == {"maternal_age": 30.0}
    assert encoder.categorical_levels == {"fodelseland": ["Norway", "Sweden"]}
