from __future__ import annotations

from ctg_ml import multimodal_registry as _registry

MultitaskTargetSpec = _registry.MultitaskTargetSpec
TabularEncoder = _registry.TabularEncoder
build_targets = _registry.build_targets
fit_tabular_encoder = _registry.fit_tabular_encoder
load_registry_for_multimodal = _registry.load_registry_for_multimodal
merge_splits_with_registry = _registry.merge_splits_with_registry
normalize_tabular_inplace = _registry.normalize_tabular_inplace
transform_tabular_inputs = _registry.transform_tabular_inputs

__all__ = [
    "MultitaskTargetSpec",
    "TabularEncoder",
    "build_targets",
    "fit_tabular_encoder",
    "load_registry_for_multimodal",
    "load_registry_labels_v2",
    "merge_splits_with_registry",
    "normalize_tabular_inplace",
    "transform_tabular_inputs",
]


def load_registry_labels_v2(registry_csv: str, at_risk_max_apgar: int = 6):
    """Legacy CTG2 label loader kept for old scripts."""
    return _registry.load_registry_labels_multimodal(registry_csv, at_risk_max_apgar)
