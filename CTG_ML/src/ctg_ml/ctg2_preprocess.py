from __future__ import annotations

from ctg_ml import multimodal_preprocess as _preprocess

CTG2SplitBuildStats = _preprocess.MultimodalSplitBuildStats
sequence_channel_names = _preprocess.sequence_channel_names

__all__ = [
    "CTG2SplitBuildStats",
    "build_ctg2_multimodal_npz_files",
    "sequence_channel_names",
]


def build_ctg2_multimodal_npz_files(*args, **kwargs):
    """Legacy CTG2 preprocessing wrapper kept for old scripts."""
    return _preprocess.build_multimodal_npz_files(*args, **kwargs)
