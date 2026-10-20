"""Validation package for AgentEvalOps datasets."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code.validation.dataset_validator import (
        ALL_CHECKS,
        DatasetSummary,
        Finding,
        Severity,
        ValidationReport,
        validate_dataset,
        validate_datasets,
    )

__all__ = [
    "ALL_CHECKS",
    "DatasetSummary",
    "Finding",
    "Severity",
    "ValidationReport",
    "validate_dataset",
    "validate_datasets",
]


def __getattr__(name: str):
    if name in __all__:
        import code.validation.dataset_validator as _dv

        return getattr(_dv, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
