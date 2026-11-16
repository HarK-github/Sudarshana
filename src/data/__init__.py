"""Data ingestion and dataset abstractions for Sudarshana Indirect Prompt Injection Detection."""

from .dataset import (
    PromptInjectionDataset,
    PromptInjectionRecord,
    generate_mock_dataset,
    raw_dict_to_record,
)

__all__ = [
    "PromptInjectionDataset",
    "PromptInjectionRecord",
    "generate_mock_dataset",
    "raw_dict_to_record",
]

try:
    from .collators import SingleStringCollator, PairInputCollator
    __all__.extend(["SingleStringCollator", "PairInputCollator"])
except ImportError:
    pass
