"""GF-6 vegetation-index utilities."""

from gf6vis.indices import (
    DEFAULT_BANDS,
    available_indices,
    calculate_indices,
    calculate_indices_from_bands,
)

__all__ = [
    "DEFAULT_BANDS",
    "available_indices",
    "calculate_indices",
    "calculate_indices_from_bands",
]

__version__ = "0.1.0"
