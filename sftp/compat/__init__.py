"""
Compatibility layer for loading AllenNLP pretrained models without AllenNLP dependency.

This module provides functionality to load models that were trained with AllenNLP
while only requiring PyTorch and Transformers as dependencies.
"""

from .model_loader import load_archive, SpanFinderModel
from .vocabulary import Vocabulary
from .predictor import SimplePredictor

__all__ = [
    "load_archive",
    "SpanFinderModel",
    "Vocabulary",
    "SimplePredictor",
]
