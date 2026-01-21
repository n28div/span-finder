"""
SpanFinder: Parse sentences by finding & labeling spans.

This package provides tools for span-based semantic parsing, including
frame semantic parsing and semantic role labeling.

Basic usage:
    >>> from sftp import load_model
    >>> predictor = load_model(device="cpu")
    >>> result = predictor.predict_sentence("Bob saw Alice eating an apple.")
    >>> result.span.tree(result.sentence)
"""

__version__ = "1.0.0"

# Core utilities (always available)
from .utils import Span

# Import compatibility layer for inference (doesn't require AllenNLP)
from .compat import SimplePredictor, load_archive

# Try to import AllenNLP-based components for training
# These are optional and only needed for training new models
_ALLENNLP_AVAILABLE = False
try:
    from .data_reader import BetterDatasetReader, SRLDatasetReader
    from .metrics import SRLMetric, BaseF, ExactMatch, FBetaMixMeasure
    from .models import SpanModel
    from .modules import MLPSpanTyping, SpanTyping, SpanFinder, BIOSpanFinder
    from .predictor import SpanPredictor
    _ALLENNLP_AVAILABLE = True
except ImportError:
    # AllenNLP not available - inference-only mode
    SpanPredictor = None
    SpanModel = None
    BetterDatasetReader = None
    SRLDatasetReader = None


def load_model(
    model_path: str = "https://public.gqin.me/framenet/20210127.fn.tar.gz",
    device: str = "cpu",
) -> SimplePredictor:
    """
    Load a SpanFinder model for inference.

    This is the simplest way to get started with SpanFinder. It returns a
    predictor that can be used to predict spans on sentences.

    Args:
        model_path: Path to the model checkpoint. Can be:
            - A URL to download the model from
            - A local path to a .tar.gz archive
            - A local path to an extracted model directory
            Default: FrameNet 1.7 pretrained model
        device: Device to run the model on. Options:
            - "cpu": Run on CPU
            - "cuda": Run on GPU (requires CUDA)
            - "cuda:0", "cuda:1", etc.: Run on specific GPU
            Default: "cpu"

    Returns:
        SimplePredictor: A predictor instance ready for inference.

    Example:
        >>> from sftp import load_model
        >>> predictor = load_model(device="cpu")
        >>> result = predictor.predict_sentence("Bob saw Alice eating an apple.")
        >>> result.span.tree(result.sentence)
    """
    return SimplePredictor.from_path(model_path, device=device)


# For backwards compatibility, also expose load_model_allennlp for those who have AllenNLP
def load_model_allennlp(
    model_path: str = "https://public.gqin.me/framenet/20210127.fn.tar.gz",
    device: str = "cpu",
):
    """
    Load model using AllenNLP (requires allennlp to be installed).

    This is the legacy loading method. Use `load_model()` instead for
    inference without AllenNLP dependency.
    """
    if not _ALLENNLP_AVAILABLE:
        raise ImportError(
            "AllenNLP is not installed. Use `load_model()` for inference "
            "without AllenNLP, or install AllenNLP for training:\n"
            "  pip install allennlp allennlp-models"
        )

    if device == "cpu":
        cuda_device = -1
    elif device == "cuda":
        cuda_device = 0
    elif device.startswith("cuda:"):
        cuda_device = int(device.split(":")[1])
    else:
        raise ValueError(f"Unknown device: {device}. Use 'cpu', 'cuda', or 'cuda:N'")

    return SpanPredictor.from_path(model_path, cuda_device=cuda_device)


__all__ = [
    # Version
    "__version__",
    # Main API (inference)
    "load_model",
    "SimplePredictor",
    "Span",
    # Compatibility
    "load_archive",
    # Legacy AllenNLP API (if available)
    "load_model_allennlp",
    "_ALLENNLP_AVAILABLE",
]

# Add AllenNLP components to __all__ if available
if _ALLENNLP_AVAILABLE:
    __all__.extend([
        "SpanPredictor",
        "SpanModel",
        "MLPSpanTyping",
        "SpanTyping",
        "SpanFinder",
        "BIOSpanFinder",
        "BetterDatasetReader",
        "SRLDatasetReader",
        "SRLMetric",
        "BaseF",
        "ExactMatch",
        "FBetaMixMeasure",
    ])
