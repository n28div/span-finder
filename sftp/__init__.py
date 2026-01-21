from .data_reader import (
    BetterDatasetReader, SRLDatasetReader
)
from .metrics import SRLMetric, BaseF, ExactMatch, FBetaMixMeasure
from .models import SpanModel
from .modules import (
    MLPSpanTyping, SpanTyping, SpanFinder, BIOSpanFinder
)
from .predictor import SpanPredictor
from .utils import Span


def load_model(
    model_path: str = "https://public.gqin.me/framenet/20210127.fn.tar.gz",
    device: str = "cpu",
) -> SpanPredictor:
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
        SpanPredictor: A predictor instance ready for inference.

    Example:
        >>> from sftp import load_model
        >>> predictor = load_model(device="cpu")
        >>> result = predictor.predict_sentence("Bob saw Alice eating an apple.")
        >>> result.span.tree(result.sentence)
    """
    # Convert device string to cuda_device int for AllenNLP compatibility
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
    # Main API
    "load_model",
    "SpanPredictor",
    "Span",
    # Model components
    "SpanModel",
    "MLPSpanTyping",
    "SpanTyping",
    "SpanFinder",
    "BIOSpanFinder",
    # Data readers
    "BetterDatasetReader",
    "SRLDatasetReader",
    # Metrics
    "SRLMetric",
    "BaseF",
    "ExactMatch",
    "FBetaMixMeasure",
]
