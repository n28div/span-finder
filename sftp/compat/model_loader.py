"""
Model loader for AllenNLP-trained SpanFinder models.

This module can load pretrained models from AllenNLP archives without
requiring AllenNLP as a dependency.
"""

import json
import os
import shutil
import tarfile
import tempfile
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .vocabulary import Vocabulary, DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN


def download_file(url: str, destination: str) -> str:
    """Download a file from URL to destination."""
    import urllib.request
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, destination)
    return destination


def load_archive(
    archive_path: str,
    device: str = "cpu",
    weights_file: Optional[str] = None,
) -> Tuple["SpanFinderModel", Vocabulary, Dict]:
    """
    Load a SpanFinder model from an AllenNLP archive.

    Args:
        archive_path: Path to .tar.gz archive, extracted directory, or URL.
        device: Device to load model on ('cpu', 'cuda', 'cuda:0', etc.).
        weights_file: Optional specific weights file to load.

    Returns:
        Tuple of (model, vocabulary, config).
    """
    # Handle URLs
    if archive_path.startswith("http://") or archive_path.startswith("https://"):
        temp_dir = tempfile.mkdtemp()
        archive_file = os.path.join(temp_dir, "model.tar.gz")
        download_file(archive_path, archive_file)
        archive_path = archive_file

    # Extract if it's a tar.gz file
    if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        extract_dir = tempfile.mkdtemp()
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        archive_path = extract_dir

    # Load config
    config_path = os.path.join(archive_path, "config.json")
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)

    # Load vocabulary
    vocab_dir = os.path.join(archive_path, "vocabulary")
    vocabulary = Vocabulary.from_files(vocab_dir)

    # Load weights
    if weights_file is None:
        weights_file = os.path.join(archive_path, "weights.th")

    # Determine torch device
    if device == "cpu":
        torch_device = torch.device("cpu")
    elif device == "cuda":
        torch_device = torch.device("cuda")
    elif device.startswith("cuda:"):
        torch_device = torch.device(device)
    else:
        torch_device = torch.device(device)

    weights = torch.load(weights_file, map_location=torch_device, weights_only=False)

    # Build model from config and load weights
    model = SpanFinderModel.from_config(config, vocabulary)
    model.load_state_dict(weights, strict=False)
    model.to(torch_device)
    model.eval()

    return model, vocabulary, config


class SmoothCRF(nn.Module):
    """Conditional Random Field layer for sequence labeling."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions, tags, mask):
        """Compute log-likelihood of tag sequence."""
        return self._compute_log_likelihood(emissions, tags, mask)

    def _compute_log_likelihood(self, emissions, tags, mask):
        """Compute the log-likelihood of the tag sequence."""
        batch_size, seq_length, num_tags = emissions.shape

        # Handle soft tags (for label smoothing)
        if tags.dim() == 3:
            tags_hard = tags.argmax(dim=-1)
        else:
            tags_hard = tags

        # Compute score of gold sequence
        score = self.start_transitions[tags_hard[:, 0]]
        for i in range(seq_length - 1):
            if mask is not None:
                valid = mask[:, i + 1]
                score = score + emissions[:, i, tags_hard[:, i]] * mask[:, i].float()
                score = score + self.transitions[tags_hard[:, i], tags_hard[:, i + 1]] * valid.float()
            else:
                score = score + emissions[:, i, tags_hard[:, i]]
                score = score + self.transitions[tags_hard[:, i], tags_hard[:, i + 1]]

        last_idx = mask.sum(dim=1).long() - 1 if mask is not None else torch.full((batch_size,), seq_length - 1)
        last_tags = tags_hard.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]

        # Compute partition function
        partition = self._compute_partition(emissions, mask)

        return (score - partition).mean()

    def _compute_partition(self, emissions, mask):
        """Compute log partition function using forward algorithm."""
        batch_size, seq_length, num_tags = emissions.shape

        # Start with start transitions
        alpha = self.start_transitions + emissions[:, 0]

        for i in range(1, seq_length):
            emit_scores = emissions[:, i].unsqueeze(1)  # (batch, 1, num_tags)
            trans_scores = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            alpha_expand = alpha.unsqueeze(2)  # (batch, num_tags, 1)

            scores = alpha_expand + trans_scores + emit_scores  # (batch, num_tags, num_tags)
            new_alpha = torch.logsumexp(scores, dim=1)  # (batch, num_tags)

            if mask is not None:
                alpha = torch.where(mask[:, i].unsqueeze(1), new_alpha, alpha)
            else:
                alpha = new_alpha

        # Add end transitions
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def viterbi_tags(self, emissions, mask):
        """Find most likely tag sequence using Viterbi algorithm."""
        batch_size, seq_length, num_tags = emissions.shape
        device = emissions.device

        # Initialize
        alpha = self.start_transitions + emissions[:, 0]
        backpointers = []

        for i in range(1, seq_length):
            emit_scores = emissions[:, i].unsqueeze(1)
            trans_scores = self.transitions.unsqueeze(0)
            alpha_expand = alpha.unsqueeze(2)

            scores = alpha_expand + trans_scores + emit_scores
            best_scores, best_tags = scores.max(dim=1)

            if mask is not None:
                alpha = torch.where(mask[:, i].unsqueeze(1), best_scores, alpha)
                backpointers.append(best_tags)
            else:
                alpha = best_scores
                backpointers.append(best_tags)

        # Add end transitions and find best final tag
        alpha = alpha + self.end_transitions
        best_final_scores, best_final_tags = alpha.max(dim=1)

        # Backtrack
        results = []
        for b in range(batch_size):
            if mask is not None:
                seq_len = int(mask[b].sum())
            else:
                seq_len = seq_length

            tags = [best_final_tags[b].item()]
            for i in range(seq_len - 2, -1, -1):
                tags.append(backpointers[i][b, tags[-1]].item())
            tags.reverse()

            results.append((tags, best_final_scores[b].item()))

        return results


class SpanFinderModel(nn.Module):
    """
    SpanFinder model implementation compatible with AllenNLP-trained weights.
    """

    def __init__(
        self,
        transformer_model: str,
        vocab: Vocabulary,
        label_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        max_decoding_spans: int = 128,
        max_recursion_depth: int = 5,
    ):
        super().__init__()
        self.vocab = vocab
        self._max_decoding_spans = max_decoding_spans
        self._max_recursion_depth = max_recursion_depth

        # Transformer encoder
        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.token_dim = self.transformer.config.hidden_size

        # Label embedding
        num_labels = vocab.get_vocab_size("span_label")
        self.label_embedding = nn.Embedding(num_labels, label_dim)

        # Span extractor (endpoint representation)
        self.span_extractor_dim = self.token_dim * 2

        # BIO encoder for span finding
        bio_input_dim = self.span_extractor_dim + self.token_dim + label_dim
        self.bio_encoder = nn.LSTM(
            bio_input_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.bio_classifier = nn.Linear(512, 3)  # B, I, O
        self.crf = SmoothCRF(3)

        # Span typing MLP
        hidden_dims = hidden_dims or [256]
        typing_input_dim = self.span_extractor_dim * 2 + label_dim
        layers = []
        in_dim = typing_input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_labels))
        self.typing_mlp = nn.Sequential(*layers)

        # Special token indices
        self._pad_idx = vocab.get_token_index(DEFAULT_PADDING_TOKEN, "tokens") if "tokens" in vocab._token_to_index else 0
        self._null_idx = vocab.get_token_index(DEFAULT_OOV_TOKEN, "span_label")
        self._vr_idx = vocab.get_token_index("VIRTUAL_ROOT", "span_label")

    @classmethod
    def from_config(cls, config: Dict, vocab: Vocabulary) -> "SpanFinderModel":
        """Create model from AllenNLP config."""
        model_config = config.get("model", config)

        # Extract transformer model name
        word_emb_config = model_config.get("word_embedding", {})
        token_emb_config = word_emb_config.get("token_embedders", {}).get("pieces", {})
        transformer_model = token_emb_config.get("model_name", "xlm-roberta-large")

        # Extract other configs
        label_dim = model_config.get("label_dim", 64)
        max_decoding_spans = model_config.get("max_decoding_spans", 128)
        max_recursion_depth = model_config.get("max_recursion_depth", 5)

        # Extract MLP hidden dims
        typing_config = model_config.get("span_typing", {})
        hidden_dims = typing_config.get("hidden_dims", [256])

        return cls(
            transformer_model=transformer_model,
            vocab=vocab,
            label_dim=label_dim,
            hidden_dims=hidden_dims,
            max_decoding_spans=max_decoding_spans,
            max_recursion_depth=max_recursion_depth,
        )

    def extract_spans(self, token_vec: torch.Tensor, span_boundary: torch.Tensor) -> torch.Tensor:
        """
        Extract span representations using endpoint method.

        Args:
            token_vec: Token representations [batch, seq_len, dim]
            span_boundary: Span boundaries [batch, num_spans, 2]

        Returns:
            Span representations [batch, num_spans, dim*2]
        """
        batch_size, num_spans, _ = span_boundary.shape

        # Get start and end indices
        start_idx = span_boundary[:, :, 0].clamp(min=0)
        end_idx = span_boundary[:, :, 1].clamp(min=0)

        # Gather start and end representations
        start_idx_expanded = start_idx.unsqueeze(-1).expand(-1, -1, token_vec.size(-1))
        end_idx_expanded = end_idx.unsqueeze(-1).expand(-1, -1, token_vec.size(-1))

        start_repr = token_vec.gather(1, start_idx_expanded)
        end_repr = token_vec.gather(1, end_idx_expanded)

        return torch.cat([start_repr, end_repr], dim=-1)

    def forward(self, input_ids, attention_mask):
        """Forward pass - returns token representations."""
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
