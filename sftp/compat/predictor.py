"""
Simple predictor for SpanFinder models that doesn't require AllenNLP.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer

from .model_loader import load_archive, SpanFinderModel
from .vocabulary import Vocabulary
from ..utils import Span, VIRTUAL_ROOT


BIO = ["B", "I", "O"]


@dataclass
class PredictionResult:
    """Result of a prediction."""
    span: Span
    sentence: List[str]
    meta: dict


class SimplePredictor:
    """
    Simple predictor for SpanFinder models.

    This predictor can load AllenNLP-trained models and run inference
    without requiring AllenNLP as a dependency.
    """

    def __init__(
        self,
        model: SpanFinderModel,
        vocab: Vocabulary,
        config: dict,
        device: str = "cpu",
    ):
        self.model = model
        self.vocab = vocab
        self.config = config
        self.device = device

        # Get transformer model name from config
        model_config = config.get("model", config)
        word_emb_config = model_config.get("word_embedding", {})
        token_emb_config = word_emb_config.get("token_embedders", {}).get("pieces", {})
        transformer_model = token_emb_config.get("model_name", "xlm-roberta-large")

        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)

        # Try to load spacy for sentence tokenization
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None

        # Get special indices
        self._vr_idx = vocab.get_token_index(VIRTUAL_ROOT, "span_label")
        self._null_idx = vocab.get_token_index("@@UNKNOWN@@", "span_label")

    @classmethod
    def from_path(
        cls,
        model_path: str,
        device: str = "cpu",
    ) -> "SimplePredictor":
        """
        Load a predictor from a model path.

        Args:
            model_path: Path to model archive, directory, or URL.
            device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.).

        Returns:
            SimplePredictor instance.
        """
        model, vocab, config = load_archive(model_path, device=device)
        return cls(model, vocab, config, device)

    def _tokenize(self, sentence: Union[str, List[str]]) -> List[str]:
        """Tokenize a sentence if needed."""
        if isinstance(sentence, str):
            if self.nlp is not None:
                doc = self.nlp(sentence)
                return [token.text for token in doc]
            else:
                # Simple whitespace tokenization fallback
                return sentence.split()
        return sentence

    def _prepare_input(self, tokens: List[str]) -> tuple:
        """Prepare input for the model."""
        # Use transformer tokenizer with word-level tokenization tracking
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )

        # Build word-to-subword mapping
        word_ids = encoding.word_ids()
        offsets = []
        current_word = None
        start_idx = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != current_word:
                if current_word is not None:
                    offsets.append((start_idx, idx - 1))
                current_word = word_id
                start_idx = idx

        if current_word is not None:
            offsets.append((start_idx, len(word_ids) - 2))  # -2 for last special token

        return encoding, offsets

    @torch.no_grad()
    def predict_sentence(
        self,
        sentence: Union[str, List[str]],
        output_format: str = "span",
    ) -> PredictionResult:
        """
        Predict spans on a single sentence.

        Args:
            sentence: Input sentence as string or list of tokens.
            output_format: Output format ('span' or 'json').

        Returns:
            PredictionResult with span tree, tokens, and metadata.
        """
        # Tokenize if needed
        tokens = self._tokenize(sentence)

        # Prepare input
        encoding, offsets = self._prepare_input(tokens)

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Get token representations
        token_vec = self.model(input_ids, attention_mask)

        # Run iterative decoding
        spans = self._decode(token_vec, attention_mask, offsets)

        # Convert to Span object
        result_span = self._build_span_tree(spans, tokens)

        if output_format == "json":
            return PredictionResult(
                span=result_span.to_json(),
                sentence=tokens,
                meta={},
            )

        return PredictionResult(
            span=result_span,
            sentence=tokens,
            meta={},
        )

    def _decode(
        self,
        token_vec: torch.Tensor,
        attention_mask: torch.Tensor,
        offsets: List[tuple],
    ) -> List[dict]:
        """
        Iteratively decode spans from token representations.

        This implements the hierarchical decoding process:
        1. Start from virtual root
        2. For each parent, find children using BIO tagging
        3. Type each child span
        4. Repeat until max depth or no more children
        """
        batch_size = token_vec.size(0)
        seq_len = token_vec.size(1)
        device = token_vec.device

        # Initialize with virtual root
        spans = [{
            "start": -1,
            "end": -1,
            "label": VIRTUAL_ROOT,
            "parent": None,
            "confidence": 1.0,
            "is_parent": True,
        }]

        # Parents to process
        parents_to_process = [0]  # Virtual root index

        for depth in range(self.model._max_recursion_depth):
            if not parents_to_process:
                break

            new_parents = []

            for parent_idx in parents_to_process:
                parent = spans[parent_idx]

                # Get parent span representation
                if parent["start"] == -1:
                    # Virtual root - use [CLS] token
                    parent_repr = token_vec[:, 0:1, :]
                else:
                    # Real span
                    start = offsets[parent["start"]][0] if parent["start"] < len(offsets) else 0
                    end = offsets[parent["end"]][1] if parent["end"] < len(offsets) else seq_len - 1
                    start_repr = token_vec[:, start:start+1, :]
                    end_repr = token_vec[:, end:end+1, :]
                    parent_repr = torch.cat([start_repr, end_repr], dim=-1)

                # Get parent label embedding
                parent_label_idx = self.vocab.get_token_index(parent["label"], "span_label")
                parent_label_emb = self.model.label_embedding(
                    torch.tensor([[parent_label_idx]], device=device)
                )

                # Expand parent representation for BIO tagging
                parent_repr_expanded = parent_repr.expand(-1, seq_len, -1)
                if parent_repr_expanded.size(-1) != self.model.span_extractor_dim:
                    # Pad or adjust if needed
                    parent_repr_expanded = token_vec[:, 0:1, :].expand(-1, seq_len, -1)
                    parent_repr_expanded = torch.cat([parent_repr_expanded, parent_repr_expanded], dim=-1)

                parent_label_expanded = parent_label_emb.expand(-1, seq_len, -1)

                # Concatenate features for BIO encoder
                bio_input = torch.cat([
                    parent_repr_expanded,
                    token_vec,
                    parent_label_expanded,
                ], dim=-1)

                # Run BIO encoder
                bio_output, _ = self.model.bio_encoder(bio_input)
                bio_logits = self.model.bio_classifier(bio_output)

                # Decode BIO tags using Viterbi
                decoded = self.model.crf.viterbi_tags(bio_logits, attention_mask.bool())
                bio_tags = decoded[0][0]  # First (only) batch item

                # Convert BIO tags to spans
                child_spans = self._bio_to_spans(bio_tags, offsets)

                # Type each child span
                for child_start, child_end in child_spans:
                    if child_start >= len(offsets) or child_end >= len(offsets):
                        continue

                    # Get child span representation
                    c_start = offsets[child_start][0]
                    c_end = offsets[child_end][1]
                    child_start_repr = token_vec[:, c_start:c_start+1, :]
                    child_end_repr = token_vec[:, c_end:c_end+1, :]
                    child_repr = torch.cat([child_start_repr, child_end_repr], dim=-1)

                    # Typing input: parent_label + parent_span + child_span
                    typing_input = torch.cat([
                        parent_label_emb,
                        parent_repr if parent_repr.size(-1) == self.model.span_extractor_dim else child_repr,
                        child_repr,
                    ], dim=-1)

                    # Get label prediction
                    typing_logits = self.model.typing_mlp(typing_input)
                    label_probs = torch.softmax(typing_logits, dim=-1)
                    label_confidence, label_idx = label_probs.max(dim=-1)

                    label = self.vocab.get_token_from_index(label_idx.item(), "span_label")
                    confidence = label_confidence.item()

                    # Add child span
                    child_span_idx = len(spans)
                    spans.append({
                        "start": child_start,
                        "end": child_end,
                        "label": label,
                        "parent": parent_idx,
                        "confidence": confidence,
                        "is_parent": True,  # Assume all can be parents
                    })
                    new_parents.append(child_span_idx)

            parents_to_process = new_parents[:self.model._max_decoding_spans - len(spans)]

        return spans

    def _bio_to_spans(self, bio_tags: List[int], offsets: List[tuple]) -> List[tuple]:
        """Convert BIO tag sequence to span boundaries."""
        spans = []
        current_start = None

        for idx, tag_idx in enumerate(bio_tags):
            if idx >= len(offsets) + 1:  # +1 for special tokens
                break

            tag = BIO[tag_idx] if tag_idx < len(BIO) else "O"

            if tag == "B":
                if current_start is not None:
                    spans.append((current_start, idx - 2))  # -2 for 0-indexed and previous
                current_start = idx - 1  # -1 for special token offset
            elif tag == "O":
                if current_start is not None:
                    spans.append((current_start, idx - 2))
                    current_start = None

        if current_start is not None:
            spans.append((current_start, len(bio_tags) - 2))

        # Filter valid spans
        valid_spans = []
        for start, end in spans:
            if 0 <= start < len(offsets) and 0 <= end < len(offsets) and start <= end:
                valid_spans.append((start, end))

        return valid_spans

    def _build_span_tree(self, spans: List[dict], tokens: List[str]) -> Span:
        """Build a Span tree from flat span list."""
        # Create Span objects
        span_objects = []
        for s in spans:
            span_obj = Span(
                start_idx=s["start"],
                end_idx=s["end"],
                label=s["label"],
                is_parent=s["is_parent"],
                confidence=s["confidence"],
            )
            span_objects.append(span_obj)

        # Link parents and children
        for idx, s in enumerate(spans):
            if s["parent"] is not None:
                parent_obj = span_objects[s["parent"]]
                child_obj = span_objects[idx]
                child_obj.parent = parent_obj
                parent_obj._children.append(child_obj)

        # Return virtual root
        return span_objects[0]

    def predict_batch_sentences(
        self,
        sentences: List[Union[str, List[str]]],
        max_tokens: int = 512,
        progress: bool = False,
    ) -> List[PredictionResult]:
        """
        Predict spans on multiple sentences.

        Args:
            sentences: List of sentences.
            max_tokens: Maximum tokens per batch (for future batching).
            progress: Show progress bar.

        Returns:
            List of PredictionResult objects.
        """
        results = []

        if progress:
            try:
                from tqdm import tqdm
                sentences = tqdm(sentences, desc="Predicting")
            except ImportError:
                pass

        for sentence in sentences:
            result = self.predict_sentence(sentence)
            results.append(result)

        return results

    def economize(
        self,
        max_decoding_spans: Optional[int] = None,
        max_recursion_depth: Optional[int] = None,
    ):
        """Limit decoding for faster inference."""
        if max_decoding_spans is not None:
            self.model._max_decoding_spans = max_decoding_spans
        if max_recursion_depth is not None:
            self.model._max_recursion_depth = max_recursion_depth
