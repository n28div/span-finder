"""
HuggingFace Hub integration for SpanFinder.

Usage:
    # Load from HF Hub
    from sftp.hf_model import SpanFinderHF
    model = SpanFinderHF.from_pretrained("username/my-span-finder")

    # Load from local directory
    model = SpanFinderHF.from_pretrained("models/spanfinder")

    # Upload to HF Hub
    model.push_to_hub("username/my-span-finder")

    # Use exactly like SpanPredictor
    result = model.predict_sentence("John gave Mary a book.")
"""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from allennlp.predictors import Predictor

# These imports register AllenNLP components via their decorators.
from .data_reader import SRLDatasetReader  # noqa: registers 'semantic_role_labeling'
from .models import SpanModel  # noqa: registers 'span' model
from .modules import BIOSpanFinder, ComboSpanExtractor, MLPSpanTyping  # noqa
from .predictor import SpanPredictor  # noqa: registers 'span' predictor


def span_tree_to_frames(tree, tokens: List[str]) -> List[dict]:
    """Convert a predicted Span tree (virtual-root format) to the frames list format."""
    frames = []
    for event in tree:
        i, j = event.start_idx, event.end_idx
        roles = [
            {
                "name": role.label,
                "filler": " ".join(tokens[role.start_idx : role.end_idx + 1]),
                "idxs": [role.start_idx, role.end_idx],
            }
            for role in event
        ]
        frames.append(
            {
                "name": event.label,
                "idxs": [i, j],
                "activation": " ".join(tokens[i : j + 1]),
                "roles": roles,
            }
        )
    return frames


class SpanFinderHF:
    """
    Thin wrapper around SpanPredictor that adds HuggingFace Hub support.

    All SpanPredictor methods (predict_sentence, predict_batch_sentences,
    force_decode, economize, …) are transparently delegated.
    """

    def __init__(
        self,
        predictor: SpanPredictor,
        model_dir: Optional[Union[str, Path]] = None,
    ):
        self._predictor = predictor
        self._model_dir = Path(model_dir) if model_dir is not None else None

    # --- transparent delegation -------------------------------------------

    def __getattr__(self, name: str):
        return getattr(self._predictor, name)

    def __repr__(self) -> str:
        src = str(self._model_dir) if self._model_dir else "in-memory"
        return f"SpanFinderHF(source={src!r})"

    # --- predict API --------------------------------------------------------

    def predict(
        self,
        *,
        tokens: Optional[List[str]] = None,
        ontology_mapping=None,
        output_format: str = "span",
    ):
        """
        Predict spans on a single input.

        Pass exactly one of:
        - ``text="…"``   — raw string, tokenized internally with SpacyTokenizer.
        - ``tokens=[…]`` — already word-tokenized list, skips SpacyTokenizer.

        Returns a :class:`~sftp.predictor.span_predictor.PredictionReturn` named-tuple.
        """
        pred = self._predictor.predict_sentence(
            tokens=tokens,
            ontology_mapping=ontology_mapping,
            output_format=output_format,
        )

        return {"tokens": tokens, "frames": span_tree_to_frames(pred.span, tokens)}

    def predict_batch(
        self,
        *,
        tokens: Optional[List[List[str]]] = None,
        max_tokens: int = 512,
        ontology_mapping=None,
        output_format: str = "span",
        progress: bool = False,
    ):
        """
        Predict spans on a batch of inputs.

        Pass exactly one of:
        - ``texts=[…]``    — list of raw strings.
        - ``tokens=[[…]]`` — list of pre-tokenized token lists.

        Returns a list of :class:`~sftp.predictor.span_predictor.PredictionReturn`.
        """
        preds = self._predictor.predict_batch_sentences(
            tokens=tokens,
            max_tokens=max_tokens,
            ontology_mapping=ontology_mapping,
            output_format=output_format,
            progress=progress,
        )

        return [
            {"tokens": tokens, "frames": span_tree_to_frames(p.span, t)}
            for t, p in zip(tokens, preds)
        ]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        cuda_device: int = -1,
        token: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> "SpanFinderHF":
        """
        Load a SpanFinder from a local directory or HuggingFace Hub repo.

        Args:
            pretrained_model_name_or_path: Local path OR HF repo id
                (e.g. ``"username/span-finder"``).
            cuda_device: CUDA device index (-1 = CPU).
            token: HF API token for private repos.
            revision: Branch / tag / commit to download from HF Hub.
        """
        path = Path(str(pretrained_model_name_or_path))
        if path.exists():
            model_dir = path
        else:
            from huggingface_hub import snapshot_download

            dl_kwargs: dict = {}
            if token is not None:
                dl_kwargs["use_auth_token"] = token
            if revision is not None:
                dl_kwargs["revision"] = revision

            model_dir = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    **dl_kwargs,
                )
            )

        predictor = Predictor.from_path(
            str(model_dir),
            predictor_name="span",
            cuda_device=cuda_device,
        )
        return cls(predictor, model_dir=model_dir)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload SpanFinder model",
        private: bool = False,
        token: Optional[str] = None,
    ) -> str:
        """
        Serialize the in-memory model and upload it to HuggingFace Hub.

        Creates the repo if it does not exist yet.

        Args:
            repo_id: Target repo, e.g. ``"username/my-span-finder"``.
            commit_message: Commit message for the HF commit.
            private: Create as private repo.
            token: HF API token.

        Returns:
            URL of the uploaded repository.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=repo_id, private=private, repo_type="model", exist_ok=True
        )

        with tempfile.TemporaryDirectory() as tmp:
            self._serialize(Path(tmp))
            api.upload_folder(
                folder_path=tmp,
                repo_id=repo_id,
                commit_message=commit_message,
            )

        return f"https://huggingface.co/{repo_id}"

    # --- serialization ------------------------------------------------------

    def _serialize(self, output_dir: Path) -> None:
        """Write weights.th + vocabulary/ + config.json into output_dir."""
        import torch

        output_dir.mkdir(parents=True, exist_ok=True)
        model = self._predictor._model

        # Model weights
        torch.save(model.state_dict(), output_dir / "weights.th")

        # Vocabulary
        model.vocab.save_to_files(str(output_dir / "vocabulary"))

        # Config — must be present so Predictor.from_path can reconstruct the model
        if self._model_dir is None or not (self._model_dir / "config.json").exists():
            raise RuntimeError(
                "config.json not found. Load the model with from_pretrained() so "
                "the source directory is tracked, or place config.json manually."
            )
        shutil.copy2(self._model_dir / "config.json", output_dir / "config.json")

    @classmethod
    def save_pretrained(
        cls,
        model_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> None:
        """
        Copy a local AllenNLP model directory into output_dir in the canonical
        layout expected by from_pretrained().

        Useful for converting an existing trained model:
            SpanFinderHF.save_pretrained("models/spanfinder", "exported/")
        """
        src, dst = Path(model_dir), Path(output_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
