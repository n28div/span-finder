"""
Vocabulary class compatible with AllenNLP's vocabulary format.
"""

import json
import os
from typing import Dict, Optional


DEFAULT_PADDING_TOKEN = "@@PADDING@@"
DEFAULT_OOV_TOKEN = "@@UNKNOWN@@"


class Vocabulary:
    """
    A vocabulary that maps tokens to indices and vice versa.
    Compatible with AllenNLP's vocabulary format for loading pretrained models.
    """

    def __init__(self):
        self._token_to_index: Dict[str, Dict[str, int]] = {}
        self._index_to_token: Dict[str, Dict[int, str]] = {}

    @classmethod
    def from_files(cls, directory: str) -> "Vocabulary":
        """
        Load vocabulary from AllenNLP-style vocabulary directory.

        Args:
            directory: Path to vocabulary directory containing namespace files.

        Returns:
            Loaded Vocabulary instance.
        """
        vocab = cls()

        # Check for .txt files (AllenNLP format)
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                namespace = filename[:-4]  # Remove .txt extension
                filepath = os.path.join(directory, filename)
                vocab._load_namespace(namespace, filepath)

        # Also check for non_padded_namespaces.txt to understand which namespaces have padding
        non_padded_path = os.path.join(directory, "non_padded_namespaces.txt")
        if os.path.exists(non_padded_path):
            with open(non_padded_path, encoding='utf-8') as f:
                vocab._non_padded_namespaces = set(line.strip() for line in f if line.strip())
        else:
            vocab._non_padded_namespaces = set()

        return vocab

    def _load_namespace(self, namespace: str, filepath: str):
        """Load a single namespace from a file."""
        self._token_to_index[namespace] = {}
        self._index_to_token[namespace] = {}

        with open(filepath, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.rstrip('\n')
                self._token_to_index[namespace][token] = idx
                self._index_to_token[namespace][idx] = token

    def get_token_index(self, token: str, namespace: str = "tokens") -> int:
        """
        Get the index of a token in the specified namespace.
        Returns OOV index if token not found.
        """
        if namespace not in self._token_to_index:
            raise KeyError(f"Namespace '{namespace}' not found in vocabulary")

        vocab = self._token_to_index[namespace]
        if token in vocab:
            return vocab[token]

        # Return OOV index if available
        if DEFAULT_OOV_TOKEN in vocab:
            return vocab[DEFAULT_OOV_TOKEN]

        raise KeyError(f"Token '{token}' not found in namespace '{namespace}'")

    def get_token_from_index(self, index: int, namespace: str = "tokens") -> str:
        """Get the token at the specified index in the namespace."""
        if namespace not in self._index_to_token:
            raise KeyError(f"Namespace '{namespace}' not found in vocabulary")

        return self._index_to_token[namespace].get(index, DEFAULT_OOV_TOKEN)

    def get_vocab_size(self, namespace: str = "tokens") -> int:
        """Get the size of the vocabulary for a namespace."""
        if namespace not in self._token_to_index:
            return 0
        return len(self._token_to_index[namespace])

    def get_index_to_token_vocabulary(self, namespace: str = "tokens") -> Dict[int, str]:
        """Get the index-to-token mapping for a namespace."""
        return self._index_to_token.get(namespace, {})

    def get_token_to_index_vocabulary(self, namespace: str = "tokens") -> Dict[str, int]:
        """Get the token-to-index mapping for a namespace."""
        return self._token_to_index.get(namespace, {})
