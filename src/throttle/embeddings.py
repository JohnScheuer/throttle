"""Direct ONNX embedding module with frozen contract.

Provides sync embedding functions using pre-exported ONNX weights from HuggingFace Hub.
No torch conversion, single-flight loading with threading.Lock.
"""

import logging
from threading import Lock
from typing import Optional

try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    np = None  # type: ignore

logger = logging.getLogger(__name__)

# Module-level singleton state
_embedder: Optional["_DirectEmbedder"] = None
_load_lock = Lock()


class _DirectEmbedder:
    """Direct ONNX embedder using pre-exported weights."""

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "Embedding dependencies not installed. "
                "Install with: pip install throttle-bench[embeddings]"
            )

        # Download pre-exported ONNX model and tokenizer
        model_path = hf_hub_download(repo_id=model_id, filename="onnx/model.onnx")
        tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.session = ort.InferenceSession(model_path)

    def embed_one(self, text: str) -> "np.ndarray":
        """Generate L2-normalized 384-dim float32 embedding for one text."""
        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })

        # Mean pooling with attention mask
        token_embeddings = outputs[0]
        mask_expanded = np.expand_dims(attention_mask, -1)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(np.sum(attention_mask, axis=1, keepdims=True), a_min=1e-9, a_max=None)
        emb = (summed / counts)[0].astype(np.float32)

        # L2 normalize for cosine via dot product
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def embed_batch(self, texts: list[str]) -> "np.ndarray":
        """Generate embeddings for batch of texts.

        Returns (len(texts), 384) array. Failed rows filled with zeros.
        """
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        results = np.zeros((len(texts), 384), dtype=np.float32)

        for i, text in enumerate(texts):
            try:
                results[i] = self.embed_one(text)
            except Exception as e:
                logger.warning(f"Failed to embed text at index {i}: {e}")
                # Row already initialized to zeros
                pass

        return results


def _get_embedder() -> Optional["_DirectEmbedder"]:
    """Get or create the singleton embedder instance. Thread-safe single-flight."""
    global _embedder

    if not EMBEDDINGS_AVAILABLE:
        return None

    if _embedder is not None:
        return _embedder

    with _load_lock:
        # Double-check after acquiring lock
        if _embedder is not None:
            return _embedder

        try:
            _embedder = _DirectEmbedder()
            return _embedder
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            return None


def get_embedding(text: str) -> Optional["np.ndarray"]:
    """Get L2-normalized 384-dim float32 embedding for text.

    Returns None if embeddings unavailable or loading fails.
    """
    embedder = _get_embedder()
    if embedder is None:
        return None

    try:
        return embedder.embed_one(text)
    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def get_embeddings(texts: list[str]) -> "np.ndarray":
    """Get embeddings for list of texts.

    Returns (len(texts), 384) array. Failed rows filled with np.zeros((384,)).
    Never returns None - if embeddings unavailable, returns all zeros.
    """
    if not EMBEDDINGS_AVAILABLE or not np:
        return np.zeros((len(texts), 384), dtype=np.float32)

    embedder = _get_embedder()
    if embedder is None:
        return np.zeros((len(texts), 384), dtype=np.float32)

    try:
        return embedder.embed_batch(texts)
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        return np.zeros((len(texts), 384), dtype=np.float32)
