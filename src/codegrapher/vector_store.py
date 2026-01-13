"""Vector embeddings and FAISS index management.

This module handles:
1. Loading the jina-embeddings-v2-base-code model from HuggingFace
2. Generating embeddings for symbol signatures + documentation
3. FAISS index operations (add, remove, search) per PRD Recipe 4

Per PRD Section 5:
- Model: jina-embeddings-v2-base-code (768-dim vectors, CPU-only)
- Index: FAISS IndexFlatL2 with L2 distance
- Text to embed: signature + first docstring line (≤40 tokens)
"""

import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from codegrapher.models import EMBEDDING_DIM, Symbol

logger = logging.getLogger(__name__)

# Model configuration from PRD Section 5
MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
MODEL_CACHE_DIR = None  # Uses default ~/.cache/huggingface


class EmbeddingModel:
    """Manages the jina-embeddings-v2-base-code model for generating embeddings.

    The model is loaded lazily on first use and cached for subsequent calls.
    Embeddings are generated via mean pooling of the last hidden state.

    Example:
        >>> model = EmbeddingModel()
        >>> embedding = model.embed_text("def foo() -> None:")
        >>> embedding.shape
        (768,)
    """

    def __init__(self) -> None:
        """Initialize the embedding model (lazy loading)."""
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._device = torch.device("cpu")

    def _load_model(self) -> None:
        """Load the tokenizer and model from HuggingFace.

        Downloads the model on first run (~307MB). Subsequent runs
        use the cached version.

        Raises:
            RuntimeError: If model download fails
        """
        if self._model is not None:
            return

        try:
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                MODEL_NAME, trust_remote_code=True
            )
            self._model.eval()  # type: ignore[attr-defined]  # AutoModel has eval() but stubs are incomplete
            logger.info("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{MODEL_NAME}'. "
                f"Options: (1) Fix network and retry, "
                f"(2) Pre-download with 'huggingface-cli download {MODEL_NAME}', "
                f"(3) Check HuggingFace access."
            ) from e

    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text string.

        Args:
            text: Text to embed (typically symbol signature + docstring)

        Returns:
            768-dim float32 numpy array

        Raises:
            RuntimeError: If model loading fails
        """
        self._load_model()

        assert self._tokenizer is not None
        assert self._model is not None

        # Tokenize input
        inputs = self._tokenizer(  # type: ignore[call-arg]  # AutoTokenizer.__call__ exists but stubs are incomplete
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,  # Model max length
        )

        # Generate embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)  # type: ignore[call-arg]  # AutoModel.__call__ exists but stubs are incomplete

        # Mean pooling over sequence length
        # Shape: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs.attention_mask

        # Expand attention mask for pooling
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        # Convert to numpy float32
        embedding = mean_pooled.squeeze(0).cpu().numpy().astype(np.float32)

        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple text strings.

        Args:
            texts: List of texts to embed

        Returns:
            List of 768-dim float32 numpy arrays

        Raises:
            RuntimeError: If model loading fails
        """
        self._load_model()

        assert self._tokenizer is not None
        assert self._model is not None

        embeddings = []

        # Process in batches to avoid OOM
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize
            inputs = self._tokenizer(  # type: ignore[call-arg]  # AutoTokenizer.__call__ exists but stubs are incomplete
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**inputs)  # type: ignore[call-arg]  # AutoModel.__call__ exists but stubs are incomplete

            # Mean pooling
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs.attention_mask

            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            # Convert to numpy
            batch_embeddings = mean_pooled.cpu().numpy().astype(np.float32)
            embeddings.extend([batch_embeddings[i] for i in range(len(batch))])

        return embeddings


class FAISSIndexManager:
    """Manages FAISS vector index for similarity search.

    Implements per PRD Recipe 4:
    - IndexFlatL2 for CPU brute-force search
    - Parallel symbol_ids list for ID lookup
    - add/remove/search operations
    - Persistence to disk

    Example:
        >>> manager = FAISSIndexManager(index_path=".codegraph/index.faiss")
        >>> manager.add_symbols(symbols)
        >>> results = manager.search(query_embedding, k=20)
        >>> manager.save()
    """

    def __init__(self, index_path: Path) -> None:
        """Initialize the FAISS index manager.

        Args:
            index_path: Path to save/load the FAISS index file
        """
        self.dim = EMBEDDING_DIM
        self.index_path = index_path
        self.symbol_ids: List[str] = []

        # Try to load existing index
        if self.index_path.exists():
            self.load()
        else:
            # Create new empty index
            self.index = faiss.IndexFlatL2(self.dim)

    def add_symbols(self, symbols: List[Symbol]) -> None:
        """Add new symbols to the index.

        Args:
            symbols: List of Symbol objects with embeddings
        """
        if not symbols:
            return

        # Stack embeddings into array
        embeddings = np.array([s.embedding for s in symbols], dtype="float32")

        # Add to FAISS index
        self.index.add(embeddings)  # type: ignore[call-arg]  # FAISS IndexFlatL2.add() exists but stubs are incomplete

        # Track symbol IDs (parallel to index)
        self.symbol_ids.extend([s.id for s in symbols])

    def remove_symbols(self, symbol_ids: List[str]) -> None:
        """Remove symbols from the index.

        Note: FAISS IndexFlatL2 doesn't support in-place removal,
        so we rebuild the index without the removed symbols.

        Args:
            symbol_ids: List of symbol IDs to remove
        """
        if not symbol_ids:
            return

        # Find indices to keep
        removed_set = set(symbol_ids)
        keep_indices = [
            i for i, sid in enumerate(self.symbol_ids) if sid not in removed_set
        ]

        if not keep_indices:
            # Empty index
            self.index = faiss.IndexFlatL2(self.dim)
            self.symbol_ids = []
            return

        # Rebuild index with kept vectors
        old_vectors = np.array(
            [self.index.reconstruct(i) for i in keep_indices], dtype="float32"  # type: ignore[call-arg]  # FAISS reconstruct() exists but stubs are incomplete
        )

        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(old_vectors)  # type: ignore[call-arg]  # FAISS IndexFlatL2.add() exists but stubs are incomplete
        self.symbol_ids = [self.symbol_ids[i] for i in keep_indices]

    def search(
        self, query_embedding: np.ndarray, k: int = 20
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors.

        Args:
            query_embedding: 768-dim query vector
            k: Number of results to return

        Returns:
            List of (symbol_id, distance) tuples, sorted by distance

        Raises:
            RuntimeError: If FAISS search fails (index corrupted)
        """
        try:
            query = np.array([query_embedding], dtype="float32")
            distances, indices = self.index.search(query, k)
        except RuntimeError as e:
            logger.error(f"FAISS index corrupted: {e}")
            raise RuntimeError(
                "FAISS search failed - index rebuild required"
            ) from e

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.symbol_ids):  # Valid index
                results.append((self.symbol_ids[idx], float(dist)))

        return results

    def save(self) -> None:
        """Save index to disk.

        Creates parent directory if needed. Saves both the FAISS
        index and the parallel symbol_ids list.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save symbol_ids as text file
        ids_path = self.index_path.with_suffix(".ids")
        with open(ids_path, "w") as f:
            f.write("\n".join(self.symbol_ids))

    def load(self) -> None:
        """Load index from disk.

        Raises:
            RuntimeError: If index file is corrupted
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load symbol_ids
            ids_path = self.index_path.with_suffix(".ids")
            with open(ids_path) as f:
                self.symbol_ids = [line.strip() for line in f]

            # Verify dimension matches
            if self.index.d != self.dim:
                raise RuntimeError(
                    f"Index dimension mismatch: expected {self.dim}, "
                    f"got {self.index.d}"
                )

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise RuntimeError("Cannot load FAISS index - rebuild required") from e

    def __len__(self) -> int:
        """Return the number of symbols in the index."""
        return len(self.symbol_ids)


def generate_symbol_embeddings(
    symbols: List[Symbol], model: EmbeddingModel
) -> List[Symbol]:
    """Generate embeddings for symbols, updating them in-place.

    Per PRD Section 5, the text to embed is:
    - Function signature
    - First line of docstring (if present)

    Args:
        symbols: List of Symbol objects to embed
        model: EmbeddingModel instance

    Returns:
        Same list with updated embeddings (modifies in-place)
    """
    # Prepare text for each symbol
    texts = []
    for symbol in symbols:
        text = symbol.signature
        if symbol.doc:
            text += " " + symbol.doc
        texts.append(text)

    # Generate embeddings in batch
    embeddings = model.embed_batch(texts)

    # Update symbols with embeddings
    for symbol, embedding in zip(symbols, embeddings):
        symbol.embedding = embedding

    return symbols
