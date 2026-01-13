"""Test to verify embedding model distinguishes between different texts.

This test ensures the jina-embeddings-v2-base-code model is loaded correctly
with trust_remote_code=True, which causes it to use the custom JinaBertModel
class instead of a generic BertModel with randomly initialized weights.

Bug: Without trust_remote_code=True, all embeddings had cosine similarity ~1.0
because the model weights weren't loaded correctly.
"""

import numpy as np
import pytest

from codegrapher.vector_store import EmbeddingModel


def test_embedding_model_distinguishes_different_texts():
    """Test that the embedding model gives different embeddings for different texts.

    This is a regression test for the bug where all embeddings were identical
    (cosine similarity ≈ 1.0) due to the model not loading pretrained weights.
    """
    model = EmbeddingModel()

    # Test with distinctly different texts
    texts = [
        "def foo(): return 1",
        "class Bar: pass",
        "import os",
    ]

    embeddings = [model.embed_text(text) for text in texts]

    # Compute pairwise cosine similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            cos_sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities.append(cos_sim)

    # All pairwise similarities should be significantly less than 1.0
    # (Bug caused all similarities to be ≈ 1.0)
    for sim in similarities:
        assert sim < 0.9, f"Embedding similarity too high ({sim:.4f}): model may not be loading weights correctly"

    # At least some pairs should have low similarity
    assert min(similarities) < 0.6, f"Minimum similarity too high ({min(similarities):.4f}): texts are too similar"


def test_embedding_model_similar_texts_have_high_similarity():
    """Test that semantically similar texts have higher similarity."""
    model = EmbeddingModel()

    # Similar texts (both about imports)
    similar_texts = [
        "def import_module(name): import importlib; return importlib.import_module(name)",
        "def load_module(module_name): from importlib import import_module; return import_module(module_name)",
    ]

    # Different texts
    different_texts = [
        "class Animal: pass",
        "def calculate_sum(a, b): return a + b",
    ]

    similar_embs = [model.embed_text(text) for text in similar_texts]
    different_embs = [model.embed_text(text) for text in different_texts]

    # Similarity between similar texts
    sim_similar = np.dot(similar_embs[0], similar_embs[1]) / (
        np.linalg.norm(similar_embs[0]) * np.linalg.norm(similar_embs[1])
    )

    # Similarity between different texts
    sim_different = np.dot(different_embs[0], different_embs[1]) / (
        np.linalg.norm(different_embs[0]) * np.linalg.norm(different_embs[1])
    )

    # Similar texts should have higher similarity than different texts
    assert (
        sim_similar > sim_different
    ), f"Similar texts ({sim_similar:.4f}) should have higher similarity than different texts ({sim_different:.4f})"


def test_query_target_match_has_highest_similarity():
    """Regression test for task_001 query matching issue.

    The query 'import_module_using_spec sys.modules importlib namespace package KeyError'
    should match the function 'import_module_using_spec' with higher similarity
    than unrelated functions.
    """
    model = EmbeddingModel()

    query = "import_module_using_spec sys.modules importlib namespace package KeyError"

    # Target function (what we expect to match)
    target_text = (
        "def import_module_using_spec(spec, module_name): "
        "Import a module using importlib and handle namespace packages."
    )

    # Unrelated functions
    unrelated_texts = [
        "class TestConfig: Configuration for testing import modules",
        "def release_module(version): Release a module with proper versioning",
        "def handle_key_error(key): Handle KeyError when accessing keys",
    ]

    query_emb = model.embed_text(query)
    target_emb = model.embed_text(target_text)
    unrelated_embs = [model.embed_text(text) for text in unrelated_texts]

    # Similarity to target
    target_sim = np.dot(query_emb, target_emb) / (
        np.linalg.norm(query_emb) * np.linalg.norm(target_emb)
    )

    # Similarities to unrelated texts
    unrelated_sims = [
        np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        for emb in unrelated_embs
    ]

    # Target should have higher similarity than any unrelated text
    for unrelated_sim in unrelated_sims:
        assert (
            target_sim > unrelated_sim
        ), f"Target similarity ({target_sim:.4f}) should be higher than unrelated ({unrelated_sim:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
