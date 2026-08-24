"""
Golden embedding fixture test.

Asserts similarity ordering around the 0.95 threshold, not raw vector
equality. Survives a legitimate implementation swap (optimum vs
onnxruntime direct) but fails loudly if the embedder changes behavior
in a way that breaks the cache hit/miss boundary.

Skipped if embedding deps are not installed.
"""

import json
import pytest
from pathlib import Path

FIXTURE = Path("tests/fixtures/embedding_golden.json")
EMBEDDING_THRESHOLD = 0.95


def _load_fixture():
    return json.loads(FIXTURE.read_text())


try:
    from throttle.embeddings import get_embedding, EMBEDDINGS_AVAILABLE
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False
    EMBEDDINGS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _IMPORT_OK or not EMBEDDINGS_AVAILABLE,
    reason="Embedding deps not installed — skipping golden fixture test"
)


@pytest.fixture(scope="module")
def similarities():
    """Compute all pairwise similarities from the fixture prompts."""
    fixture = _load_fixture()
    prompts = fixture["prompts"]
    
    vectors = [get_embedding(p) for p in prompts]
    
    import numpy as np
    sims = {}
    for pair in fixture["pairs"]:
        a, b = pair["a"], pair["b"]
        va, vb = vectors[a], vectors[b]
        if va is None or vb is None:
            sims[(a, b)] = None
        else:
            sims[(a, b)] = float(np.dot(va, vb))
    return sims


def test_strong_paraphrases_above_threshold(similarities):
    """Strong paraphrases must score above threshold — they should be cache hits."""
    fixture = _load_fixture()
    for pair in fixture["pairs"]:
        if pair["label"] == "strong_paraphrase":
            sim = similarities.get((pair["a"], pair["b"]))
            assert sim is not None, f"Embedding failed for pair {pair}"
            assert sim >= pair["expected_above"], (
                f"Strong paraphrase ({pair['a']}, {pair['b']}) scored {sim:.4f}, "
                f"expected >= {pair['expected_above']}. "
                f"Prompts: {fixture['prompts'][pair['a']]!r} vs "
                f"{fixture['prompts'][pair['b']]!r}"
            )


def test_entity_substitutions_below_threshold(similarities):
    """Entity substitutions must stay below 0.95 — they must NOT be cache hits."""
    fixture = _load_fixture()
    for pair in fixture["pairs"]:
        if pair["label"] == "entity_substitution":
            sim = similarities.get((pair["a"], pair["b"]))
            assert sim is not None, f"Embedding failed for pair {pair}"
            assert sim < EMBEDDING_THRESHOLD, (
                f"Entity substitution ({pair['a']}, {pair['b']}) scored {sim:.4f}, "
                f"expected < {EMBEDDING_THRESHOLD}. "
                f"This pair would be incorrectly served as a cache hit. "
                f"Prompts: {fixture['prompts'][pair['a']]!r} vs "
                f"{fixture['prompts'][pair['b']]!r}"
            )


def test_ordering_paraphrase_above_entity_substitution(similarities):
    """
    Core ordering assertion: genuine paraphrases must score higher than
    entity substitutions. This is the invariant that makes the cache safe.
    If this fails after an implementation swap, the new embedder has
    different semantic geometry and the published numbers no longer hold.
    """
    fixture = _load_fixture()
    for assertion in fixture["ordering_assertions"]:
        higher_pair = tuple(assertion["higher"])
        lower_pair = tuple(assertion["lower"])
        
        sim_higher = similarities.get(higher_pair)
        sim_lower = similarities.get(lower_pair)
        
        assert sim_higher is not None, f"Embedding failed for pair {higher_pair}"
        assert sim_lower is not None, f"Embedding failed for pair {lower_pair}"
        
        assert sim_higher > sim_lower, (
            f"Ordering violated: {higher_pair} scored {sim_higher:.4f}, "
            f"{lower_pair} scored {sim_lower:.4f}. "
            f"Expected {higher_pair} > {lower_pair}. "
            f"Note: {assertion['note']}"
        )


def test_fixture_structure():
    """Fixture file is well-formed and contains required fields."""
    fixture = _load_fixture()
    assert "prompts" in fixture
    assert "pairs" in fixture
    assert "ordering_assertions" in fixture
    assert len(fixture["prompts"]) >= 10
    assert len(fixture["pairs"]) >= 3
    assert any(p["label"] == "entity_substitution" for p in fixture["pairs"])
    assert any(p["label"] == "strong_paraphrase" for p in fixture["pairs"])
