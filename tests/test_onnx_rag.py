"""Tests for lightweight ONNX embedding runtime and NumPy RAG retriever."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from core import models
from phishbench.onnx_embed import ONNXEmbeddingModel
from phishbench.rag import FAISSRetriever, NumPyRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_INDEX_DIR = REPO_ROOT / "results" / "rag_index_clean"


@pytest.fixture(scope="module")
def onnx_model() -> ONNXEmbeddingModel:
    return ONNXEmbeddingModel()


def test_onnx_embedding_dimension_and_norm(onnx_model: ONNXEmbeddingModel) -> None:
    text = "Urgent: please verify your bank credentials immediately."
    vec = onnx_model.encode(text)

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert vec.dtype == np.float32

    # Vector must be unit normalized (L2 norm == 1.0)
    norm = np.linalg.norm(vec)
    assert norm == pytest.approx(1.0, rel=1e-5)


def test_onnx_batch_encoding(onnx_model: ONNXEmbeddingModel) -> None:
    texts = [
        "First short test message.",
        "Second longer test email asking for immediate invoice payment.",
        "Third legitimate company meeting notes regarding Q4 planning.",
    ]
    batch_vecs = onnx_model.encode(texts)

    assert batch_vecs.shape == (3, 384)
    for i in range(3):
        single_vec = onnx_model.encode(texts[i])
        cos_sim = float(np.dot(batch_vecs[i], single_vec))
        assert cos_sim >= 0.99999


def test_onnx_input_truncation(onnx_model: ONNXEmbeddingModel) -> None:
    # Character truncation: text beyond 1500 characters must be truncated identically
    base_text = "This is a security test alert. " * 60  # ~1860 characters
    vec_truncated_arg = onnx_model.encode(base_text, max_text_len=1500)
    vec_manual_slice = onnx_model.encode(base_text[:1500], max_text_len=1500)

    cos_sim = float(np.dot(vec_truncated_arg, vec_manual_slice))
    assert cos_sim == pytest.approx(1.0, abs=1e-6)


def test_numpy_retriever_clean_index_loading() -> None:
    assert CLEAN_INDEX_DIR.exists()
    retriever = FAISSRetriever.load(CLEAN_INDEX_DIR)

    assert retriever.exemplars is not None
    assert retriever.exemplars.shape == (1321, 384)
    assert retriever.exemplars.dtype == np.float32
    assert len(retriever.metadata) == 1321
    assert retriever.index.ntotal == 1321


def test_numpy_retrieval_ranking_and_scores() -> None:
    retriever = FAISSRetriever.load(CLEAN_INDEX_DIR)
    results = retriever.retrieve("Urgent wire transfer request to new account", top_k=5)

    assert len(results) == 5
    # Results must be sorted descending by similarity_score
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

    for item in results:
        assert "id" in item
        assert "text" in item
        assert "label" in item
        assert "label_name" in item
        assert "similarity_score" in item
        assert -1.0 <= item["similarity_score"] <= 1.0


def test_zero_torch_imports_in_isolated_process() -> None:
    """Core requirement: Production RAG inference must not import torch, sentence_transformers, or faiss."""
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"{str(REPO_ROOT)}") / "src"))

import phishbench.rag
import core.models

forbidden = ["torch", "sentence_transformers", "faiss"]
loaded = [m for m in forbidden if m in sys.modules]
if loaded:
    print(f"FAILED: Top-level import loaded {{loaded}}")
    sys.exit(1)

retriever = core.models.get_rag_retriever()
if retriever is None:
    print("FAILED: get_rag_retriever returned None")
    sys.exit(2)

results = retriever.retrieve("Immediate action required: account suspended", top_k=3)
if len(results) != 3:
    print("FAILED: retrieval did not return 3 results")
    sys.exit(3)

post_loaded = [m for m in forbidden if m in sys.modules]
if post_loaded:
    print(f"FAILED: Post-retrieval loaded {{post_loaded}}")
    sys.exit(4)

print("SUCCESS: Zero PyTorch/FAISS imports confirmed.")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"Subprocess output:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "SUCCESS: Zero PyTorch/FAISS imports confirmed." in proc.stdout


def test_m1_joblib_artifact_loaded_in_core_models() -> None:
    models.get_m1_model.cache_clear()
    vec, clf = models.get_m1_model()
    assert hasattr(vec, "transform")
    assert hasattr(clf, "predict_proba")
    # Verify predictions on a sample
    prob = clf.predict_proba(vec.transform(["urgent reset account password"]))[0, 1]
    assert 0.0 <= prob <= 1.0
