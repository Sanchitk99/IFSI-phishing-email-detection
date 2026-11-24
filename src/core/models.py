"""Core model management and resource loaders (cached without Streamlit)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from phishbench.io import read_jsonl
from phishbench.rag import FAISSRetriever

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
M1_MODEL_PATH = MODELS_DIR / "m1_model.joblib"
CLEAN_RAG_INDEX_DIR = RESULTS_DIR / "rag_index_clean"
RAG_INDEX_DIR = CLEAN_RAG_INDEX_DIR
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_m1_model() -> tuple[TfidfVectorizer, LogisticRegression]:
    """Train and cache the TF-IDF + LogisticRegression baseline model on clean train split."""
    # Fast path: load pre-trained serialized artifact if running in default production tree
    if M1_MODEL_PATH.is_file() and DATA_DIR == (BASE_DIR / "data"):
        try:
            import joblib
            data = joblib.load(M1_MODEL_PATH)
            logger.info("Loaded pre-trained M1 model artifact from %s", M1_MODEL_PATH)
            return data["vectorizer"], data["classifier"]
        except Exception as exc:
            logger.warning("Failed to load M1 artifact from %s (%s); falling back to training", M1_MODEL_PATH, exc)

    train_path = DATA_DIR / "splits" / "clean_split_v1" / "train.jsonl"
    if not train_path.is_file():
        raise FileNotFoundError(f"M1 clean training data file not found: {train_path}")
    train_records = read_jsonl(train_path)
    if not train_records:
        raise ValueError(f"M1 clean training data file is empty: {train_path}")

    class_counts = {
        label: sum(int(record["label"]) == label for record in train_records)
        for label in (0, 1)
    }
    logger.info(
        "M1 clean training data loaded from %s: %d rows (legitimate=%d, phishing=%d)",
        train_path,
        len(train_records),
        class_counts[0],
        class_counts[1],
    )

    texts = [r["text"] for r in train_records]
    labels = [int(r["label"]) for r in train_records]

    vec = TfidfVectorizer(max_features=10000, stop_words="english", sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=0.1, random_state=42, max_iter=1000)
    clf.fit(X, labels)
    return vec, clf


@lru_cache(maxsize=1)
def get_rag_retriever() -> FAISSRetriever | None:
    """Load and cache FAISS RAG Retriever for M4 strictly from the verified clean index."""
    clean_index_dir = RESULTS_DIR / "rag_index_clean"
    if clean_index_dir.exists() and (
        (clean_index_dir / "exemplars.npy").exists() or (clean_index_dir / "train.index").exists()
    ):
        try:
            retriever = FAISSRetriever.load(clean_index_dir)
            logger.info("Clean FAISS retriever successfully loaded from %s (%d vectors)", clean_index_dir, retriever.index.ntotal)
            return retriever
        except Exception as exc:
            logger.error("Failed to load clean FAISS index from %s: %s", clean_index_dir, exc)
            return None
    logger.warning("Clean FAISS index missing at %s; retrieval unavailable. No legacy fallback allowed.", clean_index_dir)
    return None


def check_ollama_service(api_base: str = "http://127.0.0.1:11434") -> tuple[bool, list[str]]:
    """Check if local Ollama daemon is reachable and return list of pulled models."""
    try:
        resp = httpx.get(f"{api_base.rstrip('/')}/api/tags", timeout=2.5)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []


def check_groq_service(
    api_key: str | None = None,
    api_base: str = "https://api.groq.com/openai/v1",
    timeout: float = 2.5,
) -> tuple[bool, str, str, list[str]]:
    """Check Groq configuration and reachability without making classification requests.

    Returns: (connected: bool, status: str, message: str, available_models: list[str])
    """
    key = (api_key or os.getenv("GROQ_API_KEY", "")).strip()
    if not key:
        return False, "missing_api_key", "GROQ_API_KEY environment variable is not configured", []

    default_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    base_url = (api_base or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")).rstrip("/")
    models_url = f"{base_url}/models"

    try:
        resp = httpx.get(models_url, headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            models_list = [
                m.get("id", "")
                for m in data.get("data", [])
                if isinstance(m, dict) and "id" in m
            ]
            return True, "online", f"Connected to Groq API (active model: {default_model})", models_list
        elif resp.status_code == 401:
            return False, "invalid_api_key", "Groq authentication failed (HTTP 401: Invalid API Key)", []
        else:
            return False, "error", f"Groq API returned HTTP {resp.status_code}", []
    except httpx.TimeoutException:
        return False, "timeout", f"Groq API connectivity check timed out after {timeout}s", []
    except Exception as exc:
        return False, "unreachable", f"Could not connect to Groq API: {exc}", []
