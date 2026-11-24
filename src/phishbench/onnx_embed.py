"""Lightweight CPU ONNX embedding runtime for sentence-transformers/all-MiniLM-L6-v2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MAX_SEQ_LENGTH = 256
DEFAULT_MAX_TEXT_LEN = 1500
EMBEDDING_DIM = 384


def resolve_onnx_model_paths(
    model_name: str = DEFAULT_MODEL_NAME,
    model_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Resolve paths to model.onnx and tokenizer.json locally or download from HF Hub.

    Resolution order:
    1. Explicit model_dir argument
    2. IFSI_ONNX_MODEL_DIR environment variable
    3. Local models/all-MiniLM-L6-v2/ directory
    4. Local Hugging Face cache (local_files_only=True)
    5. Download from Hugging Face Hub (local_files_only=False)
    """
    import os

    # Normalize HF repo name (e.g., 'all-MiniLM-L6-v2' -> 'sentence-transformers/all-MiniLM-L6-v2')
    hf_repo = f"sentence-transformers/{model_name}" if "/" not in model_name else model_name
    short_name = model_name.split("/")[-1]

    # 1. Explicit model_dir
    if model_dir:
        md = Path(model_dir)
        m_path = md / "model.onnx" if (md / "model.onnx").exists() else md / "onnx" / "model.onnx"
        t_path = md / "tokenizer.json"
        if m_path.exists() and t_path.exists():
            return m_path, t_path

    # 2. IFSI_ONNX_MODEL_DIR env var
    env_dir = os.environ.get("IFSI_ONNX_MODEL_DIR")
    if env_dir:
        md = Path(env_dir)
        m_path = md / "model.onnx" if (md / "model.onnx").exists() else md / "onnx" / "model.onnx"
        t_path = md / "tokenizer.json"
        if m_path.exists() and t_path.exists():
            return m_path, t_path

    # 3. Local repo models directory
    repo_root = Path(__file__).resolve().parents[2]
    local_candidates = [
        repo_root / "models" / short_name,
        repo_root / "models" / "all-MiniLM-L6-v2",
        repo_root / "models" / "onnx",
        repo_root / "models",
    ]
    for cand in local_candidates:
        m_path = cand / "model.onnx" if (cand / "model.onnx").exists() else cand / "onnx" / "model.onnx"
        t_path = cand / "tokenizer.json"
        if m_path.exists() and t_path.exists():
            return m_path, t_path

    # 4 & 5. Hugging Face Hub (local cache first, then download if needed)
    try:
        from huggingface_hub import hf_hub_download

        # Try local cache first
        for repo_try in (hf_repo, model_name):
            try:
                m_path = Path(hf_hub_download(repo_id=repo_try, filename="onnx/model.onnx", local_files_only=True))
                t_path = Path(hf_hub_download(repo_id=repo_try, filename="tokenizer.json", local_files_only=True))
                return m_path, t_path
            except Exception:
                continue

        # Fall back to remote download
        logger.info("Downloading ONNX model and tokenizer for %s from Hugging Face Hub...", hf_repo)
        m_path = Path(hf_hub_download(repo_id=hf_repo, filename="onnx/model.onnx", local_files_only=False))
        t_path = Path(hf_hub_download(repo_id=hf_repo, filename="tokenizer.json", local_files_only=False))
        return m_path, t_path
    except ImportError:
        pass

    raise FileNotFoundError(
        f"Could not locate ONNX model and tokenizer for '{model_name}'. "
        "Set IFSI_ONNX_MODEL_DIR or install huggingface_hub to download."
    )


class ONNXEmbeddingModel:
    """CPU ONNX embedding model producing 384-dim normalized sentence embeddings.

    Replaces SentenceTransformer('all-MiniLM-L6-v2') in production inference with zero
    PyTorch dependency, 100% vector parity, and minimal memory footprint (~40 MB).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: Path | str | None = None,
        tokenizer_path: Path | str | None = None,
        num_threads: int = 1,
    ) -> None:
        self.model_name = model_name
        self.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
        self.max_text_len = DEFAULT_MAX_TEXT_LEN

        if model_path is None or tokenizer_path is None:
            resolved_m, resolved_t = resolve_onnx_model_paths(model_name)
            self.model_path = Path(model_path or resolved_m)
            self.tokenizer_path = Path(tokenizer_path or resolved_t)
        else:
            self.model_path = Path(model_path)
            self.tokenizer_path = Path(tokenizer_path)

        # Configure ONNX Runtime CPU session
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Configure fast HuggingFace Tokenizer
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.tokenizer.enable_truncation(max_length=self.max_seq_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        logger.info(
            "ONNXEmbeddingModel initialized: model=%s, tokenizer=%s (dim=%d)",
            self.model_path.name,
            self.tokenizer_path.name,
            EMBEDDING_DIM,
        )

    def encode(
        self,
        sentences: str | Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        max_text_len: int | None = DEFAULT_MAX_TEXT_LEN,
    ) -> np.ndarray:
        """Encode sentences into 384-dimensional float32 embeddings.

        Matches SentenceTransformer.encode behavior:
        - Accepts a single string or sequence of strings.
        - Truncates query text to max_text_len characters before tokenization.
        - Truncates tokens to max_seq_length (256).
        - Computes attention-mask-aware mean pooling.
        - Computes L2 normalization when normalize_embeddings=True.
        """
        is_single = isinstance(sentences, str)
        text_list: list[str] = [sentences] if is_single else list(sentences)

        if not text_list:
            empty = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
            return empty[0] if is_single else empty

        # Character truncation (mirrors SentenceTransformer RAG query preprocessing)
        if max_text_len and max_text_len > 0:
            text_list = [t[:max_text_len] for t in text_list]

        all_embeddings: list[np.ndarray] = []

        for start_idx in range(0, len(text_list), batch_size):
            batch_texts = text_list[start_idx : start_idx + batch_size]
            encoded_batch = self.tokenizer.encode_batch(batch_texts)

            input_ids = np.array([e.ids for e in encoded_batch], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded_batch], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encoded_batch], dtype=np.int64)

            ort_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }

            outputs = self.session.run(None, ort_inputs)
            last_hidden_state = outputs[0]  # shape: (batch_size, seq_len, 384)

            # Attention-mask-aware mean pooling
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            pooled = sum_embeddings / sum_mask

            # L2 normalization
            if normalize_embeddings:
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                pooled = pooled / np.clip(norms, a_min=1e-12, a_max=None)

            all_embeddings.append(pooled.astype(np.float32))

        result = np.vstack(all_embeddings)
        return result[0] if is_single else result
