#!/usr/bin/env python3
"""
RAG System for Patient Report Q&A
Handles document chunking, embedding generation, and retrieval
"""

import json
import os
import pickle
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Disable MPS (Apple Silicon GPU) before importing torch to prevent meta tensor errors
# This must be done before any torch import
if platform.system() == "Darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.error_handling import ErrorHandler, safe_execute
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to import medical utilities
try:
    from src.utils.medical_utils import get_medical_embedder

    MEDICAL_UTILS_AVAILABLE = True
except ImportError:
    MEDICAL_UTILS_AVAILABLE = False
    logger.debug("Medical utilities not available")

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. RAG system requires this package. "
        "Install with: pip install sentence-transformers"
    )

# NOTE: On some macOS environments, importing `faiss` can segfault (native binary mismatch).
# We therefore skip FAISS on Darwin by default to keep the app/test suite stable.
if platform.system() == "Darwin":
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False
    logger.info("FAISS disabled on macOS (Darwin) to avoid native import crashes; using numpy-based search.")
else:
    try:
        import faiss  # type: ignore

        FAISS_AVAILABLE = True
    except ImportError:
        faiss = None  # type: ignore[assignment]
        FAISS_AVAILABLE = False
        logger.warning(
            "faiss-cpu not available. Using numpy-based similarity search (slower). "
            "For better performance, install with: pip install faiss-cpu"
        )


class RAGSystem:
    """
    RAG System for retrieving relevant information from patient discharge summaries
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_biobert: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        data_path: Optional[str] = None,
        embeddings_cache_dir: str = "models/rag_embeddings",
        hadm_id: Optional[int] = None,
    ):
        """
        Initialize RAG system

        Args:
          embedding_model: Name of sentence transformer model for embeddings
          use_biobert: If True, use BioBERT for medical text (overrides embedding_model)
          chunk_size: Size of text chunks for embedding
          chunk_overlap: Overlap between chunks
          data_path: Path to processed discharge summaries CSV
          embeddings_cache_dir: Directory to cache embeddings
          hadm_id: Optional HADM ID to load only a single patient's record
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings_cache_dir = Path(embeddings_cache_dir)
        self.embeddings_cache_dir.mkdir(parents=True, exist_ok=True)
        self.hadm_id = hadm_id
        self.use_biobert = use_biobert

        self.error_handler = ErrorHandler(logger)

        # Initialize embedding model
        if use_biobert and MEDICAL_UTILS_AVAILABLE:
            # Use BioBERT for medical text
            try:
                medical_embedder = get_medical_embedder()
                if medical_embedder and medical_embedder.embedder:
                    logger.info("Using BioBERT embedder for medical text")
                    self.embedding_model = medical_embedder.embedder
                    self.embedding_dim = medical_embedder.get_embedding_dimension()
                    logger.info(f"BioBERT embedder loaded. Dimension: {self.embedding_dim}")
                else:
                    logger.warning("BioBERT not available, falling back to default model")
                    use_biobert = False
            except Exception as e:
                logger.warning(f"Failed to load BioBERT: {e}. Falling back to default model.")
                use_biobert = False

        if not use_biobert and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Use standard embedding model
            try:
                logger.info(f"Loading embedding model: {embedding_model}")
                # Set cache directory explicitly for Cloud Run
                cache_dir = (
                    os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE") or os.path.expanduser("~/.cache/huggingface")
                )
                logger.info(f"Using cache directory: {cache_dir}")

                # Workaround for MPS meta tensor bug on macOS:
                # Use a custom embedding wrapper instead of SentenceTransformer
                import torch
                from transformers import AutoModel, AutoTokenizer

                model_name = f"sentence-transformers/{embedding_model}"
                logger.info(f"Loading model {model_name} with custom wrapper...")

                class SimpleEmbedder:
                    """Simple wrapper for transformer model to generate embeddings without SentenceTransformer"""

                    def __init__(self, model_name, cache_dir):
                        import torch
                        from transformers import AutoConfig
                        from huggingface_hub import hf_hub_download

                        # Load tokenizer
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

                        # Load model config
                        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
                        self.embedding_dim = config.hidden_size

                        # Force model initialization on CPU (not meta device)
                        # by using torch.device context and low_cpu_mem_usage=False
                        try:
                            # Download safetensors and load manually
                            from safetensors.torch import load_file

                            weights_path = hf_hub_download(
                                repo_id=model_name, filename="model.safetensors", cache_dir=cache_dir
                            )
                            logger.info(f"Downloaded safetensors to: {weights_path}")

                            # Load state dict to CPU
                            state_dict = load_file(weights_path, device="cpu")
                            logger.info(f"Loaded state dict with {len(state_dict)} tensors")

                            # Create model on CPU device explicitly
                            with torch.device("cpu"):
                                from transformers import BertModel

                                self.model = BertModel._from_config(config, torch_dtype=torch.float32)

                            # Load state dict (strict=False to handle minor key mismatches like position_ids)
                            self.model.load_state_dict(state_dict, strict=False)
                            logger.info("Loaded weights successfully")

                        except Exception as e:
                            logger.warning(f"Safetensors loading failed: {e}, trying pytorch_model.bin...")
                            try:
                                weights_path = hf_hub_download(
                                    repo_id=model_name, filename="pytorch_model.bin", cache_dir=cache_dir
                                )
                                state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

                                with torch.device("cpu"):
                                    from transformers import BertModel

                                    self.model = BertModel._from_config(config, torch_dtype=torch.float32)

                                self.model.load_state_dict(state_dict, strict=False)
                                logger.info("Loaded weights from pytorch_model.bin successfully")
                            except Exception as e2:
                                logger.error(f"All loading methods failed: {e2}")
                                raise

                        # Ensure eval mode
                        self.model.eval()

                    def encode(self, texts, show_progress_bar=False, batch_size=32, convert_to_numpy=True):
                        """Encode texts to embeddings using mean pooling"""
                        import numpy as np

                        if isinstance(texts, str):
                            texts = [texts]

                        all_embeddings = []
                        for i in range(0, len(texts), batch_size):
                            batch = texts[i : i + batch_size]

                            # Tokenize
                            encoded = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")

                            # Generate embeddings
                            with torch.no_grad():
                                outputs = self.model(**encoded)

                            # Mean pooling
                            attention_mask = encoded["attention_mask"]
                            token_embeddings = outputs.last_hidden_state
                            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                                input_mask_expanded.sum(1), min=1e-9
                            )

                            if convert_to_numpy:
                                embeddings = embeddings.cpu().numpy()

                            all_embeddings.append(embeddings)

                        if convert_to_numpy:
                            return np.vstack(all_embeddings)
                        return torch.cat(all_embeddings, dim=0)

                    def get_sentence_embedding_dimension(self):
                        return self.embedding_dim

                self.embedding_model = SimpleEmbedder(model_name, cache_dir)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded successfully. Dimension: {self.embedding_dim}")

                # Verify model works by encoding a test string
                try:
                    test_embedding = self.embedding_model.encode("test", convert_to_numpy=True)
                    logger.info(f"Model verification successful. Test embedding shape: {test_embedding.shape}")
                except Exception as verify_error:
                    logger.warning(f"Model loaded but verification failed: {verify_error}")
            except ImportError as e:
                logger.error(f"Import error loading embedding model: {e}. Is sentence-transformers installed?")
                self.embedding_model = None
                self.embedding_dim = None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}", exc_info=True)
                self.embedding_model = None
                self.embedding_dim = None
        elif not use_biobert:
            self.embedding_model = None
            self.embedding_dim = None
            logger.error(
                f"Embedding model not available. SENTENCE_TRANSFORMERS_AVAILABLE={SENTENCE_TRANSFORMERS_AVAILABLE}, use_biobert={use_biobert}"
            )

        # Initialize vector store
        self.index = None
        self.chunks = []
        self.metadata = []
        self.df = None

        # Load data if provided
        if data_path:
            self.load_data(data_path, hadm_id=hadm_id or self.hadm_id)

    def load_custom_documents(self, documents: List[Dict[str, any]], force_rebuild: bool = False):
        """
        Load custom documents (text, PDF, images) into the RAG system

        Args:
          documents: List of document dictionaries with 'text' and 'metadata' keys
          force_rebuild: Force rebuild embeddings even if cache exists
        """
        if not documents:
            raise ValueError("No documents provided")

        logger.info(f"Loading {len(documents)} custom documents")

        # Create chunks from custom documents
        self.chunks = []
        self.metadata = []

        for doc_idx, doc in enumerate(documents):
            text = doc.get("text", "")
            if not text or not text.strip():
                logger.warning(f"Document {doc_idx} has no text content, skipping")
                continue

            # Split text into chunks
            text_chunks = self._split_text(text)

            for chunk_idx, chunk in enumerate(text_chunks):
                self.chunks.append(chunk)
                self.metadata.append(
                    {
                        "document_index": doc_idx,
                        "document_name": doc.get("file_name", f"doc_{doc_idx}"),
                        "document_type": doc.get("file_type", "unknown"),
                        "chunk_index": chunk_idx,
                        "total_chunks": len(text_chunks),
                        **doc.get("metadata", {}),
                    }
                )

        logger.info(f"Created {len(self.chunks)} chunks from {len(documents)} documents")

        # Validate that we have chunks
        if not self.chunks:
            error_msg = "No chunks created from documents. All documents may be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Generate embeddings
        if not self.embedding_model:
            error_msg = "Cannot create embeddings: embedding model not available"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            logger.info(f"Generating embeddings for {len(self.chunks)} chunks...")
            embeddings = self._generate_embeddings()

            if embeddings is None or len(embeddings) == 0:
                error_msg = "Failed to generate embeddings: empty embeddings array"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Generated embeddings with shape: {embeddings.shape}")

            # Keep a copy so the caller can persist/re-hydrate without re-embedding.
            self._last_embeddings = embeddings

            # Build index
            self._build_index(embeddings)

            # Validate that index was built successfully
            if FAISS_AVAILABLE:
                if self.index is None:
                    error_msg = "Failed to build FAISS index"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")
            else:
                if not hasattr(self, "embeddings_normalized") or len(self.embeddings_normalized) == 0:
                    error_msg = "Failed to build numpy-based index: embeddings_normalized not set"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                logger.info(f"Numpy-based index built successfully with {len(self.embeddings_normalized)} vectors")

            logger.info(f"Successfully loaded {len(documents)} custom documents with {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error during document loading: {e}", exc_info=True)
            # Reset state on error
            self.chunks = []
            self.metadata = []
            self.index = None
            if hasattr(self, "embeddings_normalized"):
                delattr(self, "embeddings_normalized")
            raise

    @safe_execute("load_data", logger, ErrorHandler(logger))
    def load_data(self, data_path: str, force_rebuild: bool = False, hadm_id: Optional[int] = None):
        """
        Load discharge summaries and create embeddings

        Args:
          data_path: Path to processed discharge summaries CSV
          force_rebuild: Force rebuild embeddings even if cache exists
          hadm_id: Optional HADM ID to load only a single patient's record
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Loading data from {data_path}")
        self.df = pd.read_csv(data_path)

        # Filter to single patient if HADM ID provided
        if hadm_id is not None:
            hadm_id_float = float(hadm_id)
            filtered_df = self.df[self.df["hadm_id"] == hadm_id_float]
            if len(filtered_df) == 0:
                raise ValueError(f"No record found with HADM ID: {hadm_id}")
            self.df = filtered_df
            logger.info(f"Filtered to single patient record (HADM ID: {hadm_id})")
        else:
            logger.info(f"Loaded {len(self.df)} records")

        # Check for cached embeddings (include HADM ID in cache filename if filtering)
        cache_suffix = f"_{hadm_id}" if hadm_id else ""
        cache_file = self.embeddings_cache_dir / f"embeddings_{data_path.stem}{cache_suffix}.pkl"

        if cache_file.exists() and not force_rebuild:
            logger.info(f"Loading cached embeddings from {cache_file}")
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    self.chunks = cached_data["chunks"]
                    self.metadata = cached_data["metadata"]
                    embeddings = cached_data["embeddings"]

                    # Rebuild index
                    self._build_index(embeddings)
                    logger.info(f"Loaded {len(self.chunks)} chunks from cache")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Rebuilding embeddings...")

        # Create chunks and embeddings
        logger.info("Creating document chunks...")
        self._create_chunks()

        if not self.embedding_model:
            logger.error("Cannot create embeddings: embedding model not available")
            return

        logger.info(f"Generating embeddings for {len(self.chunks)} chunks...")
        embeddings = self._generate_embeddings()

        # Keep a copy so the caller can persist/re-hydrate without re-embedding.
        self._last_embeddings = embeddings

        # Build index
        self._build_index(embeddings)

        # Cache embeddings
        logger.info(f"Caching embeddings to {cache_file}")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"chunks": self.chunks, "metadata": self.metadata, "embeddings": embeddings}, f)
            logger.info("Embeddings cached successfully")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

    def _create_chunks(self):
        """Create text chunks from discharge summaries"""
        self.chunks = []
        self.metadata = []

        # Text column to use (prefer cleaned_text_final, fallback to cleaned_text)
        text_column = "cleaned_text_final" if "cleaned_text_final" in self.df.columns else "cleaned_text"

        for idx, row in self.df.iterrows():
            hadm_id = row.get("hadm_id", idx)
            subject_id = row.get("subject_id", None)
            text = str(row.get(text_column, ""))

            if not text or pd.isna(text):
                continue

            # Split text into chunks
            text_chunks = self._split_text(text)

            for chunk_idx, chunk in enumerate(text_chunks):
                self.chunks.append(chunk)
                self.metadata.append(
                    {
                        "hadm_id": hadm_id,
                        "subject_id": subject_id,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(text_chunks),
                        "record_index": idx,
                        "age": row.get("age_at_admission", None),
                        "gender": row.get("gender", None),
                        "diagnosis_count": row.get("diagnosis_count", None),
                    }
                )

        logger.info(f"Created {len(self.chunks)} chunks from {len(self.df)} records")

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks

        Args:
          text: Input text

        Returns:
          List of text chunks
        """
        # Simple chunking by sentences and character count
        # Split by sentences first
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, start new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ". ".join(current_chunk) + "."
                chunks.append(chunk_text)

                # Start new chunk with overlap (last N sentences from previous chunk)
                overlap_sentences = current_chunk[-min(5, len(current_chunk)) :] if self.chunk_overlap > 0 else []
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 2  # +2 for '. '

        # Add final chunk
        if current_chunk:
            chunk_text = ". ".join(current_chunk)
            if not chunk_text.endswith("."):
                chunk_text += "."
            chunks.append(chunk_text)

        return chunks

    def _generate_embeddings(self) -> np.ndarray:
        """Generate embeddings for all chunks"""
        if not self.embedding_model:
            raise ValueError("Embedding model not initialized")

        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True, batch_size=32, convert_to_numpy=True)

        return embeddings

    def _build_index(self, embeddings: np.ndarray):
        """Build vector index for similarity search"""
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Cannot build index: embeddings are empty")

        if FAISS_AVAILABLE:
            # Use FAISS for efficient similarity search
            try:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings.astype("float32"))

                logger.info(f"Built FAISS index with {self.index.ntotal} vectors (dimension: {dimension})")
            except Exception as e:
                logger.error(f"Failed to build FAISS index: {e}", exc_info=True)
                raise ValueError(f"Failed to build FAISS index: {e}")
        else:
            # Use numpy for similarity search (slower but works)
            try:
                # Normalize embeddings for cosine similarity
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                self.embeddings_normalized = embeddings / norms
                logger.info(
                    f"Using numpy-based similarity search with {len(embeddings)} vectors (dimension: {embeddings.shape[1]})"
                )
            except Exception as e:
                logger.error(f"Failed to build numpy-based index: {e}", exc_info=True)
                raise ValueError(f"Failed to build numpy-based index: {e}")

    def export_cached_index(self) -> Dict[str, Any]:
        """
        Export the currently loaded chunks + embeddings + metadata for persistence.
        """
        if not self.chunks or not self.metadata:
            return {"chunks": [], "embeddings": [], "metadata": []}
        embeddings = getattr(self, "_last_embeddings", None)
        if embeddings is None and self.embedding_model:
            # Fallback: compute embeddings if not cached (should be rare)
            embeddings = self.embedding_model.encode(self.chunks, convert_to_numpy=True)
        if embeddings is None:
            return {"chunks": self.chunks, "embeddings": [], "metadata": self.metadata}
        return {
            "chunks": list(self.chunks),
            "embeddings": embeddings.tolist(),
            "metadata": list(self.metadata),
        }

    def load_cached_index(self, chunks: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]]) -> None:
        """
        Load precomputed chunks/embeddings and rebuild the vector index (no re-embedding).
        """
        if not chunks or not embeddings:
            self.chunks = []
            self.metadata = []
            self.index = None
            if hasattr(self, "embeddings_normalized"):
                delattr(self, "embeddings_normalized")
            return

        arr = np.asarray(embeddings, dtype=np.float32)
        self.chunks = list(chunks)
        self.metadata = list(metadata) if metadata else [{} for _ in self.chunks]
        self._last_embeddings = arr
        self._build_index(arr)

    def retrieve(self, query: str, k: int = 5, hadm_id_filter: Optional[int] = None, min_score: float = 0.0) -> List[Dict]:
        """
        Retrieve relevant chunks for a query

        Args:
          query: Patient question
          k: Number of chunks to retrieve
          hadm_id_filter: Filter by specific HADM ID (optional)
          min_score: Minimum similarity score threshold

        Returns:
          List of relevant chunks with metadata and scores
        """
        # Enhanced validation with detailed error messages
        if not self.embedding_model:
            error_details = "Embedding model is not initialized."
            logger.error(f"RAG system not initialized: {error_details}")
            raise ValueError(f"RAG system not initialized. {error_details} Load data first.")

        if not self.chunks:
            error_details = "No document chunks available."
            logger.error(f"RAG system not initialized: {error_details}")
            raise ValueError(f"RAG system not initialized. {error_details} Load data first.")

        # Check index status
        has_faiss_index = FAISS_AVAILABLE and self.index is not None
        has_numpy_index = hasattr(self, "embeddings_normalized") and len(self.embeddings_normalized) > 0

        if not has_faiss_index and not has_numpy_index:
            error_details = f"Index not built. FAISS available: {FAISS_AVAILABLE}, FAISS index exists: {self.index is not None if FAISS_AVAILABLE else 'N/A'}, Numpy index exists: {has_numpy_index}, Chunks: {len(self.chunks)}"
            logger.error(f"RAG system not initialized: {error_details}")
            raise ValueError(f"RAG system not initialized. {error_details} Load data first.")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Search in index
        if FAISS_AVAILABLE and self.index is not None:
            # FAISS search
            query_embedding = query_embedding.reshape(1, -1).astype("float32")
            search_k = min(k * 2, max(self.index.ntotal, 1))  # Ensure at least 1
            scores, indices = self.index.search(query_embedding, search_k)
            scores = scores[0]
            indices = indices[0]
        else:
            # Numpy-based search
            if not hasattr(self, "embeddings_normalized") or len(self.embeddings_normalized) == 0:
                logger.error("No embeddings available for search")
                return []
            scores = np.dot(self.embeddings_normalized, query_embedding)
            search_k = min(k * 2, len(scores))
            indices = np.argsort(scores)[::-1][:search_k]
            scores = scores[indices]

        logger.debug(f"Search results: {len(scores)} scores, range: [{scores.min():.3f}, {scores.max():.3f}]")

        # Filter and format results
        results = []
        scores_list = []
        for score, idx in zip(scores, indices):
            scores_list.append(float(score))

            # Filter by HADM ID if specified
            if hadm_id_filter is not None:
                if self.metadata[idx].get("hadm_id") != hadm_id_filter:
                    continue

            # Filter by minimum score
            if score < min_score:
                continue

            results.append(
                {"chunk": self.chunks[idx], "metadata": self.metadata[idx], "score": float(score), "chunk_index": idx}
            )

            if len(results) >= k:
                break

        # Log scores for debugging
        if scores_list:
            # Never log raw query text (may contain PHI/PII).
            logger.info(
                f"Retrieved {len(results)} chunks for query (redacted from logs) "
                f"(scores: min={min(scores_list):.3f}, max={max(scores_list):.3f}, "
                f"mean={sum(scores_list)/len(scores_list):.3f}, threshold={min_score})"
            )
        else:
            logger.warning(
                f"No scores computed. Chunks available: {len(self.chunks)}, Index status: {self.index is not None if FAISS_AVAILABLE else 'numpy'}"
            )

        return results

    def get_full_record(self, hadm_id: int) -> Optional[Dict]:
        """Get full discharge summary for a specific HADM ID"""
        if self.df is None:
            return None

        record = self.df[self.df["hadm_id"] == hadm_id]
        if len(record) == 0:
            return None

        return record.iloc[0].to_dict()
