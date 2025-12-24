#!/usr/bin/env python3
"""
File-based Q&A System using RAG
Allows users to upload text, PDF, or image files and ask questions about them
"""

import json
import os
import platform
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rag.document_processor import DocumentProcessor
from src.rag.rag_system import RAGSystem
from src.rag.vector_db import CHROMADB_AVAILABLE, VectorDatabase
from src.privacy.redaction import redact_sources, redact_text

# Import FAISS availability check.
# NOTE: On some macOS environments, importing `faiss` can segfault (native binary mismatch).
# We therefore skip FAISS on Darwin by default to keep the app/test suite stable.
if platform.system() == "Darwin":
    faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False
else:
    try:
        import faiss  # type: ignore

        FAISS_AVAILABLE = True
    except ImportError:
        faiss = None  # type: ignore[assignment]
        FAISS_AVAILABLE = False
from src.training.gemini_inference import GeminiInference
from src.utils.error_handling import ErrorHandler, safe_execute
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Try to import medical utilities
try:
    from src.utils.medical_utils import get_medical_simplifier

    MEDICAL_UTILS_AVAILABLE = True
except ImportError:
    MEDICAL_UTILS_AVAILABLE = False
    logger.debug("Medical utilities not available")

# Try to import MedicalSummarizer for document summarization
try:
    # API/backend summarizer implementation lives here in this repo.
    from model_deployment.api.summarizer import MedicalSummarizer

    SUMMARIZER_AVAILABLE = True
    logger.debug("MedicalSummarizer available")
except Exception as e:
    SUMMARIZER_AVAILABLE = False
    logger.debug(f"MedicalSummarizer not available: {e}")

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not available for direct API calls")


class FileQA:
    """
    Q&A system that uses RAG to answer questions about uploaded documents
    Supports text files, PDFs, and images
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_biobert: bool = False,
        gemini_model: str = "gemini-2.0-flash-exp",
        gemini_api_key: Optional[str] = None,
        rag_k: int = 5,
        use_vector_db: bool = True,
        user_id: Optional[str] = None,
        collection_name: str = "file_qa_documents",
        simplify_medical_terms: bool = True,
        privacy_mode: bool = True,
        allow_external_calls: bool = True,
        pii_extra_terms: Optional[List[str]] = None,
    ):
        """
        Initialize File Q&A system

        Args:
          embedding_model: Sentence transformer model for embeddings
          use_biobert: If True, use BioBERT for medical text (better for medical documents)
          gemini_model: Gemini model for answer generation
          gemini_api_key: Optional API key for Gemini (for image analysis)
          rag_k: Number of chunks to retrieve for context
          use_vector_db: Whether to use ChromaDB for persistent storage (default: True)
          user_id: Optional user ID for user-specific collections
          collection_name: Name of the ChromaDB collection
          simplify_medical_terms: If True, simplify medical terms in answers
        """
        self.error_handler = ErrorHandler(logger)
        self.rag_k = rag_k
        self.use_vector_db = use_vector_db and CHROMADB_AVAILABLE
        self.user_id = user_id
        self.simplify_medical_terms = simplify_medical_terms and MEDICAL_UTILS_AVAILABLE
        self.privacy_mode = privacy_mode
        self.allow_external_calls = allow_external_calls
        self.pii_extra_terms = pii_extra_terms or []

        # Optional: call the deployed API backend for summarization if configured.
        self.api_base_url = (os.getenv("LAB_LENS_API_URL") or os.getenv("LAB_LENS_API_BASE_URL") or "").strip()
        if self.api_base_url:
            try:
                health_url = self.api_base_url.rstrip("/") + "/health"
                req = urllib.request.Request(health_url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logger.info(f" LAB_LENS_API_URL set and reachable (GET {health_url} -> {resp.status})")
            except Exception as e:
                logger.warning(f"⚠️ LAB_LENS_API_URL is set but /health check failed: {e}")

        logger.info("Initializing File Q&A system...")

        # Initialize medical term simplifier if enabled
        self.term_simplifier = None
        if self.simplify_medical_terms:
            try:
                self.term_simplifier = get_medical_simplifier()
                logger.info("Medical term simplifier enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize medical term simplifier: {e}")
                self.simplify_medical_terms = False

        # Initialize document processor
        # In privacy mode we disable Gemini Vision (images should not be sent to Gemini).
        self.doc_processor = DocumentProcessor(
            gemini_api_key=gemini_api_key,
            allow_external_calls=self.allow_external_calls,
            allow_gemini_vision=(not self.privacy_mode),
        )

        # Initialize RAG system (without loading data)
        logger.info(f"Initializing RAG system (use_biobert={use_biobert})...")
        self.rag = RAGSystem(embedding_model=embedding_model, use_biobert=use_biobert)

        # Initialize vector database if enabled
        self.vector_db = None
        if self.use_vector_db:
            try:
                # Create embedding function wrapper for ChromaDB
                def embedding_fn(texts):
                    if not self.rag.embedding_model:
                        raise ValueError("Embedding model not initialized")
                    return self.rag.embedding_model.encode(texts, convert_to_numpy=True).tolist()

                # Use user-specific collection if user_id provided
                collection = f"{collection_name}_{user_id}" if user_id else collection_name

                self.vector_db = VectorDatabase(
                    persist_directory="models/vector_db", collection_name=collection, embedding_function=embedding_fn
                )
                logger.info(f"Vector database initialized: {collection}")
            except Exception as e:
                logger.warning(f"Failed to initialize vector database: {e}. Using in-memory storage.")
                self.use_vector_db = False

        # Initialize Gemini for answer generation
        logger.info("Initializing Gemini model for answer generation...")
        try:
            self.gemini = GeminiInference(model_name=gemini_model)
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.gemini = None

        # Initialize MedicalSummarizer for document summarization (lazy loading)
        # Note: Summarizer is optional and won't block RAG system initialization
        self.summarizer = None
        self.summarizer_available = SUMMARIZER_AVAILABLE
        if self.summarizer_available:
            logger.info("MedicalSummarizer will be loaded on first use (optional feature)")
        else:
            logger.info("MedicalSummarizer not available - document summarization will use Gemini fallback")

    def load_file(self, file_path: str) -> Dict[str, any]:
        """
        Load and process a file (text, PDF, or image)

        Args:
          file_path: Path to the file

        Returns:
          Dictionary with processing results
        """
        logger.info(f"Processing file: {file_path}")

        # Process the file
        document = self.doc_processor.process_file(file_path)

        # Load into RAG system
        try:
            self.rag.load_custom_documents([document])

            # Validate RAG system is ready
            if not self.rag.chunks:
                error_msg = "RAG system loaded but no chunks were created"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "file_name": document["file_name"]}

            # Check if index is built
            has_faiss = hasattr(self.rag, "index") and self.rag.index is not None
            has_numpy = hasattr(self.rag, "embeddings_normalized") and len(self.rag.embeddings_normalized) > 0

            if not has_faiss and not has_numpy:
                error_msg = f"RAG system loaded chunks but index was not built"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "file_name": document["file_name"]}

            logger.info(f"File loaded and ready for Q&A: {document['file_name']} ({len(self.rag.chunks)} chunks)")
        except Exception as e:
            logger.error(f"Failed to load file into RAG: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to load file into RAG system: {str(e)}",
                "file_name": document.get("file_name", "unknown"),
            }

        return {
            "success": True,
            "file_name": document["file_name"],
            "file_type": document["file_type"],
            "text_length": len(document["text"]),
            "num_chunks": len(self.rag.chunks),
            "preview": document["text"][:200] + "..." if len(document["text"]) > 200 else document["text"],
        }

    def load_text(self, text: str, source_name: str = "user_input") -> Dict[str, any]:
        """
        Load raw text content

        Args:
          text: Raw text content
          source_name: Name/identifier for the source

        Returns:
          Dictionary with loading results
        """
        logger.info(f"Processing text input: {source_name}")

        # Process the text
        document = self.doc_processor.process_text(text, source_name)

        # Load into RAG system
        try:
            self.rag.load_custom_documents([document])

            # Validate RAG system is ready
            if not self.rag.chunks:
                error_msg = "RAG system loaded but no chunks were created"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "source_name": source_name}

            # Check if index is built
            has_faiss = hasattr(self.rag, "index") and self.rag.index is not None
            has_numpy = hasattr(self.rag, "embeddings_normalized") and len(self.rag.embeddings_normalized) > 0

            if not has_faiss and not has_numpy:
                error_msg = f"RAG system loaded chunks but index was not built"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "source_name": source_name}

            logger.info(f"Text loaded and validated: {source_name} ({len(self.rag.chunks)} chunks)")
        except Exception as e:
            logger.error(f"Failed to load text into RAG: {e}", exc_info=True)
            return {"success": False, "error": f"Failed to load text into RAG system: {str(e)}", "source_name": source_name}

        # Persist to vector database if enabled
        if self.use_vector_db and self.vector_db:
            try:
                if self.rag.embedding_model and self.rag.chunks:
                    # Use the embedding model directly
                    embeddings = self.rag.embedding_model.encode(self.rag.chunks, convert_to_numpy=True)
                    embeddings_list = embeddings.tolist()

                    self.vector_db.add_documents(
                        texts=self.rag.chunks, embeddings=embeddings_list, metadatas=self.rag.metadata
                    )
                    logger.info(f"Persisted {len(self.rag.chunks)} chunks to vector database")
            except Exception as e:
                logger.warning(f"Failed to persist to vector database: {e}")

        logger.info(f"Text loaded and ready for Q&A: {source_name}")

        return {
            "success": True,
            "source_name": source_name,
            "text_length": len(text),
            "num_chunks": len(self.rag.chunks),
            "preview": text[:200] + "..." if len(text) > 200 else text,
            "persisted": self.use_vector_db and self.vector_db is not None,
        }

    def load_multiple_files(self, file_paths: List[str]) -> Dict[str, any]:
        """
        Load and process multiple files

        Args:
          file_paths: List of file paths

        Returns:
          Dictionary with processing results
        """
        logger.info(f"Processing {len(file_paths)} files")

        # Process all files and collect errors
        documents = []
        errors = []

        for file_path in file_paths:
            try:
                result = self.doc_processor.process_file(file_path)
                if result and result.get("text"):
                    documents.append(result)
                else:
                    errors.append(f"{Path(file_path).name}: No text extracted")
            except Exception as e:
                error_msg = f"{Path(file_path).name}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"Failed to process {file_path}: {e}", exc_info=True)

        if not documents:
            error_summary = "; ".join(errors) if errors else "Unknown error"
            return {"success": False, "error": f"No documents could be processed. Errors: {error_summary}"}

        # Load into RAG system
        try:
            self.rag.load_custom_documents(documents)

            # Validate that RAG system is properly initialized
            if not self.rag.chunks:
                error_msg = "RAG system loaded but no chunks were created"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            # Check if index is built
            has_faiss = hasattr(self.rag, "index") and self.rag.index is not None
            has_numpy = hasattr(self.rag, "embeddings_normalized") and len(self.rag.embeddings_normalized) > 0

            if not has_faiss and not has_numpy:
                error_msg = f"RAG system loaded chunks but index was not built. Chunks: {len(self.rag.chunks)}, Embedding model: {self.rag.embedding_model is not None}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}

            logger.info(f"RAG system validated: {len(self.rag.chunks)} chunks, index ready")
        except Exception as e:
            logger.error(f"Failed to load documents into RAG: {e}", exc_info=True)
            return {"success": False, "error": f"Failed to load documents into RAG system: {str(e)}"}

        # Persist to vector database if enabled
        if self.use_vector_db and self.vector_db:
            try:
                # Generate embeddings for all chunks
                if self.rag.embedding_model and self.rag.chunks:
                    # Use the embedding model directly
                    embeddings = self.rag.embedding_model.encode(self.rag.chunks, convert_to_numpy=True)

                    # Convert embeddings to list of lists for ChromaDB
                    embeddings_list = embeddings.tolist()

                    # Add to vector database
                    self.vector_db.add_documents(
                        texts=self.rag.chunks, embeddings=embeddings_list, metadatas=self.rag.metadata
                    )
                    logger.info(f"Persisted {len(self.rag.chunks)} chunks to vector database")
            except Exception as e:
                logger.warning(f"Failed to persist to vector database: {e}. Continuing with in-memory storage.")

        logger.info(f"Loaded {len(documents)} files with {len(self.rag.chunks)} total chunks")

        result = {
            "success": True,
            "num_files": len(documents),
            "num_chunks": len(self.rag.chunks),
            "files": [doc["file_name"] for doc in documents],
            "persisted": self.use_vector_db and self.vector_db is not None,
        }

        if errors:
            result["warnings"] = errors

        return result

    @safe_execute("ask_question", logger, ErrorHandler(logger))
    def ask_question(self, question: str) -> Dict:
        """
        Answer a question about the loaded document(s)

        Args:
          question: User's question

        Returns:
          Dictionary with answer and metadata
        """
        if not self.gemini:
            return {
                "answer": "Sorry, the answer generation system is not available.",
                "error": "Gemini model not initialized",
                "question": question,
            }

        if not self.rag.chunks:
            return {
                "answer": "No documents loaded. Please load a file first.",
                "error": "No documents loaded",
                "question": question,
            }

        # Strict local mode: do not call external LLMs at all.
        if not self.allow_external_calls:
            try:
                retrieved = self.rag.retrieve(query=question, k=min(self.rag_k, 5), min_score=0.0)
            except Exception as e:
                return {"answer": f"Local mode: retrieval failed ({e})", "error": str(e), "question": question}

            safe_sources = redact_sources(retrieved, extra_terms=self.pii_extra_terms) if self.privacy_mode else retrieved
            if not safe_sources:
                return {
                    "answer": "Local mode: no matching excerpts found in the loaded documents.",
                    "sources": [],
                    "question": question,
                    "answer_source": "local_excerpts",
                }

            excerpts = "\n\n".join([f"- {s.get('chunk','')[:600]}..." for s in safe_sources[:3]])
            return {
                "answer": (
                    "Local-only mode is enabled (no external AI calls). Here are the most relevant excerpts I found:\n\n"
                    + excerpts
                ),
                "sources": safe_sources,
                "question": question,
                "answer_source": "local_excerpts",
            }

        # Validate RAG system is fully initialized before retrieving
        if not self.rag.embedding_model:
            error_msg = "RAG system embedding model not initialized. Please reload documents."
            logger.error(error_msg)
            return {"answer": f"System error: {error_msg}", "error": error_msg, "question": question}

        # Check if index is built
        has_faiss = FAISS_AVAILABLE and hasattr(self.rag, "index") and self.rag.index is not None
        has_numpy = hasattr(self.rag, "embeddings_normalized") and len(self.rag.embeddings_normalized) > 0

        if not has_faiss and not has_numpy:
            error_msg = f"RAG system index not built. Chunks: {len(self.rag.chunks)}, Embedding model: {self.rag.embedding_model is not None}, FAISS available: {FAISS_AVAILABLE}. Please reload documents."
            logger.error(error_msg)
            return {
                "answer": f"System error: RAG index not initialized. Please reload documents.",
                "error": error_msg,
                "question": question,
            }

        # Never log raw user text in case it contains PHI/PII.
        logger.info("Processing user question (content redacted from logs)")
        logger.info(f"RAG system ready: {len(self.rag.chunks)} chunks, index built: {has_faiss or has_numpy}")

        # Retrieve relevant chunks
        logger.info(f"Retrieving relevant information... (Total chunks available: {len(self.rag.chunks)})")

        # Try with lower threshold first, then progressively lower it
        retrieved_chunks = []
        thresholds = [0.3, 0.2, 0.1, 0.0]

        try:
            for threshold in thresholds:
                retrieved_chunks = self.rag.retrieve(query=question, k=self.rag_k, min_score=threshold)
                if retrieved_chunks:
                    logger.info(f"Retrieved {len(retrieved_chunks)} chunks with threshold {threshold}")
                    break
        except ValueError as e:
            error_msg = f"RAG retrieval failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"answer": f"System error: {error_msg}. Please reload documents.", "error": error_msg, "question": question}

        # Determine if we have good document matches
        has_good_matches = False
        document_context = ""

        if retrieved_chunks:
            # Check if we have good quality matches (score > 0.2)
            good_chunks = [c for c in retrieved_chunks if c.get("score", 0) > 0.2]
            if good_chunks:
                has_good_matches = True
                # Build context from retrieved chunks
                context_parts = []
                for i, chunk_data in enumerate(good_chunks, 1):
                    chunk = chunk_data["chunk"]
                    metadata = chunk_data["metadata"]
                    source_info = f"[Source {i}: {metadata.get('document_name', 'Document')}]"
                    context_parts.append(f"{source_info}\n{chunk}")
                document_context = "\n\n".join(context_parts)
            else:
                # Low quality matches - use them but also allow general knowledge
                context_parts = []
                for i, chunk_data in enumerate(retrieved_chunks[:3], 1):
                    chunk = chunk_data["chunk"]
                    metadata = chunk_data["metadata"]
                    source_info = f"[Source {i}: {metadata.get('document_name', 'Document')}]"
                    context_parts.append(f"{source_info}\n{chunk}")
                document_context = "\n\n".join(context_parts)

        # If no chunks found, try to get any chunks for context
        if not retrieved_chunks and len(self.rag.chunks) > 0:
            # Get a few random chunks for context about the document
            import random

            sample_size = min(3, len(self.rag.chunks))
            sample_indices = random.sample(range(len(self.rag.chunks)), sample_size)
            context_parts = []
            for idx in sample_indices:
                chunk = self.rag.chunks[idx]
                metadata = self.rag.metadata[idx]
                source_info = f"[From document: {metadata.get('document_name', 'Document')}]"
                context_parts.append(f"{source_info}\n{chunk}")
            document_context = "\n\n".join(context_parts)
            logger.info("Using sample document chunks for context (no direct matches found)")

        # Build the prompt using redacted content (privacy mode).
        # We keep retrieval on the original question (local), but redact before sending to Gemini.
        question_for_llm = question
        if self.privacy_mode:
            question_for_llm = redact_text(question, extra_terms=self.pii_extra_terms).text

        document_context_for_llm = document_context
        if self.privacy_mode and document_context:
            document_context_for_llm = redact_text(document_context, extra_terms=self.pii_extra_terms).text

        # Generate answer using Gemini with hybrid approach
        logger.info("Generating answer using Gemini (hybrid: document + general knowledge)...")
        try:
            # Create a prompt that uses document if available, but allows general knowledge
            if document_context and has_good_matches:
                # Strong document match - prioritize document
                prompt = f"""You are a helpful medical assistant. The user is asking about their medical report.
IMPORTANT PRIVACY RULES:
- Do NOT output any personal identifiers (names, emails, phone numbers, MRN/Patient IDs, addresses, exact dates of birth).
- Some content may already be redacted like [REDACTED_*]. Do not attempt to reconstruct it.
- If an answer would require personal identifiers, respond without them.

Answer based on the document content below. If the document doesn't contain enough information, you may supplement with general medical knowledge, but clearly indicate what comes from the document vs. general knowledge.

DOCUMENT CONTENT:
{document_context_for_llm}

USER QUESTION: {question_for_llm}

ANSWER:"""
            elif document_context:
                # Weak document match - use document as context but allow general knowledge
                prompt = f"""You are a helpful medical assistant. The user has asked a question about their medical report.
IMPORTANT PRIVACY RULES:
- Do NOT output any personal identifiers (names, emails, phone numbers, MRN/Patient IDs, addresses, exact dates of birth).
- Some content may already be redacted like [REDACTED_*]. Do not attempt to reconstruct it.

The document content below may or may not directly answer the question. Please:
1. First try to answer based on the document if relevant
2. If the document doesn't contain the answer, use your general medical knowledge to help
3. Clearly indicate when you're using general knowledge vs. document information

DOCUMENT CONTENT (may be partially relevant):
{document_context_for_llm}

USER QUESTION: {question_for_llm}

ANSWER:"""
            else:
                # No document context - use general knowledge
                prompt = f"""You are a helpful medical assistant.
IMPORTANT PRIVACY RULES:
- Do NOT output any personal identifiers (names, emails, phone numbers, MRN/Patient IDs, addresses, exact dates of birth).
- Do not ask the user to provide personal identifiers.

The user has asked a medical question. Please answer using general medical knowledge. Be clear, accurate, and helpful.

USER QUESTION: {question_for_llm}

ANSWER:"""

            # Generate response using Gemini
            if GEMINI_AVAILABLE:
                # Use direct API call for more control
                model = genai.GenerativeModel(self.gemini.model.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,  # Slightly higher for more natural responses
                        max_output_tokens=2048,
                    ),
                )
                answer = response.text.strip()
            else:
                # Fallback to existing method
                if document_context:
                    answer = self.gemini.model.answer_question(question, document_context)
                else:
                    # No context - create a simple prompt
                    simple_prompt = f"Answer this medical question: {question}"
                    model = self.gemini.model.model
                    response = model.generate_content(simple_prompt)
                    answer = response.text.strip()

            logger.info("Answer generated successfully")

            # Apply medical term simplification if enabled
            if self.simplify_medical_terms and self.term_simplifier and answer:
                try:
                    original_answer = answer
                    answer = self.term_simplifier.simplify_text(answer, aggressive=False)
                    logger.debug("Applied medical term simplification to answer")
                except Exception as e:
                    logger.warning(f"Failed to simplify medical terms: {e}")
                    # Continue with original answer

            # Determine answer source
            if has_good_matches:
                answer_source = "document"
            elif retrieved_chunks:
                answer_source = "document_and_knowledge"
            else:
                answer_source = "general_knowledge"

            safe_sources = (
                redact_sources(retrieved_chunks, extra_terms=self.pii_extra_terms) if self.privacy_mode else retrieved_chunks
            )

            return {
                "answer": answer,
                "sources": safe_sources if safe_sources else [],
                "question": question,
                "num_sources": len(safe_sources) if safe_sources else 0,
                "answer_source": answer_source,  # Indicates where answer came from
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"I encountered an error while generating an answer: {e}. Please try rephrasing your question.",
                "error": str(e),
                "sources": retrieved_chunks,
                "question": question,
            }

    def _get_summarizer(self):
        """Lazy load MedicalSummarizer with graceful error handling"""
        if not self.summarizer_available:
            return None

        if self.summarizer is None:
            try:
                logger.info("Loading MedicalSummarizer for document summarization...")
                # Try to load with CPU (required for Cloud Run)
                # The summarizer will handle BioBERT fallback internally
                self.summarizer = MedicalSummarizer(use_gpu=False)  # Use CPU for web app compatibility
                logger.info(" MedicalSummarizer loaded successfully")
            except ImportError as e:
                logger.error(f"Import error loading MedicalSummarizer: {e}. Missing dependencies?")
                self.summarizer_available = False
                return None
            except Exception as e:
                logger.error(f"Failed to load MedicalSummarizer: {e}", exc_info=True)
                # Don't disable summarizer_available permanently - might be a temporary issue
                # Just return None for this attempt
                logger.warning("MedicalSummarizer failed to load, but will retry on next use")
                return None

        return self.summarizer

    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF extraction artifacts like (cid:X) codes"""
        import re

        # Remove (cid:X) patterns - these are font encoding artifacts
        cleaned = re.sub(r"\(cid:\d+\)", " ", text)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def summarize_document(self, text: Optional[str] = None) -> Dict:
        """
        Generate a summary of the loaded document(s) using Gemini (preferred) or MedicalSummarizer

        Args:
          text: Optional text to summarize. If None, uses all loaded document chunks.

        Returns:
          Dictionary with summary and metadata
        """
        # Get text to summarize
        if text is None:
            if not self.rag.chunks:
                return {"success": False, "error": "No documents loaded to summarize", "summary": None}
            # Combine all chunks
            text = "\n\n".join(self.rag.chunks)

        if not text or not text.strip():
            return {"success": False, "error": "No text content to summarize", "summary": None}

        # Clean PDF extraction artifacts
        text = self._clean_pdf_text(text)

        # Strict local mode: do not call external LLMs at all.
        if not self.allow_external_calls:
            sample = "\n\n".join(self.rag.chunks[:6])
            sample = self._clean_pdf_text(sample)
            if self.privacy_mode:
                sample = redact_text(sample, extra_terms=self.pii_extra_terms).text
            return {
                "success": True,
                "summary": "Local-only mode is enabled (no external AI calls). Showing a redacted excerpt of the document:\n\n"
                + sample[:4000],
                "raw_summary": "",
                "extracted_data": {},
                "error": None,
            }

        # Try using Gemini for summarization first (more reliable)
        try:
            safe_text = text[:15000]
            if self.privacy_mode:
                safe_text = redact_text(safe_text, extra_terms=self.pii_extra_terms).text

            summary_prompt = f"""You are a helpful medical assistant.
IMPORTANT PRIVACY RULES:
- Do NOT output any personal identifiers (names, emails, phone numbers, MRN/Patient IDs, addresses, exact dates of birth).
- Some content may be redacted like [REDACTED_*]. Do not attempt to reconstruct it.

Provide a comprehensive summary of this medical document. Focus on:
1. Key diagnoses and findings
2. Important test results or measurements
3. Recommendations or treatment plans
4. Any critical alerts or abnormal values

Document (redacted if needed):
{safe_text}

Summary:"""

            if GEMINI_AVAILABLE:
                model = genai.GenerativeModel(self.gemini.model.model_name)
                response = model.generate_content(
                    summary_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.4,
                        max_output_tokens=2048,
                    ),
                )
                answer = (response.text or "").strip()
            else:
                model = self.gemini.model.model
                response = model.generate_content(summary_prompt)
                answer = (response.text or "").strip()

            if answer:
                return {
                    "success": True,
                    "summary": answer,
                    "raw_summary": "",
                    "extracted_data": {},
                    "error": None,
                }
        except Exception as e:
            logger.warning(f"Gemini summarization failed, trying alternatives: {e}")

        # If an API backend is configured, use it (preferred on Cloud Run to keep the
        # Streamlit container lightweight and avoid loading large ML models).
        api_base = (os.getenv("LAB_LENS_API_URL") or os.getenv("LAB_LENS_API_BASE_URL") or "").strip()
        if api_base:
            try:
                url = api_base.rstrip("/") + "/summarize"
                payload = json.dumps({"text": text}).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                logger.info(f"Calling Lab Lens API summarizer at {url}")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = resp.read().decode("utf-8")
                    data = json.loads(body) if body else {}

                # API returns: {summary, diagnosis, bart_summary}
                return {
                    "success": True,
                    "summary": data.get("summary", ""),
                    "raw_summary": data.get("bart_summary", ""),
                    "extracted_data": {"diagnosis": data.get("diagnosis", "")},
                    "error": None,
                }
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as e:
                logger.warning(f"API summarization failed ({api_base}); falling back to local summarizer: {e}")
            except Exception as e:
                logger.warning(f"Unexpected API summarization error; falling back to local summarizer: {e}", exc_info=True)

        summarizer = self._get_summarizer()
        if not summarizer:
            return {"success": False, "error": "MedicalSummarizer not available", "summary": None}

        try:
            logger.info("Generating document summary using MedicalSummarizer...")
            result = summarizer.generate_summary(text)

            return {
                "success": True,
                "summary": result.get("final_summary", ""),
                "raw_summary": result.get("raw_bart_summary", ""),
                "extracted_data": result.get("extracted_data", {}),
                "error": None,
            }
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"success": False, "error": str(e), "summary": None}
