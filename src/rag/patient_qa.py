#!/usr/bin/env python3
"""
Patient Q&A System using RAG
Allows patients to ask questions about their discharge summaries
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.rag.rag_system import RAGSystem
from src.training.gemini_inference import GeminiInference
from src.privacy.redaction import redact_text
from src.utils.error_handling import ErrorHandler, safe_execute
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PatientQA:
    """
    Patient Q&A system that uses RAG to answer questions about discharge summaries
    """

    def __init__(
        self,
        data_path: str = "data_pipeline/data/processed/processed_discharge_summaries.csv",
        embedding_model: str = "all-MiniLM-L6-v2",
        gemini_model: str = "gemini-2.0-flash-exp",
        rag_k: int = 5,
        hadm_id: Optional[int] = None,
    ):
        """
        Initialize Patient Q&A system

        Args:
          data_path: Path to processed discharge summaries CSV
          embedding_model: Sentence transformer model for embeddings
          gemini_model: Gemini model for answer generation
          rag_k: Number of chunks to retrieve for context
          hadm_id: Optional HADM ID to load only a single patient's record (recommended for single-patient Q&A)
        """
        self.error_handler = ErrorHandler(logger)
        self.rag_k = rag_k
        self.hadm_id = hadm_id

        logger.info("Initializing Patient Q&A system...")

        # Initialize RAG system
        logger.info("Loading RAG system...")
        if hadm_id:
            logger.info(f"Single-patient mode: Loading only HADM ID {hadm_id}")
        self.rag = RAGSystem(embedding_model=embedding_model, data_path=data_path, hadm_id=hadm_id)

        # Initialize Gemini for answer generation
        logger.info("Initializing Gemini model for answer generation...")
        try:
            self.gemini = GeminiInference(model_name=gemini_model)
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.gemini = None

    @safe_execute("ask_question", logger, ErrorHandler(logger))
    def ask_question(self, question: str, hadm_id: Optional[int] = None, include_full_context: bool = False) -> Dict:
        """
        Answer a patient's question about their discharge summary

        Args:
          question: Patient's question
          hadm_id: Hospital admission ID (optional, for filtering to specific patient)
          include_full_context: Whether to include full record in context

        Returns:
          Dictionary with answer and metadata
        """
        if not self.gemini:
            return {
                "answer": "Sorry, the answer generation system is not available.",
                "error": "Gemini model not initialized",
                "question": question,
            }

        # Never log raw user text in case it contains PHI/PII.
        logger.info("Processing patient question (content redacted from logs)")

        # Retrieve relevant chunks
        logger.info("Retrieving relevant information...")
        # If we're in single-patient mode, data is already filtered - no need to filter again
        # Otherwise, use the provided hadm_id for filtering
        filter_hadm_id = None if self.hadm_id else hadm_id
        retrieved_chunks = self.rag.retrieve(
            query=question, k=self.rag_k, hadm_id_filter=filter_hadm_id, min_score=0.3  # Minimum similarity threshold
        )

        if not retrieved_chunks:
            return {
                "answer": "I could not find relevant information in your discharge summary to answer this question. Please rephrase your question or contact your healthcare provider.",
                "sources": [],
                "question": question,
                "hadm_id": self.hadm_id or hadm_id,
            }

        # Build context from retrieved chunks
        context_parts = []
        for i, chunk_data in enumerate(retrieved_chunks, 1):
            chunk = chunk_data["chunk"]
            metadata = chunk_data["metadata"]

            context_parts.append(f"[Section {i}]\n{chunk}\n")

        context = "\n".join(context_parts)

        # Optionally include full record context
        record_hadm_id = self.hadm_id or hadm_id
        if include_full_context and record_hadm_id:
            full_record = self.rag.get_full_record(record_hadm_id)
            if full_record:
                record_summary = f"""
[Full Discharge Summary]
Patient Age: {full_record.get('age_at_admission', 'N/A')}
Gender: {full_record.get('gender', 'N/A')}
Diagnoses: {full_record.get('discharge_diagnosis', 'N/A')}
Medications: {full_record.get('discharge_medications', 'N/A')}
"""
                context = record_summary + "\n" + context

        # Generate answer using proper Q&A method
        logger.info("Generating answer using Gemini...")
        try:
            # Use dedicated Q&A method if available, otherwise fallback to summarize
            safe_question = redact_text(question).text
            safe_context = redact_text(context).text

            if hasattr(self.gemini.model, "answer_question"):
                answer = self.gemini.model.answer_question(
                    question=safe_question,
                    context=safe_context,
                    temperature=0.3,  # Lower temperature for more consistent answers
                )
            else:
                # Fallback to summarize method (older implementation)
                prompt = self._create_prompt(safe_question, safe_context)
                answer = self.gemini.model.summarize(prompt, max_length=500)
                answer = self._extract_answer(answer)

            result = {
                "answer": answer,
                "question": question,
                "sources": [
                    {
                        "chunk": (
                            redact_text(
                                chunk_data["chunk"][:200] + "..." if len(chunk_data["chunk"]) > 200 else chunk_data["chunk"]
                            ).text
                        ),
                        "score": chunk_data["score"],
                        "hadm_id": chunk_data["metadata"].get("hadm_id"),
                        "metadata": chunk_data["metadata"],
                    }
                    for chunk_data in retrieved_chunks
                ],
                "num_sources": len(retrieved_chunks),
                "hadm_id": self.hadm_id or hadm_id,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("Answer generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"I encountered an error while generating an answer: {str(e)}. Please try rephrasing your question.",
                "error": str(e),
                "question": question,
                "sources": [chunk_data["metadata"] for chunk_data in retrieved_chunks],
            }

    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create prompt for Gemini to answer question based on context

        Args:
          question: Patient's question
          context: Retrieved context from discharge summary

        Returns:
          Formatted prompt
        """
        prompt = f"""You are a helpful medical assistant helping a patient understand their discharge summary. Answer the patient's question based ONLY on the information provided in their discharge summary below.

Important guidelines:
1. Answer clearly and in simple, patient-friendly language
2. Only use information from the provided discharge summary
3. If the information is not in the summary, say so clearly
4. Do not provide medical advice beyond what's in the summary
5. For medication questions, refer to the medications listed
6. For diagnosis questions, refer to the diagnoses listed
7. Keep your answer concise but complete (2-4 sentences typically)

DISCHARGE SUMMARY CONTEXT:
{context}

PATIENT'S QUESTION: {question}

ANSWER (based only on the discharge summary above):"""

        return prompt

    def _extract_answer(self, text: str) -> str:
        """Extract answer from Gemini response (handle any wrapping)"""
        # Remove common prefixes if present
        prefixes = ["Answer:", "ANSWER:", "Based on the discharge summary:", "According to the discharge summary:"]
        text = text.strip()

        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()

        return text

    def ask_multiple_questions(self, questions: List[str], hadm_id: Optional[int] = None) -> List[Dict]:
        """
        Answer multiple questions

        Args:
          questions: List of patient questions
          hadm_id: Hospital admission ID (optional)

        Returns:
          List of answer dictionaries
        """
        results = []
        for question in questions:
            result = self.ask_question(question, hadm_id)
            results.append(result)

        return results

    def get_record_summary(self, hadm_id: int) -> Optional[Dict]:
        """Get a summary of a patient's discharge record"""
        record = self.rag.get_full_record(hadm_id)
        if not record:
            return None

        return {
            "hadm_id": hadm_id,
            "subject_id": record.get("subject_id"),
            "age": record.get("age_at_admission"),
            "gender": record.get("gender"),
            "diagnosis": record.get("discharge_diagnosis", "N/A"),
            "medications": record.get("discharge_medications", "N/A"),
            "follow_up": record.get("follow_up", "N/A"),
            "text_length": record.get("text_length", 0),
        }
