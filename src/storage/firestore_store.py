from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FirestoreStore:
    """
    Firestore persistence for per-user chats, messages, and RAG chunks/embeddings.
    """

    def __init__(self, project_id: Optional[str] = None):
        from google.cloud import firestore  # type: ignore

        self._firestore = firestore
        self._project_id = project_id or os.getenv("FIRESTORE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
        self._client = firestore.Client(project=self._project_id)

    # ---- Users ----
    def upsert_user(self, uid: str, email: Optional[str], name: Optional[str], picture: Optional[str]) -> None:
        doc = self._client.collection("users").document(uid)
        doc.set(
            {
                "email": email,
                "name": name,
                "picture": picture,
                "last_login": _utcnow_iso(),
            },
            merge=True,
        )

    # ---- Chats ----
    def create_chat(self, uid: str, chat_id: str, title: str) -> None:
        chat_ref = self._client.collection("users").document(uid).collection("chats").document(chat_id)
        now = _utcnow_iso()
        chat_ref.set(
            {
                "title": title,
                "created_at": now,
                "updated_at": now,
                "doc_count": 0,
                "chunk_count": 0,
            },
            merge=True,
        )

    def update_chat(self, uid: str, chat_id: str, **fields: Any) -> None:
        if "updated_at" not in fields:
            fields["updated_at"] = _utcnow_iso()
        chat_ref = self._client.collection("users").document(uid).collection("chats").document(chat_id)
        chat_ref.set(fields, merge=True)

    def list_chats(self, uid: str, limit: int = 50) -> List[Dict[str, Any]]:
        chats_ref = self._client.collection("users").document(uid).collection("chats")
        query = chats_ref.order_by("updated_at", direction=self._firestore.Query.DESCENDING).limit(limit)
        out: List[Dict[str, Any]] = []
        for doc in query.stream():
            data = doc.to_dict() or {}
            data["chat_id"] = doc.id
            out.append(data)
        return out

    # ---- Messages ----
    def add_message(
        self,
        uid: str,
        chat_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ts: Optional[str] = None,
    ) -> str:
        messages_ref = (
            self._client.collection("users").document(uid).collection("chats").document(chat_id).collection("messages")
        )
        doc = messages_ref.document()
        doc.set(
            {
                "role": role,
                "content": content,
                "sources": sources or [],
                "ts": ts or _utcnow_iso(),
            }
        )
        # keep chat updated
        self.update_chat(uid, chat_id)
        return doc.id

    def list_messages(self, uid: str, chat_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        messages_ref = (
            self._client.collection("users").document(uid).collection("chats").document(chat_id).collection("messages")
        )
        query = messages_ref.order_by("ts", direction=self._firestore.Query.ASCENDING).limit(limit)
        out: List[Dict[str, Any]] = []
        for doc in query.stream():
            data = doc.to_dict() or {}
            data["message_id"] = doc.id
            out.append(data)
        return out

    # ---- Chunks/Embeddings ----
    def replace_chunks(
        self,
        uid: str,
        chat_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 200,
    ) -> None:
        """
        Replace the entire chunk set for a chat (delete existing chunk docs, then write new ones).
        """
        chunks_ref = self._client.collection("users").document(uid).collection("chats").document(chat_id).collection("chunks")

        # Delete existing chunks (best-effort)
        existing = list(chunks_ref.stream())
        if existing:
            batch = self._client.batch()
            count = 0
            for doc in existing:
                batch.delete(doc.reference)
                count += 1
                if count >= batch_size:
                    batch.commit()
                    batch = self._client.batch()
                    count = 0
            if count:
                batch.commit()

        # Write new chunks
        batch = self._client.batch()
        count = 0
        now = _utcnow_iso()
        for i, (text, emb, meta) in enumerate(zip(chunks, embeddings, metadatas)):
            doc_ref = chunks_ref.document(str(i))
            batch.set(
                doc_ref,
                {
                    "text": text,
                    "embedding": emb,
                    "metadata": meta,
                    "ts": now,
                },
            )
            count += 1
            if count >= batch_size:
                batch.commit()
                batch = self._client.batch()
                count = 0
        if count:
            batch.commit()

        # Update counts
        self.update_chat(uid, chat_id, chunk_count=len(chunks))

    def load_chunks(
        self, uid: str, chat_id: str, limit: int = 5000
    ) -> Tuple[List[str], List[List[float]], List[Dict[str, Any]]]:
        chunks_ref = self._client.collection("users").document(uid).collection("chats").document(chat_id).collection("chunks")
        query = chunks_ref.order_by(self._firestore.FieldPath.document_id(), direction=self._firestore.Query.ASCENDING).limit(
            limit
        )
        chunks: List[str] = []
        embeddings: List[List[float]] = []
        metas: List[Dict[str, Any]] = []
        for doc in query.stream():
            data = doc.to_dict() or {}
            chunks.append(data.get("text", ""))
            embeddings.append(list(data.get("embedding", [])))
            metas.append(dict(data.get("metadata", {})))
        return chunks, embeddings, metas

    # ---- Sessions (refresh persistence) ----
    def upsert_session(self, sid: str, user: Dict[str, Any], exp_ts: int) -> None:
        """
        Persist a browser session id -> user mapping.
        `sid` is a random UUID stored in the URL; `exp_ts` is epoch seconds.
        """
        doc = self._client.collection("sessions").document(sid)
        doc.set(
            {
                "user": user,
                "exp": int(exp_ts),
                "updated_at": _utcnow_iso(),
            },
            merge=True,
        )

    def get_session(self, sid: str) -> Optional[Dict[str, Any]]:
        doc = self._client.collection("sessions").document(sid).get()
        if not doc.exists:
            return None
        return doc.to_dict() or None

    def delete_session(self, sid: str) -> None:
        self._client.collection("sessions").document(sid).delete()
