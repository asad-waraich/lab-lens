from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from src.utils.logging_config import get_logger
from src.privacy.redaction import sanitize_filename

logger = get_logger(__name__)


@dataclass(frozen=True)
class UploadedObject:
    bucket: str
    blob_name: str

    @property
    def gs_uri(self) -> str:
        return f"gs://{self.bucket}/{self.blob_name}"


class GCSStore:
    def __init__(self, bucket_name: Optional[str] = None):
        from google.cloud import storage  # type: ignore

        self._client = storage.Client()
        self._bucket_name = bucket_name or os.getenv("GCS_BUCKET_USER_UPLOADS") or ""
        if not self._bucket_name:
            raise ValueError("GCS_BUCKET_USER_UPLOADS env var is required for GCS uploads")
        self._bucket = self._client.bucket(self._bucket_name)

    def upload_bytes(
        self, *, uid: str, chat_id: str, filename: str, data: bytes, content_type: Optional[str] = None
    ) -> UploadedObject:
        # Never persist potentially identifying filenames in object paths.
        safe_name = sanitize_filename(filename).replace("/", "_")
        blob_name = f"{uid}/{chat_id}/{safe_name}"
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        logger.info(f"Uploaded file to {self._bucket_name}/{blob_name}")
        return UploadedObject(bucket=self._bucket_name, blob_name=blob_name)
