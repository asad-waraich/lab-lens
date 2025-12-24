from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FirebaseUser:
    uid: str
    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None


_FIREBASE_APP_INITIALIZED = False


def _init_firebase_admin() -> None:
    """
    Initialize firebase-admin once (uses ADC on Cloud Run).

    Supports optional JSON credentials via FIREBASE_SERVICE_ACCOUNT_JSON for local dev.
    """
    global _FIREBASE_APP_INITIALIZED
    if _FIREBASE_APP_INITIALIZED:
        return

    try:
        import firebase_admin
        from firebase_admin import credentials

        project_id = (os.getenv("FIREBASE_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip() or None
        sa_json = (os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or "").strip()

        if sa_json:
            cred = credentials.Certificate(json.loads(sa_json))
            firebase_admin.initialize_app(cred, {"projectId": project_id} if project_id else None)
            logger.info("Initialized firebase-admin with service account JSON")
        else:
            # Cloud Run / local ADC (GOOGLE_APPLICATION_CREDENTIALS)
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {"projectId": project_id} if project_id else None)
            logger.info("Initialized firebase-admin with Application Default Credentials")

        _FIREBASE_APP_INITIALIZED = True
    except Exception as e:
        logger.error(f"Failed to initialize firebase-admin: {e}", exc_info=True)
        raise


def verify_firebase_id_token(id_token: str) -> FirebaseUser:
    """
    Verify Firebase ID token and return a normalized user identity.

    Raises if invalid/expired.
    """
    _init_firebase_admin()
    from firebase_admin import auth  # type: ignore

    decoded: Dict[str, Any] = auth.verify_id_token(id_token)

    uid = decoded.get("uid") or decoded.get("user_id") or decoded.get("sub")
    if not uid:
        raise ValueError("Firebase token missing uid")

    return FirebaseUser(
        uid=str(uid),
        email=decoded.get("email"),
        name=decoded.get("name"),
        picture=decoded.get("picture"),
    )
