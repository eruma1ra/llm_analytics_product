from __future__ import annotations

import time
from typing import Any, Optional

import streamlit as st


CACHE_TTL_SECONDS = 24 * 60 * 60


@st.cache_resource
def _history_cache() -> dict[str, dict[str, Any]]:
    # Кэш хранится в памяти процесса, отдельно по user_id.
    return {}


def cleanup_expired_cache(ttl_seconds: int = CACHE_TTL_SECONDS) -> None:
    now = time.time()
    cache = _history_cache()
    stale_users = [
        user_id
        for user_id, payload in cache.items()
        if now - float(payload.get("updated_ts", 0.0)) > ttl_seconds
    ]
    for user_id in stale_users:
        cache.pop(user_id, None)


def persist_user_cache(
    user_id: str,
    conversations: list[dict[str, Any]],
    active_conversation_id: Optional[str],
) -> None:
    cache = _history_cache()
    cache[user_id] = {
        "updated_ts": time.time(),
        "conversations": conversations,
        "active_conversation_id": active_conversation_id,
    }


def restore_user_cache(
    user_id: str,
    ttl_seconds: int = CACHE_TTL_SECONDS,
) -> Optional[dict[str, Any]]:
    payload = _history_cache().get(user_id)
    if not payload:
        return None

    if time.time() - float(payload.get("updated_ts", 0.0)) > ttl_seconds:
        _history_cache().pop(user_id, None)
        return None

    return payload
