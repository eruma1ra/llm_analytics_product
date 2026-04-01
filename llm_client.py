from __future__ import annotations

import json
import os
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()


def _env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"В .env не задано обязательное поле: {name}")
    return value


def _env_int(name: str) -> int:
    try:
        return int(_env(name))
    except ValueError:
        raise ValueError(f"В .env поле {name} должно быть целым числом.")


def _env_float(name: str) -> float:
    try:
        return float(_env(name))
    except ValueError:
        raise ValueError(f"В .env поле {name} должно быть числом.")


def load_llm_config() -> dict[str, Any]:
    # Конфиг читаем строго из .env, без дефолтов в коде.
    return {
        "provider_label": _env("LLM_PROVIDER_LABEL"),
        "api_base_url": _env("LLM_API_BASE_URL"),
        "chat_path": _env("LLM_CHAT_COMPLETIONS_PATH"),
        "model": _env("LLM_MODEL"),
        "timeout_seconds": _env_int("LLM_TIMEOUT_SECONDS"),
        "temperature": _env_float("LLM_TEMPERATURE"),
        "max_tokens": _env_int("LLM_MAX_TOKENS"),
        "max_context_rows": _env_int("LLM_MAX_CONTEXT_ROWS"),
        "max_charts": _env_int("LLM_MAX_CHARTS"),
    }


def _resolve_api_key(api_key: str = "") -> str:
    # Ключ берем из аргумента или из .env, чтобы UI оставался чистым.
    candidate = (api_key or "").strip()
    if candidate:
        return candidate
    return _env("LLM_API_KEY")


def build_dataframe_context(df: pd.DataFrame, max_rows: int) -> dict[str, Any]:
    numeric_cols = [str(col) for col in df.select_dtypes(include="number").columns][:12]
    numeric_summary: dict[str, dict[str, float]] = {}
    if numeric_cols:
        summary = (
            df[numeric_cols]
            .apply(pd.to_numeric, errors="coerce")
            .describe()
            .round(3)
            .fillna(0)
        )
        numeric_summary = {
            str(col): {str(k): float(v) for k, v in values.items()}
            for col, values in summary.to_dict().items()
        }

    missing = df.isna().sum().sort_values(ascending=False).head(12)
    return {
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": [str(col) for col in df.columns],
        "column_info": [
            {
                "name": str(col),
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
            }
            for col in df.columns[:40]
        ],
        "numeric_summary": numeric_summary,
        "missing_by_column": {str(k): int(v) for k, v in missing.items()},
        "sample_rows": df.head(max_rows).fillna("").astype(str).to_dict(orient="records"),
    }


def build_llm_messages(
    user_prompt: str,
    df: Optional[pd.DataFrame],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    if df is None:
        return [
            {
                "role": "system",
                "content": (
                    "Вы аналитический ассистент. Отвечайте обычным текстом на русском языке. "
                    "Без JSON и без кода."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

    context = build_dataframe_context(df, max_rows=config["max_context_rows"])
    return [
        {
            "role": "system",
            "content": (
                "Вы аналитический ассистент по таблицам. "
                "Отвечайте только обычным текстом на русском, без JSON и без markdown-блоков кода. "
                "Давайте короткий, понятный вывод по запросу пользователя."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Запрос пользователя: {user_prompt}\n\n"
                f"Контекст таблицы JSON:\n{json.dumps(context, ensure_ascii=False)}"
            ),
        },
    ]


def _chat_url(config: dict[str, Any]) -> str:
    base = config["api_base_url"].rstrip("/")
    path = config["chat_path"].strip()
    return f"{base}{path if path.startswith('/') else '/' + path}"


def _extract_content(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        raise ValueError("Модель вернула пустой ответ.")

    content = choices[0].get("message", {}).get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [
            str(part.get("text", ""))
            if isinstance(part, dict) and part.get("type") == "text"
            else str(part)
            for part in content
            if isinstance(part, (dict, str))
        ]
        merged = "\n".join(parts).strip()
        if merged:
            return merged

    raise ValueError("Не удалось извлечь текст ответа модели.")


def call_chat_completion(
    messages: list[dict[str, str]],
    api_key: str,
    config: dict[str, Any],
) -> str:
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        _chat_url(config),
        headers=headers,
        json=payload,
        timeout=config["timeout_seconds"],
    )
    response.raise_for_status()
    return _extract_content(response.json())


def get_ai_response(
    user_prompt: str,
    df: Optional[pd.DataFrame],
    config: dict[str, Any],
    api_key: str = "",
) -> dict[str, Any]:
    resolved_api_key = _resolve_api_key(api_key)

    messages = build_llm_messages(user_prompt=user_prompt, df=df, config=config)
    raw_output = call_chat_completion(messages=messages, api_key=resolved_api_key, config=config)
    summary = raw_output.strip() or "Модель не вернула текстовый ответ."
    return {"summary": summary, "key_metrics": [], "charts": []}
