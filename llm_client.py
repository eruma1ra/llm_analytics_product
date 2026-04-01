from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ChatCompletionResult:
    content: str
    finish_reason: str


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


def _wants_charts(prompt: str) -> bool:
    text = (prompt or "").lower()
    markers = ("график", "диаграм", "чарт", "chart", "plot", "plotly", "тренд", "trend")
    return any(marker in text for marker in markers)


def build_text_messages(
    user_prompt: str,
    df: Optional[pd.DataFrame],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    if df is None:
        return [
            {
                "role": "system",
                "content": (
                    "Вы аналитический ассистент. "
                    "На любые вопросы отвечайте строго с аналитической точки зрения. "
                    "Даже если вопрос общий (например, про город), давайте ответ через метрики, "
                    "сравнения, факторы и проверяемые выводы. "
                    "Отвечайте по факту: только на то, что спросили, без лишних отступлений. "
                    "Формулировки должны быть конкретными, не расплывчатыми. "
                    "Ответ должен быть коротким: 3-6 предложений или 3-6 коротких пунктов. "
                    "Без воды, без повторов, без вводных фраз. "
                    "Верните только текст на русском языке, без JSON и без кода."
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
                "Отвечайте строго как аналитик: фактами, проверяемыми выводами и конкретикой. "
                "Всегда держите ответ в рамках вопроса пользователя: не добавляйте лишние темы. "
                "Если вопрос широкий, структурируйте ответ через метрики, причины и следствия. "
                "Избегайте расплывчатых формулировок. "
                "Ответ должен быть коротким: 4-8 коротких пунктов по сути. "
                "Пишите только факты и выводы, без длинных вступлений и общих рассуждений. "
                "Верните только текст на русском языке, без JSON и без кода."
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


def build_chart_messages(
    user_prompt: str,
    df: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    context = build_dataframe_context(df, max_rows=config["max_context_rows"])
    max_charts = max(1, min(int(config["max_charts"]), 5))
    return [
        {
            "role": "system",
            "content": (
                "Вы генератор спецификаций графиков для UI. "
                "Верните ТОЛЬКО JSON без markdown. "
                "Формат: {\"charts\":[{\"type\":\"bar|line|scatter|pie|histogram\","
                "\"x\":\"<col>\",\"y\":\"<col|null>\",\"agg\":\"sum|mean|count|median|max|min\","
                "\"title\":\"<text>\",\"top_n\":30}]}. "
                f"Максимум графиков: {max_charts}. "
                "Используйте только реальные названия колонок из данных."
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


def _extract_finish_reason(data: dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    return str(choices[0].get("finish_reason", "") or "")


def _looks_truncated(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return True

    if cleaned.endswith(("**", "*", "```", "`", "(", "[", "{", ":", "-", "—", "–", ",")):
        return True
    if cleaned.count("**") % 2 == 1:
        return True
    if cleaned.count("```") % 2 == 1:
        return True
    return False


def _merge_with_overlap(base: str, continuation: str) -> str:
    left = (base or "").rstrip()
    right = (continuation or "").strip()
    if not right:
        return left
    if right in left:
        return left

    max_overlap = min(len(left), len(right), 300)
    overlap = 0
    for size in range(max_overlap, 2, -1):
        if left[-size:] == right[:size]:
            overlap = size
            break

    if overlap > 0:
        return (left + right[overlap:]).strip()
    return (left + "\n" + right).strip()


def call_chat_completion(
    messages: list[dict[str, str]],
    api_key: str,
    config: dict[str, Any],
) -> ChatCompletionResult:
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
    data = response.json()
    return ChatCompletionResult(
        content=_extract_content(data),
        finish_reason=_extract_finish_reason(data),
    )


def _extract_json_candidates(text: str) -> list[str]:
    candidates = [text.strip()]
    for match in JSON_BLOCK_RE.finditer(text or ""):
        body = (match.group("body") or "").strip()
        if body:
            candidates.append(body)

    raw = text or ""
    left_brace = raw.find("{")
    right_brace = raw.rfind("}")
    if left_brace >= 0 and right_brace > left_brace:
        candidates.append(raw[left_brace : right_brace + 1])
    left_bracket = raw.find("[")
    right_bracket = raw.rfind("]")
    if left_bracket >= 0 and right_bracket > left_bracket:
        candidates.append(raw[left_bracket : right_bracket + 1])

    # Убираем дубли, сохраняя порядок.
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _parse_chart_specs(raw: str, max_charts: int) -> list[dict[str, Any]]:
    limit = max(0, min(int(max_charts), 5))
    for candidate in _extract_json_candidates(raw):
        try:
            data = json.loads(candidate)
        except Exception:
            continue

        charts: list[Any] = []
        if isinstance(data, list):
            charts = data
        elif isinstance(data, dict):
            payload = data.get("charts")
            if isinstance(payload, list):
                charts = payload

        if not charts:
            continue

        cleaned: list[dict[str, Any]] = []
        for item in charts:
            if isinstance(item, dict):
                cleaned.append(item)
            if len(cleaned) >= limit:
                break
        if cleaned:
            return cleaned
    return []


def get_ai_response(
    user_prompt: str,
    df: Optional[pd.DataFrame],
    config: dict[str, Any],
    api_key: str = "",
) -> dict[str, Any]:
    resolved_api_key = _resolve_api_key(api_key)

    text_messages = build_text_messages(user_prompt=user_prompt, df=df, config=config)
    text_result = call_chat_completion(
        messages=text_messages,
        api_key=resolved_api_key,
        config=config,
    )
    summary = text_result.content.strip() or "Модель не вернула текстовый ответ."

    # Если модель оборвала текст, делаем 1-2 дозапроса на продолжение.
    attempts = 0
    while attempts < 2 and (
        text_result.finish_reason.lower() == "length" or _looks_truncated(summary)
    ):
        continuation_messages = text_messages + [
            {"role": "assistant", "content": summary},
            {
                "role": "user",
                "content": (
                    "Продолжите ответ с того же места. "
                    "Нужен только хвост продолжения, без повторов уже написанного."
                ),
            },
        ]
        text_result = call_chat_completion(
            messages=continuation_messages,
            api_key=resolved_api_key,
            config=config,
        )
        summary = _merge_with_overlap(summary, text_result.content)
        attempts += 1

    chart_specs: list[dict[str, Any]] = []
    if df is not None and _wants_charts(user_prompt):
        try:
            chart_raw = call_chat_completion(
                messages=build_chart_messages(user_prompt=user_prompt, df=df, config=config),
                api_key=resolved_api_key,
                config=config,
            ).content
            chart_specs = _parse_chart_specs(chart_raw, max_charts=config["max_charts"])
        except Exception:
            chart_specs = []

    return {
        "summary": summary,
        "chart_specs": chart_specs,
    }
