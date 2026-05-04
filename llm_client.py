from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from code_interpreter import execute_python_analysis
from llm_stream import stream_chat_completion


load_dotenv()
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(?P<body>.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ChatCompletionResult:
    content: str
    finish_reason: str
    tool_calls: list[dict[str, Any]]
    message: dict[str, Any]


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


def _env_int_default(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"В .env поле {name} должно быть целым числом.")


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
        "agent_max_steps": _env_int_default("LLM_AGENT_MAX_STEPS", 5),
        "code_timeout_seconds": _env_int_default("LLM_CODE_TIMEOUT_SECONDS", 12),
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
                "Если в данных есть безымянные колонки вида 'Unnamed: N', "
                "давайте им осмысленные названия по контексту и используйте их в ответе. "
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
                "\"title\":\"<text>\",\"x_label\":\"<alias|null>\",\"y_label\":\"<alias|null>\",\"top_n\":30}]}. "
                f"Максимум графиков: {max_charts}. "
                "Используйте только реальные названия колонок из данных. "
                "Если type='pie' и есть числовой столбец (например, Остаток/Количество/Revenue), "
                "обязательно ставьте его в y и agg='sum', не используйте count. "
                "Если пользователь просит график по метрике (например, выручка/revenue, остаток/stock, продажи/sales), "
                "обязательно укажите эту метрику в y и agg='sum' или 'mean' по смыслу. "
                "Не возвращайте график с agg='count', если пользователь явно не просил количество записей. "
                "Если пользователь просит конкретное количество графиков (например, 1 график), "
                "верните ровно это количество, без дополнительных графиков. "
                "Не добавляйте лишние графики «для полноты». "
                "Если пользователь просит конкретный тип (например, круговая/линейная/столбчатая), "
                "возвращайте только этот тип графика. "
                "Если колонка называется 'Unnamed: N', придумайте осмысленный псевдоним по данным "
                "и передайте его в x_label/y_label. "
                "Язык псевдонима выбирайте по языку данных: "
                "если названия в данных на английском, псевдоним на английском; "
                "если на русском, псевдоним на русском. "
                "Не смешивайте языки в одном псевдониме. "
                "В x и y всегда оставляйте оригинальные названия колонок."
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


def _extract_message(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices", [])
    if not choices:
        return {}
    message = choices[0].get("message", {})
    return message if isinstance(message, dict) else {}


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
    messages: list[dict[str, Any]],
    api_key: str,
    config: dict[str, Any],
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[Any] = None,
) -> ChatCompletionResult:
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
    }
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
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
    message = _extract_message(data)
    return ChatCompletionResult(
        content=_extract_content(data) if message.get("content") is not None else "",
        finish_reason=_extract_finish_reason(data),
        tool_calls=message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else [],
        message=message,
    )


def stream_text_response(
    user_prompt: str,
    df: Optional[pd.DataFrame],
    config: dict[str, Any],
    api_key: str = "",
) -> Iterator[str]:
    if df is not None:
        result = run_dataframe_agent(
            user_prompt=user_prompt,
            df=df,
            config=config,
            api_key=api_key,
        )
        yield str(result.get("summary", "")).strip() or "Модель не вернула текстовый ответ."
        return

    resolved_api_key = _resolve_api_key(api_key)
    text_messages = build_text_messages(user_prompt=user_prompt, df=df, config=config)
    plain_user_content = text_messages[-1]["content"] if text_messages else user_prompt

    def _call_nonempty_text(messages: list[dict[str, str]], max_attempts: int = 3) -> str:
        attempt_messages = list(messages)
        for attempt in range(max_attempts):
            result = call_chat_completion(
                messages=attempt_messages,
                api_key=resolved_api_key,
                config=config,
            )
            cleaned = (result.content or "").strip()
            if cleaned:
                return cleaned

            # 1) Просим явно вернуть непустой текст.
            if attempt == 0:
                attempt_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Ответ должен быть непустым. "
                            "Верните краткий содержательный ответ на русском языке."
                        ),
                    }
                ]
                continue

            # 2) Максимально простой системный запрос без сложных ограничений.
            if attempt == 1:
                attempt_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Вы помощник. Верните непустой, краткий и конкретный ответ "
                            "на русском языке по вопросу пользователя."
                        ),
                    },
                    {"role": "user", "content": plain_user_content},
                ]

        return ""

    try:
        streamed_has_text = False
        for chunk in stream_chat_completion(
            url=_chat_url(config),
            api_key=resolved_api_key,
            model=config["model"],
            messages=text_messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            timeout_seconds=config["timeout_seconds"],
        ):
            if str(chunk).strip():
                streamed_has_text = True
            yield chunk
        if streamed_has_text:
            return
    except Exception:
        # Если stream не поддерживается провайдером, мягко откатываемся на обычный запрос.
        pass

    fallback_text = _call_nonempty_text(text_messages, max_attempts=3)
    if fallback_text:
        yield fallback_text
        return

    yield "Сервис временно не дал содержательный ответ. Я сделал несколько попыток автоматически."


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


ANALYSIS_TOOL_NAME = "execute_python"


def _analysis_tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": ANALYSIS_TOOL_NAME,
                "description": (
                    "Выполняет Python/pandas-код над загруженным датасетом. "
                    "В коде уже доступна переменная df. "
                    "Код должен посчитать метрики по df, записать текстовый вывод в answer, "
                    "а спецификации графиков при необходимости в charts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python-код. Доступны df, pd и px. "
                                "Положите краткий текстовый результат в переменную answer. "
                                "Для графиков задайте charts как список словарей с полями "
                                "type, title, x, y, agg, top_n."
                            ),
                        }
                    },
                    "required": ["code"],
                },
            },
        }
    ]


def _analysis_system_prompt(max_charts: int) -> str:
    return (
        "Вы ИИ-агент для анализа табличных данных. "
        "Загруженный датасет доступен только внутри tool execute_python как pandas DataFrame df. "
        "Перед любым содержательным ответом по датасету обязательно вызовите execute_python "
        "и посчитайте нужные метрики кодом. Не отвечайте по памяти и не выдумывайте значения. "
        "В первом вызове при необходимости исследуйте df.shape, df.columns, df.dtypes, df.head(), "
        "пропуски и базовые распределения. "
        "Код должен записать текстовый результат в переменную answer. "
        "Если пользователь просит графики или они нужны для результата, задайте charts. "
        "Формат charts: список JSON-совместимых словарей "
        "{type: 'bar|line|scatter|pie|histogram', x: '<column>', y: '<column|null>', "
        "agg: 'sum|mean|count|median|max|min', title: '<title>', top_n: 30}. "
        f"Максимум графиков: {max(1, min(int(max_charts), 5))}. "
        "После результата tool дайте финальный ответ на русском языке: коротко, по делу, "
        "с конкретными числами из вычислений."
    )


def _tool_choice_required() -> dict[str, Any]:
    return {"type": "function", "function": {"name": ANALYSIS_TOOL_NAME}}


def _parse_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function", {})
    raw_args = function.get("arguments", "{}")
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        return {}
    try:
        parsed = json.loads(raw_args)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _collect_chart_specs(tool_result: dict[str, Any]) -> list[dict[str, Any]]:
    charts = tool_result.get("charts", [])
    if not isinstance(charts, list):
        return []
    return [item for item in charts if isinstance(item, dict)]


def _extract_code_from_json_response(raw: str) -> str:
    for candidate in _extract_json_candidates(raw):
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("code"), str):
            return data["code"].strip()
    return ""


def _fallback_agent_via_code_protocol(
    user_prompt: str,
    df: pd.DataFrame,
    config: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    max_charts = max(1, min(int(config["max_charts"]), 5))
    code_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Вы пишете код для tool execute_python. "
                "Верните ТОЛЬКО JSON без markdown в формате {\"code\":\"...\"}. "
                "В коде доступен pandas DataFrame df, а также pd и px. "
                "Код обязан исследовать df и посчитать ответ на вопрос пользователя. "
                "Запишите краткий текстовый вывод в переменную answer. "
                "Если нужны графики, запишите charts как список спецификаций "
                "type/x/y/agg/title/top_n, максимум "
                f"{max_charts}. Не используйте внешние файлы и import."
            ),
        },
        {"role": "user", "content": f"Запрос пользователя: {user_prompt}"},
    ]

    code_result = call_chat_completion(
        messages=code_messages,
        api_key=api_key,
        config=config,
    )
    code = _extract_code_from_json_response(code_result.content)
    if not code:
        return {
            "summary": "Модель не смогла сформировать код анализа для интерпретатора.",
            "chart_specs": [],
            "tool_used": False,
            "tool_results": [],
        }

    tool_result = execute_python_analysis(
        df=df,
        code=code,
        timeout_seconds=config["code_timeout_seconds"],
    )

    if not tool_result.get("ok"):
        repair_messages = code_messages + [
            {"role": "assistant", "content": json.dumps({"code": code}, ensure_ascii=False)},
            {
                "role": "user",
                "content": (
                    "Код упал в execute_python. Верните исправленный JSON {\"code\":\"...\"}. "
                    f"Ошибка:\n{tool_result.get('error', '')}"
                ),
            },
        ]
        repaired = call_chat_completion(
            messages=repair_messages,
            api_key=api_key,
            config=config,
        )
        repaired_code = _extract_code_from_json_response(repaired.content)
        if repaired_code:
            tool_result = execute_python_analysis(
                df=df,
                code=repaired_code,
                timeout_seconds=config["code_timeout_seconds"],
            )

    charts = _collect_chart_specs(tool_result)
    final_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Сформируйте финальный аналитический ответ на русском языке. "
                "Используйте только результат execute_python, не добавляйте непроверенные числа."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Запрос пользователя: {user_prompt}\n\n"
                f"Результат execute_python:\n{json.dumps(tool_result, ensure_ascii=False)}"
            ),
        },
    ]
    final_result = call_chat_completion(
        messages=final_messages,
        api_key=api_key,
        config=config,
    )
    summary = final_result.content.strip() or str(tool_result.get("answer", "")).strip()
    if not summary and tool_result.get("error"):
        summary = f"Интерпретатор кода вернул ошибку: {tool_result['error']}"

    return {
        "summary": summary or "Не удалось получить содержательный результат анализа.",
        "chart_specs": charts,
        "tool_used": True,
        "tool_results": [tool_result],
    }


def run_dataframe_agent(
    user_prompt: str,
    df: pd.DataFrame,
    config: dict[str, Any],
    api_key: str = "",
) -> dict[str, Any]:
    resolved_api_key = _resolve_api_key(api_key)
    max_steps = max(2, min(int(config["agent_max_steps"]), 8))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _analysis_system_prompt(config["max_charts"])},
        {
            "role": "user",
            "content": (
                "Датасет уже загружен в tool execute_python как переменная df. "
                f"Запрос пользователя: {user_prompt}"
            ),
        },
    ]
    tools = _analysis_tool_schema()
    chart_specs: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    tool_used = False

    try:
        for step in range(max_steps):
            result = call_chat_completion(
                messages=messages,
                api_key=resolved_api_key,
                config=config,
                tools=tools,
                tool_choice=_tool_choice_required() if not tool_used else "auto",
            )

            if not result.tool_calls:
                if tool_used and result.content.strip():
                    return {
                        "summary": result.content.strip(),
                        "chart_specs": chart_specs,
                        "tool_used": True,
                        "tool_results": tool_results,
                    }
                break

            assistant_message = {
                "role": "assistant",
                "content": result.message.get("content") or "",
                "tool_calls": result.tool_calls,
            }
            messages.append(assistant_message)

            for tool_call in result.tool_calls:
                function = tool_call.get("function", {})
                if function.get("name") != ANALYSIS_TOOL_NAME:
                    continue

                args = _parse_tool_arguments(tool_call)
                code = str(args.get("code", "")).strip()
                tool_result = execute_python_analysis(
                    df=df,
                    code=code,
                    timeout_seconds=config["code_timeout_seconds"],
                )
                tool_used = True
                tool_results.append(tool_result)
                chart_specs.extend(_collect_chart_specs(tool_result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", "execute_python"),
                        "name": ANALYSIS_TOOL_NAME,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

        if tool_used:
            final_messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Теперь дайте финальный краткий ответ на русском языке, "
                        "используя только результаты execute_python."
                    ),
                }
            ]
            final_result = call_chat_completion(
                messages=final_messages,
                api_key=resolved_api_key,
                config=config,
            )
            return {
                "summary": final_result.content.strip()
                or str(tool_results[-1].get("answer", "")).strip()
                or "Анализ выполнен, но модель не вернула текстовый ответ.",
                "chart_specs": chart_specs,
                "tool_used": True,
                "tool_results": tool_results,
            }
    except requests.HTTPError:
        pass
    except Exception:
        pass

    return _fallback_agent_via_code_protocol(
        user_prompt=user_prompt,
        df=df,
        config=config,
        api_key=resolved_api_key,
    )


def get_chart_specs(
    user_prompt: str,
    df: Optional[pd.DataFrame],
    config: dict[str, Any],
    api_key: str = "",
) -> list[dict[str, Any]]:
    if df is None or not _wants_charts(user_prompt):
        return []

    resolved_api_key = _resolve_api_key(api_key)
    try:
        chart_raw = call_chat_completion(
            messages=build_chart_messages(user_prompt=user_prompt, df=df, config=config),
            api_key=resolved_api_key,
            config=config,
        ).content
        return _parse_chart_specs(chart_raw, max_charts=config["max_charts"])
    except Exception:
        return []


def get_ai_response(
    user_prompt: str,
    df: Optional[pd.DataFrame],
    config: dict[str, Any],
    api_key: str = "",
) -> dict[str, Any]:
    if df is not None:
        return run_dataframe_agent(
            user_prompt=user_prompt,
            df=df,
            config=config,
            api_key=api_key,
        )

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
