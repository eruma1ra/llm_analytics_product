from __future__ import annotations

from datetime import datetime
from io import BytesIO, StringIO
import re
import time
from typing import Any, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st

from analytics_core import (
    build_chart_figure,
    build_fallback_chart_specs,
    normalize_chart_specs,
    wants_chart_request,
)
from chat_cache import cleanup_expired_cache, persist_user_cache, restore_user_cache
from llm_client import get_ai_response, load_llm_config


st.set_page_config(page_title="LLM-аналитика", layout="wide")

MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = ("csv", "xlsx", "xls")
DEFAULT_ASSISTANT_TEXT = (
    "Здравствуйте. Прикрепите CSV/Excel через скрепку и (или) напишите запрос. "
    "Я покажу краткую аналитику в чате."
)


def get_file_extension(filename: str) -> str:
    parts = filename.lower().rsplit(".", maxsplit=1)
    return parts[-1] if len(parts) == 2 else ""


def read_tabular_file(file_bytes: bytes, extension: str) -> pd.DataFrame:
    if extension == "csv":
        return pd.read_csv(BytesIO(file_bytes))
    if extension in ("xlsx", "xls"):
        return pd.read_excel(BytesIO(file_bytes))
    raise ValueError(f"Неподдерживаемый формат таблицы: {extension}")


def validate_file_size(file_size: int) -> Optional[str]:
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        return (
            f"Файл слишком большой: {file_size / 1024 / 1024:.1f} МБ. "
            f"Максимум: {MAX_FILE_SIZE_MB} МБ."
        )
    return None


def _convert_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for col in converted.columns:
        numeric = pd.to_numeric(converted[col], errors="coerce")
        if float(numeric.notna().mean()) >= 0.8:
            converted[col] = numeric
    return converted


def _finalize_text_table_candidate(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None

    prepared = df.copy()
    prepared = prepared.dropna(axis=1, how="all")
    prepared.columns = [str(col).strip() for col in prepared.columns]
    if prepared.shape[1] < 2 or prepared.shape[0] < 2:
        return None

    prepared = _convert_numeric_like_columns(prepared)
    numeric_cols = prepared.select_dtypes(include="number").shape[1]
    if numeric_cols < 1:
        return None
    return prepared


def _try_parse_delimited_text(cleaned: str) -> Optional[pd.DataFrame]:
    for sep in ("\t", ";", "|", ","):
        try:
            candidate = pd.read_csv(StringIO(cleaned), sep=sep)
        except Exception:
            continue
        ready = _finalize_text_table_candidate(candidate)
        if ready is not None:
            return ready
    return None


def _try_parse_spaced_columns(cleaned: str) -> Optional[pd.DataFrame]:
    lines = [line.rstrip() for line in cleaned.splitlines() if line.strip()]
    if len(lines) < 3:
        return None

    split_lines = [re.split(r"\t| {2,}", line.strip()) for line in lines]
    width = len(split_lines[0])
    if width < 2:
        return None
    if not all(len(parts) == width for parts in split_lines[1:]):
        return None

    header = split_lines[0]
    rows = split_lines[1:]
    if len(set(header)) != len(header):
        return None

    candidate = pd.DataFrame(rows, columns=header)
    return _finalize_text_table_candidate(candidate)


def try_parse_table_from_text(text: str) -> Optional[pd.DataFrame]:
    cleaned = text.strip()
    if not cleaned:
        return None

    # Часто запрос идет перед таблицей в одной строке.
    anchor = re.search(
        r"\bLocation\s+Country\s+Category\s+Visitors\s+Rating\s+Revenue\s+Accommodation_Available\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    if anchor:
        cleaned = cleaned[anchor.start() :]

    # 0) Универсальный парсинг для таблиц из Excel/текста.
    delimited = _try_parse_delimited_text(cleaned)
    if delimited is not None:
        return delimited
    spaced = _try_parse_spaced_columns(cleaned)
    if spaced is not None:
        return spaced

    # 1) Кейс, когда таблицу вставили несколькими строками.
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    split_lines = [re.split(r"\s+", line) for line in lines]
    if len(split_lines) >= 3:
        width = len(split_lines[0])
        if width >= 4 and all(len(parts) == width for parts in split_lines[1:]):
            header = split_lines[0]
            rows = split_lines[1:]
            if len(set(header)) == len(header):
                df = pd.DataFrame(rows, columns=header)
                df = _convert_numeric_like_columns(df)
                if df.select_dtypes(include="number").shape[1] >= 2:
                    return df

    # 2) Кейс, когда таблица пришла одной строкой (как в чате).
    tokens = re.split(r"\s+", cleaned)
    if len(tokens) < 28:
        return None

    best_df: Optional[pd.DataFrame] = None
    best_score = -1
    max_width = min(14, len(tokens) // 3)
    for width in range(4, max_width + 1):
        header = tokens[:width]
        body = tokens[width:]
        if len(body) % width != 0:
            continue

        row_count = len(body) // width
        if row_count < 3:
            continue
        if len(set(header)) != len(header):
            continue
        if not all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", col) for col in header):
            continue

        rows = [body[i : i + width] for i in range(0, len(body), width)]
        df = pd.DataFrame(rows, columns=header)
        df = _convert_numeric_like_columns(df)
        numeric_cols = df.select_dtypes(include="number").shape[1]
        if numeric_cols < 2:
            continue

        score = row_count * 10 + numeric_cols
        if score > best_score:
            best_score = score
            best_df = df

    return best_df


def _default_assistant_message() -> dict[str, Any]:
    return {"role": "assistant", "content": DEFAULT_ASSISTANT_TEXT}


def _short_title(text: str) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return "Новый чат"
    return cleaned[:48] + ("..." if len(cleaned) > 48 else "")


def _read_uid_from_query() -> str:
    try:
        raw = st.query_params.get("uid", "")
    except Exception:
        raw = ""
    if isinstance(raw, list):
        return str(raw[0]) if raw else ""
    return str(raw or "").strip()


def _write_uid_to_query(uid: str) -> None:
    try:
        st.query_params["uid"] = uid
    except Exception:
        try:
            st.experimental_set_query_params(uid=uid)
        except Exception:
            pass


def _cleanup_expired_cache() -> None:
    cleanup_expired_cache()


def _persist_user_cache() -> None:
    persist_user_cache(
        user_id=st.session_state.user_id,
        conversations=st.session_state.conversations,
        active_conversation_id=st.session_state.active_conversation_id,
    )


def _restore_user_cache(user_id: str) -> bool:
    payload = restore_user_cache(user_id)
    if not payload:
        return False

    st.session_state.conversations = payload.get("conversations", [])
    st.session_state.active_conversation_id = payload.get("active_conversation_id")
    return True


def _touch_conversation(conversation: dict[str, Any]) -> None:
    conversation["updated_ts"] = time.time()
    conversation["updated_at"] = datetime.now().strftime("%H:%M")


def _ensure_conversation_meta(conversation: dict[str, Any]) -> None:
    created_ts = float(conversation.get("created_ts", 0.0) or 0.0)
    if created_ts <= 0:
        fallback_ts = float(conversation.get("updated_ts", 0.0) or time.time())
        conversation["created_ts"] = fallback_ts
    if not conversation.get("created_at"):
        conversation["created_at"] = datetime.fromtimestamp(
            float(conversation["created_ts"])
        ).strftime("%d.%m %H:%M")
    if "updated_ts" not in conversation:
        conversation["updated_ts"] = float(conversation["created_ts"])
    if "updated_at" not in conversation:
        conversation["updated_at"] = conversation["created_at"]


def _new_conversation(title: str = "Новый чат") -> dict[str, Any]:
    now_ts = time.time()
    conversation = {
        "id": uuid4().hex[:10],
        "title": title,
        "messages": [_default_assistant_message()],
        "df": None,
        "file_name": None,
        "created_at": datetime.now().strftime("%d.%m %H:%M"),
        "created_ts": now_ts,
        "updated_at": datetime.now().strftime("%H:%M"),
        "updated_ts": now_ts,
    }
    return conversation


def _get_active_conversation() -> dict[str, Any]:
    active_id = st.session_state.active_conversation_id
    for conversation in st.session_state.conversations:
        _ensure_conversation_meta(conversation)
        if conversation["id"] == active_id:
            return conversation
    # Если активный чат по какой-то причине потерялся — поднимем первый.
    if st.session_state.conversations:
        st.session_state.active_conversation_id = st.session_state.conversations[0]["id"]
        _ensure_conversation_meta(st.session_state.conversations[0])
        return st.session_state.conversations[0]

    fallback = _new_conversation()
    st.session_state.conversations = [fallback]
    st.session_state.active_conversation_id = fallback["id"]
    return fallback


def _sync_active_to_cache() -> None:
    conversation = _get_active_conversation()
    conversation["messages"] = list(st.session_state.messages)
    conversation["df"] = st.session_state.df
    conversation["file_name"] = st.session_state.file_name
    _touch_conversation(conversation)
    _persist_user_cache()


def _activate_conversation(conversation_id: str) -> None:
    for conversation in st.session_state.conversations:
        _ensure_conversation_meta(conversation)
        if conversation["id"] == conversation_id:
            st.session_state.active_conversation_id = conversation_id
            st.session_state.messages = list(conversation["messages"])
            st.session_state.df = conversation["df"]
            st.session_state.file_name = conversation["file_name"]
            _persist_user_cache()
            return


def _is_empty_conversation(conversation: dict[str, Any]) -> bool:
    messages = conversation.get("messages") or []
    if len(messages) != 1:
        return False
    first = messages[0]
    return (
        first.get("role") == "assistant"
        and first.get("content") == DEFAULT_ASSISTANT_TEXT
        and conversation.get("df") is None
        and not conversation.get("file_name")
    )


def _start_new_conversation() -> bool:
    # Если пустой новый чат уже есть, просто переключаемся на него.
    for conversation in st.session_state.conversations:
        if _is_empty_conversation(conversation):
            _activate_conversation(conversation["id"])
            return False

    if st.session_state.conversations:
        _sync_active_to_cache()
    convo = _new_conversation()
    st.session_state.conversations.insert(0, convo)
    _activate_conversation(convo["id"])
    _persist_user_cache()
    return True


def _ensure_active_conversation() -> None:
    if not st.session_state.conversations:
        convo = _new_conversation()
        st.session_state.conversations = [convo]
        st.session_state.active_conversation_id = convo["id"]
    for conversation in st.session_state.conversations:
        _ensure_conversation_meta(conversation)
    if not st.session_state.active_conversation_id:
        st.session_state.active_conversation_id = st.session_state.conversations[0]["id"]
    _activate_conversation(st.session_state.active_conversation_id)
    _persist_user_cache()


def _append_and_store_message(message: dict[str, Any]) -> None:
    st.session_state.messages.append(message)
    _sync_active_to_cache()


def _update_active_conversation_title(title: str) -> None:
    cleaned = (title or "").strip()
    if not cleaned:
        return
    conversation = _get_active_conversation()
    conversation["title"] = cleaned[:120]
    _touch_conversation(conversation)
    _persist_user_cache()


def init_state() -> None:
    _cleanup_expired_cache()

    if "user_id" not in st.session_state:
        user_id = _read_uid_from_query() or uuid4().hex[:18]
        st.session_state.user_id = user_id
        _write_uid_to_query(user_id)
    else:
        requested_uid = _read_uid_from_query()
        if requested_uid and requested_uid != st.session_state.user_id:
            st.session_state.user_id = requested_uid
            st.session_state.conversations = []
            st.session_state.active_conversation_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = []
    if not st.session_state.conversations:
        _restore_user_cache(st.session_state.user_id)
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None


def parse_uploaded_file(uploaded_file) -> None:
    file_error = validate_file_size(uploaded_file.size)
    if file_error:
        st.error(file_error)
        return

    extension = get_file_extension(uploaded_file.name)
    if extension not in ALLOWED_EXTENSIONS:
        st.error("Такой тип файла пока не поддерживается.")
        return

    try:
        df = read_tabular_file(uploaded_file.getvalue(), extension)
    except Exception as exc:
        st.error(f"Не удалось разобрать файл: {exc}")
        return

    st.session_state.df = df
    st.session_state.file_name = uploaded_file.name


def render_sidebar() -> None:
    with st.sidebar:
        if st.button("Новый чат", use_container_width=True):
            _start_new_conversation()
            st.rerun()

        st.subheader("История")
        history = sorted(
            st.session_state.conversations,
            key=lambda item: float(item.get("created_ts", item.get("updated_ts", 0.0))),
            reverse=True,
        )
        if not history:
            st.caption("Пока пусто.")
            return

        current_id = st.session_state.active_conversation_id
        options = [item["id"] for item in history[:30]]
        if current_id not in options:
            current_id = options[0]

        labels = {
            item["id"]: str(item.get("title") or "Новый чат")
            for item in history[:30]
        }
        selected_id = st.radio(
            "Список чатов",
            options=options,
            index=options.index(current_id),
            format_func=lambda chat_id: labels.get(chat_id, "Чат"),
            label_visibility="collapsed",
        )
        if selected_id != st.session_state.active_conversation_id:
            _sync_active_to_cache()
            _activate_conversation(selected_id)
            st.rerun()


def process_chat_payload(payload) -> Optional[str]:
    if payload is None:
        return None

    user_prompt = payload.strip() if isinstance(payload, str) else payload.text.strip()
    uploaded_files = [] if isinstance(payload, str) else payload.files

    # Если пользователь прикрепил файл, сразу пробуем подключить его в контекст.
    if uploaded_files:
        uploaded_file = uploaded_files[0]
        parse_uploaded_file(uploaded_file)

        if st.session_state.df is not None:
            _sync_active_to_cache()
            _append_and_store_message(
                {
                    "role": "assistant",
                    "content": f"Файл **{uploaded_file.name}** подключен.",
                }
            )

        if len(uploaded_files) > 1:
            _append_and_store_message(
                {
                    "role": "assistant",
                    "content": "Подключен только первый файл из списка.",
                }
            )
    elif st.session_state.df is None and user_prompt:
        parsed_df = try_parse_table_from_text(user_prompt)
        if parsed_df is not None:
            st.session_state.df = parsed_df
            st.session_state.file_name = "Таблица из текста"
            _sync_active_to_cache()
            _append_and_store_message(
                {
                    "role": "assistant",
                    "content": "Таблица из сообщения распознана и подключена.",
                }
            )

    return user_prompt if user_prompt else None


def render_chat_message(message: dict[str, Any], index: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] != "assistant":
            return

        chart_specs = message.get("chart_specs", [])
        if not chart_specs:
            return

        if st.session_state.df is None:
            st.caption("Для построения графиков нужно снова подключить файл.")
            return

        for chart_idx, chart_spec in enumerate(chart_specs):
            chart_title = chart_spec.get("title", "Без названия")
            try:
                figure = build_chart_figure(st.session_state.df, chart_spec)
                if figure is None:
                    st.caption(f"Не удалось построить график: {chart_title}")
                    continue
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                    key=f"chart_{index}_{chart_idx}",
                )
            except Exception as exc:
                st.caption(f"Не удалось построить график: {chart_title}. Ошибка: {exc}")


def render_chat(llm_config: dict[str, Any]) -> None:
    st.markdown(
        f'<div style="font-size:0.84rem; color:#6b7280; margin-top:0.35rem; margin-bottom:0.02rem;">Model: <code>{llm_config["model"]}</code></div>',
        unsafe_allow_html=True,
    )
    st.title("LLM-аналитика")
    st.caption("Задайте вопрос по данным или отправьте текстовый запрос.")

    for idx, message in enumerate(st.session_state.messages):
        render_chat_message(message, idx)

    if st.session_state.df is not None:
        c1, c2 = st.columns([6, 1])
        with c1:
            rows, cols = st.session_state.df.shape
            st.caption(f"Подключен файл: **{st.session_state.file_name}** ({rows:,} x {cols:,})")
        with c2:
            if st.button("Сбросить", use_container_width=True):
                st.session_state.df = None
                st.session_state.file_name = None
                _sync_active_to_cache()
                _append_and_store_message({"role": "assistant", "content": "Файл отсоединен."})
                st.rerun()

    payload = st.chat_input(
        "Введите запрос",
        accept_file=True,
        file_type=list(ALLOWED_EXTENSIONS),
    )
    user_prompt = process_chat_payload(payload)
    if not user_prompt:
        return

    user_message = {"role": "user", "content": user_prompt}
    _append_and_store_message(user_message)
    # Держим заголовок чата актуальным по последнему запросу.
    _update_active_conversation_title(_short_title(user_prompt))

    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        # Берем текст и спецификации графиков от модели.
        ai_payload = get_ai_response(
            user_prompt=user_prompt,
            df=st.session_state.df,
            config=llm_config,
        )

        chart_specs: list[dict[str, Any]] = []
        if st.session_state.df is not None:
            chart_specs = normalize_chart_specs(
                ai_payload.get("chart_specs", []),
                st.session_state.df,
                max_charts=llm_config["max_charts"],
            )
            if wants_chart_request(user_prompt) and not chart_specs:
                chart_specs = build_fallback_chart_specs(
                    user_prompt,
                    st.session_state.df,
                    max_charts=llm_config["max_charts"],
                )

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": ai_payload["summary"],
            "chart_specs": chart_specs,
        }
    except Exception as exc:
        assistant_message = {
            "role": "assistant",
            "content": "Не удалось получить ответ от модели. Проверьте настройки API и повторите запрос.",
            "chart_specs": [],
        }
        st.warning(f"Ответ от модели временно недоступен: {exc}")

    _append_and_store_message(assistant_message)
    render_chat_message(assistant_message, len(st.session_state.messages) - 1)


def apply_ui_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1200px;
                padding-top: 3.2rem;
                padding-bottom: 1.5rem;
            }
            .block-container h1 {
                padding: 1rem 0px 1rem !important;
            }
            div[data-testid="stChatInput"] [data-baseweb="textarea"] {
                border-radius: 16px !important;
                overflow: hidden !important;
            }
            div[data-testid="stChatInput"] [data-baseweb="textarea"] textarea {
                min-height: 72px !important;
                line-height: 1.35 !important;
                padding-top: 24px !important;
                padding-bottom: 24px !important;
                padding-left: 16px !important;
                padding-right: 60px !important;
                box-sizing: border-box !important;
            }
            div[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] {
                border-right: none !important;
                align-self: center !important;
                margin-top: 0 !important;
            }
            div[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] > button {
                width: 40px !important;
                height: 40px !important;
                min-height: 40px !important;
                border-radius: 999px !important;
                border: 1px solid #d7dbe0 !important;
                background: #ffffff !important;
                padding: 0 !important;
                display: inline-flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            div[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] + div {
                display: block !important;
                width: 1px !important;
                min-width: 1px !important;
                height: 32px !important;
                margin-top: 0 !important;
                margin-left: 10px !important;
                align-self: center !important;
                background: #e5e7eb !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
                border-radius: 14px !important;
                min-height: 2.5rem !important;
                justify-content: flex-start !important;
                text-align: left !important;
                box-shadow: none !important;
                transition: background-color 120ms ease, border-color 120ms ease;
            }
            section[data-testid="stSidebar"] div[data-testid="stButton"] > button p {
                margin: 0 !important;
                width: 100% !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                display: block !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] {
                margin-right: 0 !important;
                padding-right: 0 !important;
                box-sizing: border-box !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] > div,
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] {
                width: 100% !important;
                margin-right: 0 !important;
                padding-right: 0 !important;
                box-sizing: border-box !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] {
                margin: 0.14rem 0 !important;
                margin-right: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
                border: 1px solid transparent;
                border-radius: 14px !important;
                padding: 0.42rem 0.7rem;
                transition: background-color 120ms ease, border-color 120ms ease;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"]:hover {
                background: #eef0f2 !important;
                border: 1px solid #eef0f2 !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"]:has(input:checked) {
                background: #e9ecef !important;
                border: 1px solid #e0e5ea !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
                display: none !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] > div:last-child {
                min-width: 0 !important;
                width: 100% !important;
                overflow: hidden !important;
            }
            section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] p {
                margin: 0 !important;
                width: 100% !important;
                max-width: 100% !important;
                white-space: nowrap !important;
                overflow: hidden !important;
                text-overflow: ellipsis !important;
                display: block !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    _ensure_active_conversation()
    llm_config = load_llm_config()
    apply_ui_styles()
    render_sidebar()
    render_chat(llm_config)


if __name__ == "__main__":
    main()
