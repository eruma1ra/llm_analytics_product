from __future__ import annotations

from io import BytesIO
import re
from typing import Any, Optional

import pandas as pd
import streamlit as st

from analytics_core import build_chart_figure, build_local_fallback_payload
from llm_client import get_ai_response, load_llm_config


st.set_page_config(page_title="LLM-аналитика", layout="wide")

MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = ("csv", "xlsx", "xls")


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


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Здравствуйте. Прикрепите CSV/Excel через скрепку и (или) напишите запрос. "
                    "Я покажу краткую аналитику в чате."
                ),
            }
        ]
    if "df" not in st.session_state:
        st.session_state.df = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None


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
        st.subheader("История")
        st.caption("История чатов")
        st.info("Пока история пуста.")


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
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Файл **{uploaded_file.name}** подключен.",
                }
            )

        if len(uploaded_files) > 1:
            st.session_state.messages.append(
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
            st.session_state.messages.append(
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

        key_metrics = message.get("key_metrics", [])
        if key_metrics:
            st.markdown("**Ключевые метрики:**")
            for metric in key_metrics:
                st.markdown(f"- {metric}")

        chart_specs = message.get("charts", [])
        if chart_specs and st.session_state.df is None:
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
            except Exception:
                st.caption(f"Не удалось построить график: {chart_title}")


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
                st.session_state.messages.append({"role": "assistant", "content": "Файл отсоединен."})
                st.rerun()

    payload = st.chat_input(
        "Введите запрос",
        accept_file=True,
        file_type=list(ALLOWED_EXTENSIONS),
    )
    user_prompt = process_chat_payload(payload)
    if not user_prompt:
        return

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    try:
        # Берем текстовый ответ от модели, а метрики/графики строим локально в UI.
        ai_payload = get_ai_response(
            user_prompt=user_prompt,
            df=st.session_state.df,
            config=llm_config,
        )
        helper_payload = build_local_fallback_payload(user_prompt, st.session_state.df)
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": ai_payload["summary"],
            "key_metrics": helper_payload["key_metrics"],
            "charts": helper_payload["charts"],
        }
    except Exception as exc:
        helper_payload = build_local_fallback_payload(user_prompt, st.session_state.df)
        assistant_message = {
            "role": "assistant",
            "content": helper_payload["summary"],
            "key_metrics": helper_payload["key_metrics"],
            "charts": helper_payload["charts"],
        }
        st.warning(f"Ответ от модели временно недоступен: {exc}")

    st.session_state.messages.append(assistant_message)
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
                padding-right: 12px !important;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    llm_config = load_llm_config()
    apply_ui_styles()
    render_sidebar()
    render_chat(llm_config)


if __name__ == "__main__":
    main()
