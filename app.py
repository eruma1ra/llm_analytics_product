from __future__ import annotations

import re
from collections import Counter
from io import BytesIO
from typing import Optional

import pandas as pd
import streamlit as st


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
    if "uploaded_signature" not in st.session_state:
        st.session_state.uploaded_signature = None


def file_brief(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    numeric = df.select_dtypes(include="number")
    numeric_cols = len(numeric.columns)
    text_cols = cols - numeric_cols

    lines = [
        f"- Строк: **{rows:,}**",
        f"- Колонок: **{cols:,}**",
        f"- Числовых колонок: **{numeric_cols:,}**",
        f"- Текстовых/других колонок: **{text_cols:,}**",
        f"- Пустых ячеек: **{missing:,}**",
    ]
    return "\n".join(lines)


def text_brief(text: str) -> str:
    lines = text.count("\n") + 1
    words = re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", text.lower())
    word_count = len(words)
    char_count = len(text)

    # Небольшой список, чтобы топ-слова выглядели полезнее.
    stop_words = {
        "и",
        "в",
        "на",
        "по",
        "с",
        "к",
        "как",
        "что",
        "это",
        "для",
        "the",
        "and",
        "to",
        "of",
        "in",
        "is",
    }
    useful_words = [w for w in words if len(w) > 2 and w not in stop_words]
    top_words = Counter(useful_words).most_common(5)

    result = [
        f"- Строк: **{lines:,}**",
        f"- Слов: **{word_count:,}**",
        f"- Символов: **{char_count:,}**",
    ]
    if top_words:
        result.append("- Частые слова: " + ", ".join(f"`{w}` ({c})" for w, c in top_words))

    return "\n".join(result)


def build_reply(user_prompt: str) -> str:
    df = st.session_state.df
    prompt = user_prompt.strip()

    if df is None:
        return "\n".join(
            [
                "Сделал разбор вашего текста.",
                "",
                text_brief(prompt),
                "",
                "Если хотите аналитику по данным, прикрепите CSV/Excel через скрепку.",
            ]
        )

    answer = ["Сделал быстрый разбор загруженного файла.", "", file_brief(df)]

    # Пока даем простую локальную аналитику по таблице.
    if "колон" in prompt.lower() or "столб" in prompt.lower():
        cols_preview = ", ".join(map(str, df.columns[:12]))
        answer.append("")
        answer.append(f"Первые колонки: {cols_preview}")
    elif "пропуск" in prompt.lower() or "пуст" in prompt.lower():
        missing_by_col = df.isna().sum().sort_values(ascending=False).head(5)
        if int(missing_by_col.sum()) > 0:
            answer.append("")
            answer.append("Топ-5 колонок по пропускам:")
            for col, val in missing_by_col.items():
                answer.append(f"- {col}: {int(val):,}")
        else:
            answer.append("")
            answer.append("Пропусков не найдено.")
    elif "график" in prompt.lower() or "диаграм" in prompt.lower():
        answer.append("")
        answer.append("Сейчас могу дать текстовый разбор и метрики. Графики можно добавить отдельным блоком.")
    else:
        answer.append("")
        answer.append(
            "Если нужно, уточните запрос: например, «покажите пропуски», "
            "«какие тут ключевые метрики» или «сделайте summary по данным»."
        )

    return "\n".join(answer)


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

    user_prompt = ""
    uploaded_files = []

    if isinstance(payload, str):
        user_prompt = payload.strip()
    else:
        user_prompt = payload.text.strip()
        uploaded_files = payload.files

    if uploaded_files:
        uploaded_file = uploaded_files[0]
        parse_uploaded_file(uploaded_file)

        if st.session_state.df is not None:
            st.session_state.uploaded_signature = f"{uploaded_file.name}:{uploaded_file.size}"
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

    return user_prompt if user_prompt else None


def render_chat() -> None:
    st.title("LLM-аналитика")
    st.caption("Задайте вопрос по данным или отправьте текстовый запрос.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.df is not None:
        c1, c2 = st.columns([6, 1])
        with c1:
            r, c = st.session_state.df.shape
            st.caption(f"Подключен файл: **{st.session_state.file_name}** ({r:,} x {c:,})")
        with c2:
            if st.button("Сбросить", use_container_width=True):
                st.session_state.df = None
                st.session_state.file_name = None
                st.session_state.uploaded_signature = None
                st.session_state.messages.append(
                    {"role": "assistant", "content": "Файл отсоединен."}
                )
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

    assistant_reply = build_reply(user_prompt)
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)


def main() -> None:
    init_state()

    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1200px;
                padding-top: 1.5rem;
                padding-bottom: 1.5rem;
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
            div[data-testid="stChatInput"] [data-baseweb="textarea"] textarea::placeholder {
                opacity: 0.9 !important;
            }
            div[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] {
                border-right: none !important;
                box-shadow: none !important;
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
            div[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] > button:hover,
            div[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] > button:focus,
            div[data-testid="stChatInput"] [data-testid="stChatInputFileUploadButton"] > button:focus-visible {
                border-color: #d7dbe0 !important;
                box-shadow: none !important;
                background: #ffffff !important;
                outline: none !important;
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

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
