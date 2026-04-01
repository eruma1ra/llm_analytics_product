from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

import pandas as pd
import plotly.express as px

STOP_WORDS = {
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
ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "pie"}
ALLOWED_AGG = {"sum", "mean", "count", "median", "max", "min"}


def resolve_column_name(df: pd.DataFrame, requested: str) -> Optional[str]:
    if not requested or not str(requested).strip():
        return None

    requested = str(requested).strip()
    columns = [str(col) for col in df.columns]
    if requested in columns:
        return requested

    return {col.lower(): col for col in columns}.get(requested.lower())


def file_brief(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    missing = int(df.isna().sum().sum())
    numeric_cols = len(df.select_dtypes(include="number").columns)
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

    useful_words = [word for word in words if len(word) > 2 and word not in STOP_WORDS]
    top_words = Counter(useful_words).most_common(5)

    result = [
        f"- Строк: **{lines:,}**",
        f"- Слов: **{word_count:,}**",
        f"- Символов: **{char_count:,}**",
    ]
    if top_words:
        result.append(
            "- Частые слова: " + ", ".join(f"`{word}` ({count})" for word, count in top_words)
        )
    return "\n".join(result)


def build_fallback_reply(user_prompt: str, df: Optional[pd.DataFrame]) -> str:
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


def build_local_fallback_payload(user_prompt: str, df: Optional[pd.DataFrame]) -> dict[str, Any]:
    summary = build_fallback_reply(user_prompt, df)
    if df is None:
        return {"summary": summary, "key_metrics": [], "charts": []}

    prompt = user_prompt.lower()
    numeric_cols = [str(col) for col in df.select_dtypes(include="number").columns]
    key_metrics: list[str] = []
    charts: list[dict[str, Any]] = []

    need_stats = any(word in prompt for word in ("диспер", "медиан", "мат", "ожидан", "variance", "median", "mean"))
    need_trend = any(word in prompt for word in ("тренд", "trend", "линия"))

    if need_stats and numeric_cols:
        for col in numeric_cols[:4]:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            key_metrics.append(
                f"{col}: мат. ожидание={series.mean():.3f}, "
                f"дисперсия={series.var(ddof=0):.3f}, медиана={series.median():.3f}"
            )

    if need_trend and numeric_cols:
        x_candidates = [str(col) for col in df.columns if str(col) not in numeric_cols]
        x_col = "Location" if "Location" in df.columns else (x_candidates[0] if x_candidates else str(df.columns[0]))

        y_priority = ["Revenue", "Visitors", "Rating"]
        y_for_trend = [col for col in y_priority if col in numeric_cols][:2]
        if not y_for_trend:
            y_for_trend = [numeric_cols[0]]

        for y_col in y_for_trend:
            charts.append(
                {
                    "type": "line",
                    "title": f"Линия тренда: {y_col} по {x_col}",
                    "x": x_col,
                    "y": y_col,
                    "agg": "mean",
                    "top_n": 30,
                }
            )

    return {"summary": summary, "key_metrics": key_metrics[:8], "charts": charts[:2]}


def normalize_chart_spec(spec: dict[str, Any], df: pd.DataFrame) -> Optional[dict[str, Any]]:
    chart_type = str(spec.get("type", "")).strip().lower()
    if chart_type not in ALLOWED_CHART_TYPES:
        return None

    x_col = resolve_column_name(df, str(spec.get("x", "")))
    y_col = resolve_column_name(df, str(spec.get("y", "")))

    agg = str(spec.get("agg", "mean")).strip().lower()
    if agg not in ALLOWED_AGG:
        agg = "mean"

    try:
        top_n = int(spec.get("top_n", 20))
    except (TypeError, ValueError):
        top_n = 20
    top_n = max(5, min(top_n, 50))

    if chart_type == "histogram":
        if not x_col:
            return None
        return {
            "type": chart_type,
            "title": str(spec.get("title", "Распределение")).strip() or "Распределение",
            "x": x_col,
            "y": None,
            "agg": "count",
            "top_n": top_n,
        }

    if not x_col:
        return None

    if not y_col:
        agg = "count"

    title = str(spec.get("title", "")).strip()
    if not title:
        title = f"{chart_type.title()} по {x_col}"

    return {
        "type": chart_type,
        "title": title,
        "x": x_col,
        "y": y_col,
        "agg": agg,
        "top_n": top_n,
    }


def build_chart_figure(df: pd.DataFrame, spec: dict[str, Any]):
    # Все графики строим на стороне UI, модель только предлагает спецификацию.
    chart_type = spec["type"]
    x_col = spec["x"]
    y_col = spec.get("y")
    agg = spec.get("agg", "mean")
    top_n = int(spec.get("top_n", 20))
    title = spec.get("title", "График")

    if chart_type == "histogram":
        work = pd.DataFrame({x_col: pd.to_numeric(df[x_col], errors="coerce")}).dropna()
        if work.empty:
            return None
        fig = px.histogram(work, x=x_col, title=title)
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        return fig

    work = df.copy()
    work[x_col] = work[x_col].astype(str)

    if agg == "count" or not y_col:
        grouped = work.groupby(x_col, dropna=False).size().reset_index(name="Количество")
        value_col = "Количество"
    else:
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[y_col])
        if work.empty:
            return None
        grouped = work.groupby(x_col, dropna=False)[y_col].agg(agg).reset_index()
        value_col = y_col

    grouped = grouped.head(top_n) if chart_type == "line" else grouped.sort_values(value_col, ascending=False).head(top_n)

    if chart_type == "bar":
        fig = px.bar(grouped, x=x_col, y=value_col, title=title)
    elif chart_type == "line":
        fig = px.line(grouped, x=x_col, y=value_col, title=title, markers=True)
    elif chart_type == "scatter":
        fig = px.scatter(grouped, x=x_col, y=value_col, title=title)
    elif chart_type == "pie":
        fig = px.pie(grouped.head(top_n), names=x_col, values=value_col, title=title)
    else:
        return None

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
