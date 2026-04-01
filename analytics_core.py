from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd
import plotly.express as px


ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "pie"}
ALLOWED_AGG = {"sum", "mean", "count", "median", "max", "min"}
UNNAMED_RE = re.compile(r"^Unnamed:\s*\d+$", re.IGNORECASE)
CHART_COUNT_RE = re.compile(
    r"\b(?P<count>\d{1,2})\s*(?:график(?:а|ов)?|диаграмм(?:а|ы|у|е|)|charts?|plots?)\b",
    flags=re.IGNORECASE,
)
CHART_WORD_COUNT_RE = re.compile(
    r"\b(?P<count_word>один|одну|одного|два|две|три|четыре|пять|one|two|three|four|five)\s*"
    r"(?:график(?:а|ов)?|диаграмм(?:а|ы|у|е|)|charts?|plots?)\b",
    flags=re.IGNORECASE,
)
WORD_TO_NUMBER = {
    "один": 1,
    "одну": 1,
    "одного": 1,
    "два": 2,
    "две": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}
CHART_TYPE_MARKERS: dict[str, tuple[str, ...]] = {
    "pie": ("кругов", "кольцев", "pie", "donut", "доля", "структур"),
    "line": ("линейн", "линия", "тренд", "динам", "line"),
    "bar": ("столбч", "bar", "column"),
    "scatter": ("точеч", "scatter"),
    "histogram": ("гистограмм", "histogram"),
}


def _display_label(name: str, fallback: str) -> str:
    text = str(name or "").strip()
    if not text:
        return fallback
    if UNNAMED_RE.match(text):
        return fallback
    return text


def infer_requested_chart_limit(prompt: str, default_limit: int = 5) -> int:
    text = (prompt or "").lower()
    limit = max(1, min(int(default_limit), 5))

    m_num = CHART_COUNT_RE.search(text)
    if m_num:
        try:
            requested = int(m_num.group("count"))
            return max(1, min(requested, 5))
        except Exception:
            return limit

    m_word = CHART_WORD_COUNT_RE.search(text)
    if m_word:
        requested = WORD_TO_NUMBER.get(m_word.group("count_word").lower().strip())
        if requested is not None:
            return max(1, min(int(requested), 5))

    singular_chart = re.search(
        r"\b(график|диаграмм[ауе]|chart|plot)\b",
        text,
        flags=re.IGNORECASE,
    )
    plural_chart = re.search(
        r"\b(графики|диаграммы|charts|plots|несколько|multiple|many)\b",
        text,
        flags=re.IGNORECASE,
    )
    two_charts_phrase = re.search(
        r"(?:график|диаграмм[ауе]|chart|plot)\s*(?:,|\sи\s|\sand\s).*(?:график|диаграмм[ауе]|chart|plot)",
        text,
        flags=re.IGNORECASE,
    )
    if singular_chart and not plural_chart and not two_charts_phrase:
        return 1

    return limit


def infer_requested_chart_types(prompt: str) -> list[str]:
    text = (prompt or "").lower()
    found: list[tuple[int, str]] = []
    for chart_type, markers in CHART_TYPE_MARKERS.items():
        positions = [text.find(marker) for marker in markers if text.find(marker) >= 0]
        if positions:
            found.append((min(positions), chart_type))
    found.sort(key=lambda item: item[0])
    ordered_types: list[str] = []
    for _, chart_type in found:
        if chart_type not in ordered_types:
            ordered_types.append(chart_type)
    return ordered_types


def select_chart_specs_by_prompt(
    specs: list[dict[str, Any]],
    prompt: str,
    default_limit: int = 5,
) -> list[dict[str, Any]]:
    if not specs:
        return []

    limit = infer_requested_chart_limit(prompt, default_limit=default_limit)
    requested_types = infer_requested_chart_types(prompt)
    candidates = list(specs)

    if requested_types:
        filtered = [
            item
            for item in candidates
            if str(item.get("type", "")).strip().lower() in requested_types
        ]
        if filtered:
            order = {name: idx for idx, name in enumerate(requested_types)}
            filtered.sort(
                key=lambda item: order.get(str(item.get("type", "")).strip().lower(), 999)
            )
            candidates = filtered
        else:
            return []

    return candidates[:limit]


def wants_chart_request(prompt: str) -> bool:
    text = (prompt or "").lower()
    markers = ("график", "диаграм", "чарт", "chart", "plot", "plotly", "тренд", "trend")
    return any(marker in text for marker in markers)


def resolve_column_name(df: pd.DataFrame, requested: str) -> Optional[str]:
    if not requested or not str(requested).strip():
        return None
    requested = str(requested).strip()
    columns = [str(col) for col in df.columns]
    if requested in columns:
        return requested
    mapping = {col.lower(): col for col in columns}
    return mapping.get(requested.lower())


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
        top_n = int(spec.get("top_n", 30))
    except (TypeError, ValueError):
        top_n = 30
    top_n = max(5, min(top_n, 200))

    if chart_type == "histogram":
        if not x_col:
            return None
        title = str(spec.get("title", "")).strip() or f"Распределение: {x_col}"
        return {"type": "histogram", "title": title, "x": x_col, "y": None, "agg": "count", "top_n": top_n}

    if not x_col:
        return None

    if chart_type in {"bar", "line", "scatter", "pie"} and not y_col:
        guessed_value_col = _pick_quantity_column(df)
        if guessed_value_col and guessed_value_col != x_col:
            y_col = guessed_value_col
            # Если модель не передала y, count почти всегда дает "бесполезный" график.
            # Для числовой метрики по умолчанию используем sum.
            if agg == "count":
                agg = "sum"

    if not y_col:
        agg = "count"

    x_label = str(spec.get("x_label", "")).strip() or _display_label(x_col, "Категория")
    if y_col:
        y_label = str(spec.get("y_label", "")).strip() or _display_label(y_col, "Значение")
    else:
        y_label = str(spec.get("y_label", "")).strip() or "Количество"

    title = str(spec.get("title", "")).strip()
    if not title:
        if y_col:
            title = f"{chart_type.title()}: {y_label} по {x_label}"
        else:
            title = f"{chart_type.title()}: количество по {x_label}"

    return {
        "type": chart_type,
        "title": title,
        "x": x_col,
        "y": y_col,
        "x_label": x_label,
        "y_label": y_label,
        "agg": agg,
        "top_n": top_n,
    }


def normalize_chart_specs(specs: list[dict[str, Any]], df: pd.DataFrame, max_charts: int = 5) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    limit = max(0, min(int(max_charts), 5))
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        item = normalize_chart_spec(spec, df)
        if not item:
            continue
        normalized.append(item)
        if len(normalized) >= limit:
            break
    return normalized


def _pick_group_column(df: pd.DataFrame) -> str:
    names = [str(col) for col in df.columns]
    low = {str(col).lower(): str(col) for col in df.columns}
    for marker in ("группа", "group", "category", "категор"):
        for key, real in low.items():
            if marker in key:
                return real
    object_cols = [str(col) for col in df.select_dtypes(exclude="number").columns]
    return object_cols[0] if object_cols else names[0]


def _pick_quantity_column(df: pd.DataFrame) -> Optional[str]:
    numeric_cols = [str(col) for col in df.select_dtypes(include="number").columns]
    if not numeric_cols:
        converted = df.copy()
        for col in converted.columns:
            converted[col] = pd.to_numeric(converted[col], errors="coerce")
        numeric_cols = [str(col) for col in converted.columns if converted[col].notna().sum() > 0]
    if not numeric_cols:
        return None
    low = {str(col).lower(): str(col) for col in numeric_cols}
    for marker in ("остаток", "колич", "qty", "count", "stock", "amount", "revenue", "sales"):
        for key, real in low.items():
            if marker in key:
                return real
    return numeric_cols[0]


def _pick_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        name = str(col).lower()
        if "date" in name or "дата" in name:
            return str(col)
    return None


def build_fallback_chart_specs(prompt: str, df: pd.DataFrame, max_charts: int = 5) -> list[dict[str, Any]]:
    if df.empty:
        return []
    group_col = _pick_group_column(df)
    qty_col = _pick_quantity_column(df)
    date_col = _pick_date_column(df)

    specs: list[dict[str, Any]] = []
    text = (prompt or "").lower()

    wants_trend = "тренд" in text or "trend" in text or "динам" in text
    wants_group = "групп" in text or "category" in text or "диаграм" in text or "pie" in text

    if wants_trend and qty_col:
        x_for_trend = date_col or group_col
        specs.append(
            {
                "type": "line",
                "title": f"Тренд {qty_col} по {x_for_trend}",
                "x": x_for_trend,
                "y": qty_col,
                "agg": "sum",
                "top_n": 100,
            }
        )

    if wants_group:
        if qty_col:
            specs.append(
                {
                    "type": "pie",
                    "title": f"Распределение {qty_col} по {group_col}",
                    "x": group_col,
                    "y": qty_col,
                    "agg": "sum",
                    "top_n": 30,
                }
            )
            specs.append(
                {
                    "type": "bar",
                    "title": f"{qty_col} по {group_col}",
                    "x": group_col,
                    "y": qty_col,
                    "agg": "sum",
                    "top_n": 30,
                }
            )
        else:
            specs.append(
                {
                    "type": "pie",
                    "title": f"Распределение по {group_col}",
                    "x": group_col,
                    "y": None,
                    "agg": "count",
                    "top_n": 30,
                }
            )
            specs.append(
                {
                    "type": "bar",
                    "title": f"Количество по {group_col}",
                    "x": group_col,
                    "y": None,
                    "agg": "count",
                    "top_n": 30,
                }
            )

    if not specs:
        specs.append(
            {
                "type": "bar",
                "title": f"Количество по {group_col}",
                "x": group_col,
                "y": None,
                "agg": "count",
                "top_n": 30,
            }
        )
        if qty_col:
            specs.append(
                {
                    "type": "line",
                    "title": f"{qty_col} по {group_col}",
                    "x": group_col,
                    "y": qty_col,
                    "agg": "sum",
                    "top_n": 50,
                }
            )

    return normalize_chart_specs(specs, df, max_charts=max_charts)


def build_chart_figure(df: pd.DataFrame, spec: dict[str, Any]):
    chart_type = spec["type"]
    x_col = spec["x"]
    y_col = spec.get("y")
    agg = spec.get("agg", "mean")
    top_n = int(spec.get("top_n", 30))
    title = spec.get("title", "График")
    x_label = str(spec.get("x_label", "")).strip() or _display_label(x_col, "Категория")

    if chart_type == "histogram":
        work = pd.DataFrame({x_col: pd.to_numeric(df[x_col], errors="coerce")}).dropna()
        if work.empty:
            return None
        fig = px.histogram(work, x=x_col, title=title, labels={x_col: x_label})
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        return fig

    work = df.copy()
    if chart_type != "line":
        work[x_col] = work[x_col].astype(str)

    if agg == "count" or not y_col:
        grouped = work.groupby(x_col, dropna=False, sort=False).size().reset_index(name="Количество")
        value_col = "Количество"
    else:
        work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
        work = work.dropna(subset=[y_col])
        if work.empty:
            return None
        grouped = work.groupby(x_col, dropna=False, sort=False)[y_col].agg(agg).reset_index()
        value_col = y_col

    if chart_type == "line":
        grouped = grouped.head(top_n)
    else:
        grouped = grouped.sort_values(value_col, ascending=False).head(top_n)

    y_label = str(spec.get("y_label", "")).strip()
    if not y_label:
        if y_col:
            y_label = _display_label(value_col, "Значение")
        else:
            y_label = "Количество"

    if chart_type == "bar":
        fig = px.bar(grouped, x=x_col, y=value_col, title=title, labels={x_col: x_label, value_col: y_label})
    elif chart_type == "line":
        fig = px.line(grouped, x=x_col, y=value_col, title=title, markers=True, labels={x_col: x_label, value_col: y_label})
    elif chart_type == "scatter":
        fig = px.scatter(grouped, x=x_col, y=value_col, title=title, labels={x_col: x_label, value_col: y_label})
    elif chart_type == "pie":
        fig = px.pie(grouped, names=x_col, values=value_col, title=title, labels={x_col: x_label, value_col: y_label})
    else:
        return None

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
