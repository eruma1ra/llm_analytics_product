from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.express as px


ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "histogram", "pie"}
ALLOWED_AGG = {"sum", "mean", "count", "median", "max", "min"}


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

    if not y_col:
        agg = "count"

    title = str(spec.get("title", "")).strip()
    if not title:
        if y_col:
            title = f"{chart_type.title()}: {y_col} по {x_col}"
        else:
            title = f"{chart_type.title()}: количество по {x_col}"

    return {
        "type": chart_type,
        "title": title,
        "x": x_col,
        "y": y_col,
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

    if chart_type == "histogram":
        work = pd.DataFrame({x_col: pd.to_numeric(df[x_col], errors="coerce")}).dropna()
        if work.empty:
            return None
        fig = px.histogram(work, x=x_col, title=title)
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

    if chart_type == "bar":
        fig = px.bar(grouped, x=x_col, y=value_col, title=title)
    elif chart_type == "line":
        fig = px.line(grouped, x=x_col, y=value_col, title=title, markers=True)
    elif chart_type == "scatter":
        fig = px.scatter(grouped, x=x_col, y=value_col, title=title)
    elif chart_type == "pie":
        fig = px.pie(grouped, names=x_col, values=value_col, title=title)
    else:
        return None

    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    return fig
