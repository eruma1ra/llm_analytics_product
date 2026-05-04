from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import multiprocessing as mp
import traceback
from typing import Any

import pandas as pd
import plotly.express as px


MAX_TEXT_CHARS = 8000


SAFE_BUILTINS = {
    "Exception": Exception,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "slice": slice,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}


def _trim_text(value: Any, max_chars: int = MAX_TEXT_CHARS) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, pd.DataFrame):
        return value.head(50).fillna("").astype(str).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.head(50).fillna("").astype(str).to_dict()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in list(value)[:100]]
    return _trim_text(value)


def _worker(code: str, df: pd.DataFrame, queue: mp.Queue) -> None:
    stdout = StringIO()
    env: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "px": px,
        "df": df.copy(),
        "answer": "",
        "charts": [],
        "metrics": {},
    }

    try:
        with redirect_stdout(stdout):
            exec(code, env, env)
        queue.put(
            {
                "ok": True,
                "stdout": _trim_text(stdout.getvalue()),
                "answer": _trim_text(env.get("answer", "")),
                "metrics": _jsonable(env.get("metrics", {})),
                "charts": _jsonable(env.get("charts", [])),
            }
        )
    except Exception:
        queue.put(
            {
                "ok": False,
                "stdout": _trim_text(stdout.getvalue()),
                "error": _trim_text(traceback.format_exc(), max_chars=5000),
                "answer": "",
                "metrics": {},
                "charts": [],
            }
        )


def execute_python_analysis(
    df: pd.DataFrame,
    code: str,
    timeout_seconds: int = 12,
) -> dict[str, Any]:
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_worker, args=(code, df, queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(1)
        return {
            "ok": False,
            "stdout": "",
            "error": f"Код выполнялся дольше {timeout_seconds} секунд и был остановлен.",
            "answer": "",
            "metrics": {},
            "charts": [],
        }

    if queue.empty():
        return {
            "ok": False,
            "stdout": "",
            "error": "Интерпретатор не вернул результат.",
            "answer": "",
            "metrics": {},
            "charts": [],
        }
    return queue.get()
