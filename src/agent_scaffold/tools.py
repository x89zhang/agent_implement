from __future__ import annotations

import ast
import json
import operator as op
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


_OPS: dict[type[ast.AST], Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Num):  # py<3.8
        return node.n
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """
    Safe arithmetic calculator. Supports only numbers and + - * / // % ** ().
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string")
    cleaned = re.sub(r"[^0-9\.\+\-\*\/\%\(\)\s]", "", expression)
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError("Empty expression")
    tree = ast.parse(cleaned, mode="eval")
    return str(_eval(tree.body))


def web_search(query: str, max_results: int = 5) -> str:
    """
    Simple web search via DuckDuckGo Instant Answer API.
    Returns a JSON string of results with title, url, and snippet.
    """
    if not query:
        raise ValueError("Query is required")
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    }
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(params)
    data = _http_get_json(url)
    results: list[dict[str, str]] = []

    def _add_item(item: dict[str, Any]) -> None:
        text = str(item.get("Text", "")).strip()
        first_url = str(item.get("FirstURL", "")).strip()
        if text and first_url:
            results.append({"title": text, "url": first_url, "snippet": text})

    for item in data.get("Results", []) or []:
        if isinstance(item, dict):
            _add_item(item)

    for item in data.get("RelatedTopics", []) or []:
        if isinstance(item, dict) and "Topics" in item:
            for sub in item.get("Topics", []) or []:
                if isinstance(sub, dict):
                    _add_item(sub)
        elif isinstance(item, dict):
            _add_item(item)

    return json.dumps(results[: max(1, int(max_results))], ensure_ascii=False, indent=2)


def open_url(url: str, max_chars: int = 4000) -> str:
    """
    Fetch a URL and return raw text (truncated).
    """
    if not url:
        raise ValueError("URL is required")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "agent-scaffold/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read()
    text = content.decode("utf-8", errors="ignore")
    return text[: max(1, int(max_chars))]


def get_weather(city: str, start_date: str | None = None, end_date: str | None = None) -> str:
    """
    Get daily weather via Open-Meteo. Dates are optional (YYYY-MM-DD).
    """
    if not city:
        raise ValueError("City is required")
    geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": city, "count": 1, "language": "en", "format": "json"}
    )
    geo = _http_get_json(geo_url)
    results = geo.get("results") or []
    if not results:
        raise ValueError("City not found")
    loc = results[0]
    lat = loc["latitude"]
    lon = loc["longitude"]

    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    forecast_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    forecast = _http_get_json(forecast_url)
    return json.dumps(forecast, ensure_ascii=False, indent=2)


def write_text_file(path: str, content: str, mode: str = "w") -> str:
    """
    Write text content to a file under the current working directory.
    """
    if not path:
        raise ValueError("Path is required")
    base = Path.cwd().resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError("Path must be under the current working directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, mode, encoding="utf-8") as f:
        f.write(content)
    return str(target)


def _http_get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "agent-scaffold/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read()
    return json.loads(content.decode("utf-8", errors="ignore"))
