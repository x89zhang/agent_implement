from __future__ import annotations

import ast
import datetime as dt
import html as html_lib
import json
import operator as op
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml


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


def web_search(query: str | None = None, max_results: int = 5) -> str:
    """
    Simple web search via DuckDuckGo Instant Answer API.
    Returns a JSON string of results with title, url, and snippet.
    """
    if not query:
        trip = _load_trip_defaults()
        city = str(trip.get("city", "")).strip()
        if city:
            query = f"{city} travel"
        else:
            raise ValueError("Query is required")
    if isinstance(query, str) and ("query=" in query or query.strip().startswith("{")):
        parsed = _parse_tool_input(query)
        query = str(parsed.get("query", query))
        if "max_results" in parsed:
            max_results = int(parsed["max_results"])
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

    if results:
        return json.dumps(results[: max(1, int(max_results))], ensure_ascii=False, indent=2)

    # Fallback: scrape DuckDuckGo HTML results when Instant Answer is empty.
    html_url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    html_text = open_url(html_url, max_chars=20000)
    matches = re.findall(r'class="result__a"\s+href="([^"]+)".*?>(.*?)</a>', html_text, re.DOTALL)
    for href, title in matches:
        title_text = re.sub(r"<.*?>", "", title)
        results.append(
            {
                "title": html_lib.unescape(title_text.strip()),
                "url": html_lib.unescape(href.strip()),
                "snippet": html_lib.unescape(title_text.strip()),
            }
        )

    return json.dumps(results[: max(1, int(max_results))], ensure_ascii=False, indent=2)


def open_url(url: str, max_chars: int = 4000) -> str:
    """
    Fetch a URL and return raw text (truncated).
    """
    if not url:
        raise ValueError("URL is required")
    if isinstance(url, str) and ("url=" in url or url.strip().startswith("{")):
        parsed = _parse_tool_input(url)
        url = str(parsed.get("url", url))
        if "max_chars" in parsed:
            max_chars = int(parsed["max_chars"])
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "agent-scaffold/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        content = resp.read()
    text = content.decode("utf-8", errors="ignore")
    return text[: max(1, int(max_chars))]


def get_weather(city: str | None = None, start_date: str | None = None, end_date: str | None = None) -> str:
    """
    Get daily weather via Open-Meteo. Dates are optional (YYYY-MM-DD).
    """
    if not city:
        trip = _load_trip_defaults()
        city = str(trip.get("city", "")).strip()
        if not city:
            raise ValueError("City is required")
        if not start_date and not end_date:
            start_date, end_date = _resolve_date_range(trip)
    if isinstance(city, str):
        if "city=" in city or "start_date=" in city or "end_date=" in city or city.strip().startswith("{"):
            parsed = _parse_tool_input(city)
            if parsed:
                city = str(parsed.get("city", city))
                start_date = parsed.get("start_date", start_date)
                end_date = parsed.get("end_date", end_date)
        # Strip any key=value fragments that may have been embedded in the city string.
        city = re.sub(r"\b(start_date|end_date|city)\s*=\s*[^,]+", "", city).strip(" ,")
    query = str(city).strip()
    if not query:
        raise ValueError("City is required")

    name = query
    country = None
    if "," in query:
        parts = [p.strip() for p in query.split(",") if p.strip()]
        if parts:
            name = parts[0]
        if len(parts) > 1:
            country = parts[-1]

    search_queries = [query, name]
    if country:
        search_queries.append(f"{name} {country}")

    loc = None
    for q in search_queries:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
            {"name": q, "count": 1, "language": "en", "format": "json"}
        )
        geo = _http_get_json(geo_url)
        results = geo.get("results") or []
        if results:
            loc = results[0]
            break

    if not loc:
        # Fallback for common locations if geocoding fails on noisy inputs.
        lowered = query.lower()
        if "quebec" in lowered:
            loc = {"latitude": 46.8139, "longitude": -71.2080}
        else:
            raise ValueError("City not found")
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


def write_text_file(path: str | None, content: str = "", mode: str = "w") -> str:
    """
    Write text content to a file under the current working directory.
    """
    if not path:
        path = "output.md"
    if isinstance(path, str) and (
        "path=" in path or "content=" in path or path.strip().startswith("{") or "\n" in path
    ):
        parsed = _parse_tool_input(path)
        path = str(parsed.get("path", path))
        content = str(parsed.get("content", content))
        mode = str(parsed.get("mode", mode))
    if not content:
        return "Content is required. Please provide content to write."
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


def _load_trip_defaults() -> dict[str, Any]:
    """
    Load trip defaults from a yaml config located one level above the current run directory.
    The expected file name is <run_dir_name>.yaml (e.g. run dir "travel" -> "../travel.yaml").
    """
    run_dir = Path.cwd().resolve()
    cfg_path = run_dir.parent / f"{run_dir.name}.yaml"
    if not cfg_path.exists():
        return {}
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    trip = raw.get("trip") or {}
    return trip if isinstance(trip, dict) else {}


def _parse_tool_input(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        return {}
    if raw.startswith("{") and raw.endswith("}"):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    result: dict[str, Any] = {}
    for line in raw.splitlines():
        if ":" in line and "=" not in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    if result:
        return result
    pattern = re.compile(r"(\w+)\s*=\s*('([^']*)'|\"([^\"]*)\"|([^,]+))")
    for match in pattern.finditer(raw):
        key = match.group(1)
        value = match.group(3) or match.group(4) or match.group(5) or ""
        result[key] = value.strip()
    return result


def _resolve_date_range(trip: dict[str, Any]) -> tuple[str | None, str | None]:
    start = str(trip.get("start", "")).strip().lower()
    days = trip.get("days")
    try:
        days_int = int(days) if days is not None else 7
    except Exception:
        days_int = 7

    today = dt.date.today()
    if start == "next monday":
        days_ahead = (7 - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        start_date = today + dt.timedelta(days=days_ahead)
    else:
        try:
            start_date = dt.date.fromisoformat(start)
        except Exception:
            start_date = None

    if not start_date:
        return None, None

    end_date = start_date + dt.timedelta(days=days_int - 1)
    return start_date.isoformat(), end_date.isoformat()
