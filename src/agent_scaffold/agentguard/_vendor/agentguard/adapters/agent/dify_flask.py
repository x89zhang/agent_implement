"""Shared Flask app registry for Dify adapters."""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

_dify_flask_app: Any | None = None
_app_ready_callbacks: list[Callable[[Any], None]] = []
_lock = threading.Lock()


def register_dify_flask_app(app: Any) -> bool:
    """Remember the Dify Flask app created by Dify itself."""
    if app is None or not callable(getattr(app, "app_context", None)):
        return False
    global _dify_flask_app
    with _lock:
        _dify_flask_app = app
        callbacks = list(_app_ready_callbacks)
        _app_ready_callbacks.clear()
    for callback in callbacks:
        try:
            callback(app)
        except Exception:
            pass
    return True


def get_dify_flask_app() -> Any | None:
    if _dify_flask_app is not None:
        return _dify_flask_app
    try:
        from flask import current_app  # type: ignore

        app = current_app._get_current_object()
    except Exception:
        return None
    register_dify_flask_app(app)
    return app


def on_dify_flask_app_ready(callback: Callable[[Any], None]) -> bool:
    """Run callback when Dify's Flask app has been captured."""
    with _lock:
        app = _dify_flask_app
        if app is None:
            _app_ready_callbacks.append(callback)
            return False
    try:
        callback(app)
    except Exception:
        pass
    return True


__all__ = ["get_dify_flask_app", "on_dify_flask_app_ready", "register_dify_flask_app"]
