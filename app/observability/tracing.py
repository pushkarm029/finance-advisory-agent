from __future__ import annotations

import hashlib
import json
import logging
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import get_settings


def _build_logger() -> logging.Logger:
    lg = logging.getLogger("financial_advisor")
    if lg.handlers:
        return lg
    handler = logging.StreamHandler(sys.stderr)
    if get_settings().is_prod:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                payload = {
                    "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                return json.dumps(payload)

        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg


logger = _build_logger()

_lf_client = None
_lf_init_attempted = False


def _get_langfuse():
    global _lf_client, _lf_init_attempted
    if _lf_init_attempted:
        return _lf_client
    _lf_init_attempted = True

    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        _lf_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("langfuse_enabled | host=%s", settings.langfuse_host)
    except Exception as exc:
        logger.warning("langfuse_init_failed | %s", exc)
        _lf_client = None
    return _lf_client


class Trace:
    def __init__(self, trace_id: str, trace_url: str | None = None) -> None:
        self.trace_id = trace_id
        self.trace_url = trace_url

    def update(self, **metadata: Any) -> None:
        lf = _get_langfuse()
        if lf is not None:
            try:
                lf.update_current_span(metadata=metadata)
            except Exception as exc:
                logger.debug("update_current_span_failed | %s", exc)
        logger.info("trace_update | id=%s | %s", self.trace_id, metadata)


def _fallback_trace_id(name: str, payload: Any) -> str:
    digest = hashlib.sha256(f"{name}|{payload}".encode("utf-8")).hexdigest()
    return digest[:32]


@contextmanager
def trace_span(name: str, **input_data: Any) -> Iterator[Trace]:
    lf = _get_langfuse()
    logger.info("trace_start | name=%s | input=%s", name, input_data)
    if lf is not None:
        with lf.start_as_current_observation(name=name, as_type="span", input=input_data):
            tid = lf.get_current_trace_id() or _fallback_trace_id(name, input_data)
            try:
                url = lf.get_trace_url()
            except Exception:
                url = None
            handle = Trace(trace_id=tid, trace_url=url)
            try:
                yield handle
            finally:
                logger.info("trace_end | name=%s | id=%s", name, handle.trace_id)
    else:
        handle = Trace(trace_id=_fallback_trace_id(name, input_data), trace_url=None)
        try:
            yield handle
        finally:
            logger.info("trace_end | name=%s | id=%s", name, handle.trace_id)


def openai_client():
    """OpenAI client auto-instrumented for Langfuse when enabled."""
    settings = get_settings()
    if settings.langfuse_enabled and _get_langfuse() is not None:
        from langfuse.openai import OpenAI
    else:
        from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key, timeout=30.0)


def flush() -> None:
    lf = _get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass
