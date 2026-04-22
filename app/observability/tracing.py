from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from app.config import get_settings

logger = logging.getLogger("financial_advisor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class _NullSpan:
    trace_id: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> None:
        self.metadata.update(kwargs)

    def log_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        output: Any,
        usage: dict[str, Any] | None = None,
    ) -> None:
        logger.info(
            "generation | trace=%s | span=%s | name=%s | model=%s | usage=%s",
            self.trace_id,
            self.name,
            name,
            model,
            usage or {},
        )


class Tracer:
    """Thin wrapper around Langfuse — falls back to local logging when keys are absent."""

    def __init__(self) -> None:
        s = get_settings()
        self._enabled = bool(s.langfuse_public_key and s.langfuse_secret_key)
        self._client = None
        if self._enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=s.langfuse_public_key,
                    secret_key=s.langfuse_secret_key,
                    host=s.langfuse_host,
                )
                logger.info("Langfuse tracing enabled")
            except Exception as exc:  # pragma: no cover - optional dependency path
                logger.warning("Langfuse initialisation failed (%s); falling back to logs", exc)
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def trace(self, name: str, **metadata: Any) -> Iterator[_NullSpan]:
        trace_id = metadata.get("trace_id") or str(uuid.uuid4())
        span = _NullSpan(trace_id=trace_id, name=name, metadata=dict(metadata))
        lf_trace = None
        if self._enabled and self._client is not None:
            try:
                lf_trace = self._client.trace(id=trace_id, name=name, metadata=metadata)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to open Langfuse trace: %s", exc)
        logger.info("trace_start | name=%s | id=%s", name, trace_id)
        try:
            yield span
        finally:
            if lf_trace is not None:
                try:
                    lf_trace.update(metadata=span.metadata)
                except Exception:  # pragma: no cover
                    pass
            logger.info("trace_end   | name=%s | id=%s", name, trace_id)

    def log_generation(
        self,
        span: _NullSpan,
        *,
        name: str,
        model: str,
        input: Any,
        output: Any,
        usage: dict[str, Any] | None = None,
    ) -> None:
        span.log_generation(name=name, model=model, input=input, output=output, usage=usage)
        if self._enabled and self._client is not None:
            try:
                self._client.generation(
                    trace_id=span.trace_id,
                    name=name,
                    model=model,
                    input=input,
                    output=output,
                    usage=usage or {},
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to log Langfuse generation: %s", exc)

    def flush(self) -> None:
        if self._enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception:  # pragma: no cover
                pass


_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
