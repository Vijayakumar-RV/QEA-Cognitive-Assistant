# src/multi_agent_langgraph/utils/tracking/mlflow_traces.py
import time
import json
from contextlib import contextmanager

try:
    from mlflow.tracing import start_span as _mlflow_start_span  # MLflow >= 2.14
    _MLFLOW_TRACING_AVAILABLE = True
except Exception:
    _MLFLOW_TRACING_AVAILABLE = False
    _mlflow_start_span = None

from .mlflow_utils import log_json_artifact, log_text_artifact, log_metrics


@contextmanager
def span(name: str, inputs: dict | None = None, attrs: dict | None = None):
    """
    Context manager that:
      - Uses MLflow Tracing span if available (shows in 'Traces' tab)
      - Otherwise logs to artifacts under traces/<name>_*.json

    Usage:
      with span("agent_testcase", inputs={"prompt": "..."} ) as s:
          # ... work ...
          s.output({"answer_preview": "..."})
          s.metric("latency_sec", 0.123)
    """
    start = time.time()
    _span_ctx = None
    _span_obj = None

    if _MLFLOW_TRACING_AVAILABLE:
        try:
            _span_ctx = _mlflow_start_span(name)
            _span_obj = _span_ctx.__enter__()
            # Best-effort: attach inputs/attrs as attributes on the span
            if inputs:
                try:
                    _span_obj.set_attribute("inputs", json.dumps(inputs)[:4000])
                except Exception:
                    pass
            if attrs:
                try:
                    _span_obj.set_attribute("attrs", json.dumps(attrs)[:2000])
                except Exception:
                    pass
        except Exception:
            _span_ctx = None
            _span_obj = None
            # fallback to artifacts if tracing init fails
            if inputs:
                log_json_artifact(inputs, f"traces/{name}_inputs.json")
            if attrs:
                log_json_artifact(attrs, f"traces/{name}_attrs.json")
    else:
        # No tracing support: artifact fallback
        if inputs:
            log_json_artifact(inputs, f"traces/{name}_inputs.json")
        if attrs:
            log_json_artifact(attrs, f"traces/{name}_attrs.json")

    class _SpanProxy:
        def output(self, obj: dict | str, limit: int = 200000):
            if _span_obj:
                try:
                    # Attach as attribute (best-effort)
                    text = obj if isinstance(obj, str) else json.dumps(obj)
                    _span_obj.set_attribute("output", str(text)[:4000])
                except Exception:
                    pass
            else:
                # artifact fallback
                if isinstance(obj, str):
                    log_text_artifact(obj[:limit], f"traces/{name}_output.txt")
                else:
                    log_json_artifact(obj, f"traces/{name}_output.json")

        def metric(self, key: str, val: float):
            # Always emit a metric; shows up in Metrics even without Traces
            log_metrics({f"{name}.{key}": float(val)})

        def attribute(self, key: str, val: str | float | int):
            # Side attribute if needed
            if _span_obj:
                try:
                    _span_obj.set_attribute(str(key), str(val)[:1000])
                except Exception:
                    pass
            else:
                # As a quick fallback, store as text
                try:
                    log_text_artifact(f"{key}={val}", f"traces/{name}_attrs.txt")
                except Exception:
                    pass

    proxy = _SpanProxy()
    try:
        yield proxy
    finally:
        elapsed = time.time() - start
        proxy.metric("latency_sec", elapsed)
        if _span_ctx:
            _span_ctx.__exit__(None, None, None)
