# utils/timers.py
from time import perf_counter
from contextlib import contextmanager

@contextmanager
def track_latency(metric_name: str, log_metrics):
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        try:
            log_metrics({metric_name: float(dt)})
        except Exception:
            pass
