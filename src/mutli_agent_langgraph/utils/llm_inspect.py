# src/mutli_agent_langgraph/utils/llm_inspect.py
from typing import Any, Tuple, Dict, Optional
import re

def _get_first_attr(obj: Any, names: list[str]) -> Optional[Any]:
    for n in names:
        if hasattr(obj, n):
            try:
                return getattr(obj, n)
            except Exception:
                pass
    return None

def _lower(s: Any) -> str:
    try:
        return str(s).lower()
    except Exception:
        return ""

def _deployment_from_endpoint(url: str) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"/deployments/([^/]+)/", url)
    return m.group(1) if m else None

def infer_llm_metadata(llm: Any) -> Tuple[str, str, Optional[float], Dict]:
    """
    Returns (vendor, model_name, temperature, extras) for various wrappers
    including Azure OpenAI, OpenAI, Groq, Google, Ollama, etc.
    """
    if llm is None:
        return ("unknown", "unknown", None, {})

    cls = llm.__class__
    mod = _lower(cls.__module__)
    name = _lower(cls.__name__)

    # Vendor guess
    vendor = "unknown"
    if "openai" in mod or "openai" in name:
        vendor = "openai"
    if "groq" in mod or "groq" in name:
        vendor = "groq"
    if "google" in mod or "generativelanguage" in mod or "gemini" in mod:
        vendor = "google"
    if "ollama" in mod or "ollama" in name:
        vendor = "ollama"

    # Gather extras we might want to log
    extras: Dict[str, Any] = {}
    for k in [
        "openai_api_type", "openai_api_version", "azure_endpoint", "deployment_name",
        "streaming", "top_p", "top_k", "max_tokens", "temperature"
    ]:
        if hasattr(llm, k):
            try:
                extras[k] = getattr(llm, k)
            except Exception:
                pass

    # Temperature (common attr)
    temperature = _get_first_attr(llm, ["temperature"])
    if temperature is None:
        # some wrappers expose only top_p; not a temperature but useful signal
        temperature = _get_first_attr(llm, ["top_p"])

    # Model name logic
    # First, generic candidates
    model_name = _get_first_attr(llm, ["model_name", "model", "deployment_name", "azure_deployment", "azure_deployment_name", "deployment"])

    # Azure-specific fallback
    openai_type = _get_first_attr(llm, ["openai_api_type"])
    if (not model_name) and (vendor == "openai") and (str(openai_type).lower() == "azure"):
        # Try parse from endpoint
        azure_endpoint = _get_first_attr(llm, ["azure_endpoint"])
        parsed = _deployment_from_endpoint(str(azure_endpoint) if azure_endpoint else "")
        if parsed:
            model_name = parsed

    # Final fallback
    if not model_name:
        model_name = "unknown"

    return (vendor, str(model_name), temperature, extras)
