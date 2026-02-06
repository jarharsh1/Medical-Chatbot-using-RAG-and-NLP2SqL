"""
Ollama Health Monitoring & Graceful Error Handling

This module provides:
1. Health checks to verify Ollama is running and models are loaded
2. Graceful error messages when Ollama is unavailable
3. Caching to avoid hammering the health endpoint on every request
4. Model availability verification

Integration:
- Called by app.py before processing queries
- Exposed via /api/health endpoint for monitoring
- Used by agents to provide graceful fallbacks
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
import httpx

from backend.config import (
    LLM_MODEL,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_HEALTH_CACHE_TTL,
    OLLAMA_RETRY_ATTEMPTS,
    OLLAMA_RETRY_BACKOFF,
)

logger = logging.getLogger(__name__)

# Ollama API endpoints
OLLAMA_HEALTH_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_VERSION_ENDPOINT = f"{OLLAMA_BASE_URL}/api/version"

# Cache settings
_health_cache: Dict[str, Any] = {}
CACHE_TTL_SECONDS = OLLAMA_HEALTH_CACHE_TTL


class OllamaStatus(Enum):
    """Ollama service status."""
    HEALTHY = "healthy"           # Running and models loaded
    DEGRADED = "degraded"         # Running but missing models
    UNAVAILABLE = "unavailable"   # Not running or unreachable
    UNKNOWN = "unknown"           # Haven't checked yet


@dataclass
class HealthCheckResult:
    """Result of an Ollama health check."""
    status: OllamaStatus
    ollama_running: bool
    ollama_version: Optional[str]
    models_loaded: List[str]
    required_models: List[str]
    missing_models: List[str]
    latency_ms: float
    error_message: Optional[str]
    checked_at: float  # timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "ollama_running": self.ollama_running,
            "ollama_version": self.ollama_version,
            "models_loaded": self.models_loaded,
            "required_models": self.required_models,
            "missing_models": self.missing_models,
            "latency_ms": round(self.latency_ms, 2),
            "error_message": self.error_message,
            "checked_at": self.checked_at,
        }


def check_ollama_health(force_refresh: bool = False) -> HealthCheckResult:
    """
    Check if Ollama is running and required models are available.

    Uses caching to avoid excessive health checks (default 30s TTL).

    Args:
        force_refresh: Skip cache and perform fresh check

    Returns:
        HealthCheckResult with status and details
    """
    global _health_cache

    now = time.time()

    # Return cached result if valid
    if not force_refresh and _health_cache:
        cached_at = _health_cache.get("checked_at", 0)
        if now - cached_at < CACHE_TTL_SECONDS:
            return HealthCheckResult(**_health_cache)

    required_models = [LLM_MODEL, EMBED_MODEL]
    start_time = time.time()

    try:
        # Check if Ollama is running
        with httpx.Client(timeout=5.0) as client:
            # Get version
            version_response = client.get(OLLAMA_VERSION_ENDPOINT)
            version_response.raise_for_status()
            version_data = version_response.json()
            ollama_version = version_data.get("version", "unknown")

            # Get loaded models
            tags_response = client.get(OLLAMA_HEALTH_ENDPOINT)
            tags_response.raise_for_status()
            tags_data = tags_response.json()

        latency_ms = (time.time() - start_time) * 1000

        # Extract model names (handle both "name" and "model" formats)
        models = tags_data.get("models", [])
        loaded_models = []
        for m in models:
            name = m.get("name") or m.get("model", "")
            # Normalize: "qwen2.5:14b" matches "qwen2.5:14b"
            # Also handle cases like "qwen2.5:14b-instruct-q4_0"
            base_name = name.split("-")[0] if "-" in name else name
            loaded_models.append(base_name)

        # Check which required models are missing
        missing = []
        for req in required_models:
            # Check if any loaded model starts with the required model name
            found = any(
                loaded.startswith(req) or req.startswith(loaded.split(":")[0])
                for loaded in loaded_models
            )
            if not found:
                missing.append(req)

        # Determine status
        if not missing:
            status = OllamaStatus.HEALTHY
            error_msg = None
        else:
            status = OllamaStatus.DEGRADED
            error_msg = f"Missing models: {', '.join(missing)}. Run: ollama pull {' && ollama pull '.join(missing)}"

        result = HealthCheckResult(
            status=status,
            ollama_running=True,
            ollama_version=ollama_version,
            models_loaded=loaded_models,
            required_models=required_models,
            missing_models=missing,
            latency_ms=latency_ms,
            error_message=error_msg,
            checked_at=now,
        )

    except httpx.ConnectError:
        latency_ms = (time.time() - start_time) * 1000
        result = HealthCheckResult(
            status=OllamaStatus.UNAVAILABLE,
            ollama_running=False,
            ollama_version=None,
            models_loaded=[],
            required_models=required_models,
            missing_models=required_models,
            latency_ms=latency_ms,
            error_message="Ollama is not running. Start it with: ollama serve",
            checked_at=now,
        )
        logger.warning("Ollama health check failed: service not running")

    except httpx.TimeoutException:
        latency_ms = (time.time() - start_time) * 1000
        result = HealthCheckResult(
            status=OllamaStatus.UNAVAILABLE,
            ollama_running=False,
            ollama_version=None,
            models_loaded=[],
            required_models=required_models,
            missing_models=required_models,
            latency_ms=latency_ms,
            error_message="Ollama health check timed out. Service may be overloaded.",
            checked_at=now,
        )
        logger.warning("Ollama health check timed out")

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        result = HealthCheckResult(
            status=OllamaStatus.UNKNOWN,
            ollama_running=False,
            ollama_version=None,
            models_loaded=[],
            required_models=required_models,
            missing_models=required_models,
            latency_ms=latency_ms,
            error_message=f"Unexpected error: {str(e)}",
            checked_at=now,
        )
        logger.exception(f"Ollama health check error: {e}")

    # Cache the result
    _health_cache = {
        "status": result.status,
        "ollama_running": result.ollama_running,
        "ollama_version": result.ollama_version,
        "models_loaded": result.models_loaded,
        "required_models": result.required_models,
        "missing_models": result.missing_models,
        "latency_ms": result.latency_ms,
        "error_message": result.error_message,
        "checked_at": result.checked_at,
    }

    return result


def get_ollama_status() -> OllamaStatus:
    """Quick status check - returns cached status if available."""
    result = check_ollama_health()
    return result.status


def is_ollama_available() -> bool:
    """Simple boolean check - is Ollama healthy or degraded (usable)?"""
    status = get_ollama_status()
    return status in (OllamaStatus.HEALTHY, OllamaStatus.DEGRADED)


def get_user_friendly_error(status: OllamaStatus, error_message: Optional[str] = None) -> str:
    """
    Get a user-friendly error message for Ollama issues.

    Used by app.py to return helpful messages to the frontend.
    """
    if status == OllamaStatus.UNAVAILABLE:
        return (
            "The AI service (Ollama) is not running. "
            "Please start Ollama with `ollama serve` and try again."
        )
    elif status == OllamaStatus.DEGRADED:
        return (
            f"The AI service is running but some models are missing. "
            f"{error_message or 'Please pull the required models.'}"
        )
    elif status == OllamaStatus.UNKNOWN:
        return (
            "Unable to verify AI service status. "
            "Please ensure Ollama is installed and running."
        )
    return ""


# Retry decorator for LLM calls
def with_ollama_retry(max_retries: int = 2, backoff_seconds: float = 1.0):
    """
    Decorator for retrying Ollama calls on transient failures.

    Usage:
        @with_ollama_retry(max_retries=2)
        def call_llm(prompt):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except httpx.TimeoutException as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Ollama timeout (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {backoff_seconds}s..."
                        )
                        time.sleep(backoff_seconds * (attempt + 1))  # exponential backoff
                    continue
                except httpx.ConnectError as e:
                    # Don't retry connection errors - Ollama is down
                    raise
                except Exception as e:
                    # Don't retry unknown errors
                    raise
            # All retries exhausted
            raise last_error
        return wrapper
    return decorator
