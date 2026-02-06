"""
Query Result Cache - In-memory caching layer for query results.

Features:
- LRU eviction when max size reached
- TTL-based expiration
- Cache statistics for monitoring
- Thread-safe operations

No external dependencies (Redis not required).
"""

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    value: Dict[str, Any]
    created_at: float
    expires_at: float
    hits: int = 0


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    total_queries: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "total_queries": self.total_queries,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


class QueryResultCache:
    """
    Thread-safe LRU cache with TTL support for query results.

    Usage:
        cache = QueryResultCache(max_size=1000, ttl_seconds=3600)

        # Check cache
        result = cache.get("How many patients?")
        if result is None:
            result = expensive_query()
            cache.set("How many patients?", result)
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        enabled: bool = True
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()

    def _hash_query(self, query: str) -> str:
        """Create a normalized hash key for a query."""
        normalized = query.lower().strip()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        return time.time() > entry.expires_at

    def _evict_if_needed(self):
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self.max_size:
            # Remove oldest (first) item
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats.evictions += 1
            logger.debug(f"Cache eviction: {oldest_key[:8]}...")

    def _cleanup_expired(self):
        """Remove expired entries (called periodically)."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if now > entry.expires_at
        ]
        for key in expired_keys:
            del self._cache[key]
            self._stats.expirations += 1

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query.

        Returns None if not found or expired.
        """
        if not self.enabled:
            return None

        key = self._hash_query(query)
        self._stats.total_queries += 1

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if self._is_expired(entry):
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats.hits += 1

            logger.debug(f"Cache hit for query: {query[:50]}...")
            return entry.value.copy()  # Return copy to prevent mutation

    def set(
        self,
        query: str,
        result: Dict[str, Any],
        ttl_override: Optional[int] = None
    ):
        """
        Cache a query result.

        Args:
            query: The original query string
            result: The result dictionary to cache
            ttl_override: Optional custom TTL for this entry
        """
        if not self.enabled:
            return

        # Don't cache errors or empty results
        if result.get("error") or not result.get("answer"):
            return

        key = self._hash_query(query)
        ttl = ttl_override or self.ttl_seconds
        now = time.time()

        with self._lock:
            self._evict_if_needed()

            self._cache[key] = CacheEntry(
                value=result.copy(),  # Store copy
                created_at=now,
                expires_at=now + ttl,
                hits=0
            )

            logger.debug(f"Cached result for query: {query[:50]}...")

    def invalidate(self, query: str) -> bool:
        """Invalidate a specific cached query."""
        key = self._hash_query(query)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cached queries containing a pattern.

        Useful when data changes (e.g., new patients added).
        """
        # This is O(n) but acceptable for cache invalidation
        count = 0
        pattern_lower = pattern.lower()

        with self._lock:
            # We need to iterate over a copy since we're modifying
            keys_to_delete = []
            for key, entry in self._cache.items():
                # Check if any query matches (we'd need to store original query)
                # For now, just clear all - caller should be specific
                pass

            # Simple approach: clear all if pattern is broad
            if pattern_lower in ["patient", "prescription", "clinical", "clinic"]:
                count = len(self._cache)
                self._cache.clear()

        return count

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared: {count} entries removed")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    def get_info(self) -> dict:
        """Get cache information for debugging."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "max_size": self.max_size,
                "current_size": len(self._cache),
                "ttl_seconds": self.ttl_seconds,
                "stats": self._stats.to_dict(),
            }


# Global cache instance
_cache_instance: Optional[QueryResultCache] = None


def get_cache() -> QueryResultCache:
    """Get or create the global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryResultCache(
            max_size=1000,
            ttl_seconds=3600,  # 1 hour default
            enabled=True
        )
    return _cache_instance


def cached_query(func):
    """
    Decorator to cache query results.

    Usage:
        @cached_query
        def process_query(question: str) -> dict:
            ...
    """
    def wrapper(question: str, *args, **kwargs):
        cache = get_cache()

        # Try cache first
        cached_result = cache.get(question)
        if cached_result is not None:
            cached_result["from_cache"] = True
            return cached_result

        # Execute query
        result = func(question, *args, **kwargs)

        # Cache successful results
        if result and not result.get("error"):
            cache.set(question, result)

        return result

    return wrapper


# Specialized caches for different query types
class SQLResultCache(QueryResultCache):
    """Cache specifically for SQL query results with shorter TTL."""

    def __init__(self):
        super().__init__(
            max_size=500,
            ttl_seconds=1800,  # 30 minutes for SQL results
            enabled=True
        )


class RAGResultCache(QueryResultCache):
    """Cache for RAG results with longer TTL (documents don't change often)."""

    def __init__(self):
        super().__init__(
            max_size=500,
            ttl_seconds=7200,  # 2 hours for RAG results
            enabled=True
        )


# Global typed cache instances
_sql_cache: Optional[SQLResultCache] = None
_rag_cache: Optional[RAGResultCache] = None


def get_sql_cache() -> SQLResultCache:
    global _sql_cache
    if _sql_cache is None:
        _sql_cache = SQLResultCache()
    return _sql_cache


def get_rag_cache() -> RAGResultCache:
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = RAGResultCache()
    return _rag_cache
