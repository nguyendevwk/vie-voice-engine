"""
Fallback LLM Provider with automatic failover.

Tries providers in order until one succeeds.
Supports both generate and generate_stream with configurable timeout.

Usage:
    >>> from voice_assistant.core.llm_fallback import FallbackProvider, create_fallback_provider
    >>> provider = create_fallback_provider()
    >>> response = await provider.generate(messages)
    >>> health = await provider.health_check()
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Any, Optional

from .llm_base import BaseLLMProvider, LLMResponse, ToolDefinition
from ..config import settings
from ..utils.logging import logger, debug_log


@dataclass
class ProviderMetrics:
    """Metrics for a single provider."""
    name: str
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0
    last_error: Optional[str] = None
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None

    @property
    def avg_latency_ms(self) -> float:
        total = self.success_count + self.failure_count
        return self.total_latency_ms / total if total > 0 else 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def record_success(self, latency_ms: float):
        self.success_count += 1
        self.total_latency_ms += latency_ms
        self.last_success_time = time.time()

    def record_failure(self, error: str):
        self.failure_count += 1
        self.last_error = error
        self.last_failure_time = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "success_rate": round(self.success_rate, 3),
            "last_error": self.last_error,
        }


class FallbackProvider(BaseLLMProvider):
    """
    Wraps multiple providers and tries them in order.

    If a provider fails (network error, rate limit, auth error, etc.),
    the next provider in the chain is attempted automatically.
    """

    def __init__(
        self,
        providers: List[BaseLLMProvider],
        timeout_s: Optional[float] = None,
    ):
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self.providers = providers
        self.timeout_s = timeout_s or settings.llm.provider_timeout_s
        self._metrics: Dict[str, ProviderMetrics] = {
            p.__class__.__name__: ProviderMetrics(name=p.__class__.__name__)
            for p in providers
        }

    def _get_metrics(self, provider: BaseLLMProvider) -> ProviderMetrics:
        return self._metrics[provider.__class__.__name__]

    async def _run_with_timeout(self, coro, provider_name: str):
        """Run a coroutine with timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=self.timeout_s)
        except asyncio.TimeoutError:
            raise TimeoutError(f"Provider '{provider_name}' timed out after {self.timeout_s}s")

    async def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate response, falling back on failure."""
        last_error: Optional[Exception] = None

        for provider in self.providers:
            name = provider.__class__.__name__
            metrics = self._get_metrics(provider)
            start = time.perf_counter()
            try:
                debug_log(f"Trying provider: {name}")
                result = await self._run_with_timeout(
                    provider.generate(messages, **kwargs),
                    name,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record_success(latency_ms)
                debug_log(f"Provider {name} succeeded ({latency_ms:.0f}ms)")
                return result
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record_failure(str(e))
                last_error = e
                logger.warning(f"Provider {name} failed ({latency_ms:.0f}ms): {e}")
                continue

        raise RuntimeError(
            f"All providers exhausted. Last error: {last_error}"
        ) from last_error

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream response, falling back on failure."""
        last_error: Optional[Exception] = None

        for provider in self.providers:
            name = provider.__class__.__name__
            metrics = self._get_metrics(provider)
            start = time.perf_counter()
            try:
                debug_log(f"Trying stream provider: {name}")
                stream = provider.generate_stream(messages, **kwargs)
                async for chunk in stream:
                    yield chunk
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record_success(latency_ms)
                debug_log(f"Provider {name} stream completed ({latency_ms:.0f}ms)")
                return
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record_failure(str(e))
                last_error = e
                logger.warning(f"Provider {name} stream failed ({latency_ms:.0f}ms): {e}")
                continue

        raise RuntimeError(
            f"All stream providers exhausted. Last error: {last_error}"
        ) from last_error

    def supports_tools(self) -> bool:
        """Check if any provider supports tools."""
        return any(p.supports_tools() for p in self.providers)

    def supports_vision(self) -> bool:
        """Check if any provider supports vision."""
        return any(p.supports_vision() for p in self.providers)

    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[ToolDefinition],
        **kwargs
    ) -> LLMResponse:
        """Generate with tools, falling back on failure."""
        last_error: Optional[Exception] = None

        for provider in self.providers:
            if not provider.supports_tools():
                continue
            name = provider.__class__.__name__
            metrics = self._get_metrics(provider)
            start = time.perf_counter()
            try:
                debug_log(f"Trying tool provider: {name}")
                result = await self._run_with_timeout(
                    provider.generate_with_tools(messages, tools, **kwargs),
                    name,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record_success(latency_ms)
                debug_log(f"Tool provider {name} succeeded ({latency_ms:.0f}ms)")
                return result
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                metrics.record_failure(str(e))
                last_error = e
                logger.warning(f"Tool provider {name} failed ({latency_ms:.0f}ms): {e}")
                continue

        raise RuntimeError(
            f"All tool providers exhausted. Last error: {last_error}"
        ) from last_error

    def get_model_info(self) -> Dict[str, Any]:
        """Get info from the first available provider."""
        for provider in self.providers:
            try:
                info = provider.get_model_info()
                info["fallback_chain"] = [
                    p.__class__.__name__ for p in self.providers
                ]
                return info
            except Exception:
                continue
        return {"name": "fallback", "providers": len(self.providers)}

    def get_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for all providers."""
        return [m.to_dict() for m in self._metrics.values()]

    async def health_check(self) -> Dict[str, Any]:
        """
        Test each provider with a minimal request.

        Returns dict mapping provider name to health status.
        """
        results = {}
        test_messages = [{"role": "user", "content": "Say hi"}]

        for provider in self.providers:
            name = provider.__class__.__name__
            start = time.perf_counter()
            try:
                await self._run_with_timeout(
                    provider.generate(test_messages, max_tokens=5),
                    name,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                results[name] = {"status": "healthy", "latency_ms": round(latency_ms, 1)}
            except Exception as e:
                latency_ms = (time.perf_counter() - start) * 1000
                results[name] = {"status": "unhealthy", "error": str(e), "latency_ms": round(latency_ms, 1)}

        return results


def create_fallback_provider() -> Optional[FallbackProvider]:
    """
    Create a FallbackProvider from config.

    Reads settings.llm.fallback_providers to build the chain.
    Falls back through registered providers in order.
    Returns None if no providers are available.
    """
    from .llm_providers import _auto_register_providers
    from .llm_base import get_registry

    _auto_register_providers()
    registry = get_registry()
    config_chain = settings.llm.fallback_providers

    providers: List[BaseLLMProvider] = []
    for name in config_chain:
        provider = registry.get_provider(name)
        if provider:
            providers.append(provider)
            debug_log(f"Fallback chain: added {name}")
        else:
            logger.debug(f"Fallback chain: provider '{name}' not registered, skipping")

    if not providers:
        logger.warning("No providers available for fallback chain")
        return None

    return FallbackProvider(providers)
