"""Hosted-model agent for any OpenAI-compatible chat endpoint (default: OpenRouter).

Stdlib only (``urllib``). The HTTP call goes through an injectable ``post`` callable so tests
never touch the network.

API key handling: the key is read from the environment variable named by ``api_key_env`` at
construction time. It is never a constructor parameter, never stored in ``metadata``, and is
removed from every error message this module raises. Errors are raised ``from None`` so no
chained exception carries it either.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from passk.agents._common import (
    ModelCallInfo,
    build_messages,
    content_and_usage,
    parse_action,
)
from passk.schema import AgentAction, Step, Task, ToolSpec

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
REDACTED = "[REDACTED]"
MAX_ERROR_SNIPPET_CHARS = 300
MAX_RETRY_AFTER_S = 300.0
SEED_MODULUS = 2**31 - 1


class ProviderError(RuntimeError):
    """The provider request failed for good: a non-retryable status, or retries ran out.

    Raised out of ``act`` and therefore out of the run. It is not recorded as a model failure,
    because a provider outage says nothing about the agent.
    """


@dataclass(frozen=True)
class HttpResponse:
    """Minimal HTTP response returned by a ``PostFn``.

    Attributes:
        status: HTTP status code.
        headers: Response headers with lower-case names.
        body: Raw response body.
    """

    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""


PostFn = Callable[[str, dict[str, str], bytes, float], HttpResponse]
"""``post(url, headers, body, timeout_s) -> HttpResponse``. Must not raise on HTTP errors."""


def urllib_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> HttpResponse:
    """Send one POST with ``urllib`` and return the response, including non-2xx ones.

    Args:
        url: Full endpoint URL.
        headers: Request headers.
        body: Request body bytes.
        timeout: Socket timeout in seconds.

    Returns:
        The response. HTTP error statuses are returned, not raised.

    Raises:
        OSError: For connection failures and timeouts (``URLError`` and ``TimeoutError``
            are subclasses).
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return HttpResponse(
                status=response.status,
                headers={k.lower(): v for k, v in response.headers.items()},
                body=response.read(),
            )
    except urllib.error.HTTPError as err:
        return HttpResponse(
            status=err.code,
            headers={k.lower(): v for k, v in err.headers.items()},
            body=err.read(),
        )


def _is_retryable(status: int) -> bool:
    """Return True for statuses that are retried: 429 and every 5xx."""
    return status == 429 or status >= 500


class OpenAICompatAgent:
    """Agent that calls a hosted OpenAI-compatible chat endpoint with the JSON action prompt.

    Pacing: consecutive requests are at least ``min_interval_s`` apart. Retries: HTTP 429,
    5xx, timeouts and connection errors are retried with exponential backoff
    (``backoff_base_s * 2**attempt``, capped at ``backoff_max_s``). A ``Retry-After`` header
    in seconds takes precedence (capped at 300 s). Other 4xx responses are not retried.

    The provider-reported model id (the ``model`` field of each response) is collected in
    ``reported_models``. ``metadata`` exposes it for ``env.json``.
    """

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        temperature: float = 0.7,
        max_tokens: int = 256,
        seed: int = 0,
        min_interval_s: float = 3.0,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        backoff_max_s: float = 60.0,
        timeout_s: float = 120.0,
        prompt_variant: Literal["v1", "v2"] = "v1",
        provider_name: str = "openrouter",
        post: PostFn | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Configure the agent. Reads the API key from the environment.

        Args:
            model: Model id sent to the endpoint, for example
                ``nvidia/nemotron-3-ultra-550b-a55b:free``.
            base_url: API root; ``/chat/completions`` is appended.
            api_key_env: Name of the environment variable that holds the key.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens per call.
            seed: Base sampling seed; the runner overrides it per episode via
                ``begin_episode``. Sent as a best-effort ``seed`` field.
            min_interval_s: Minimum spacing between requests, in seconds.
            max_retries: Retries after the first attempt for retryable failures.
            backoff_base_s: Base delay of the exponential backoff, in seconds.
            backoff_max_s: Cap on a single backoff delay, in seconds.
            timeout_s: Per-request timeout, in seconds.
            prompt_variant: ``v1`` (with policy section) or ``v2`` (terse, no policy).
            provider_name: Value for ``gen_ai.provider.name`` in spans.
            post: HTTP callable. Defaults to ``urllib_post``. Tests inject fakes.
            sleep: Sleep function used for pacing and backoff. Tests inject a recorder.
            clock: Monotonic clock used for pacing and latency. Tests inject a fake.

        Raises:
            ValueError: If the API key environment variable is unset or empty. The message
                names the variable, never its value.
        """
        key = os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"environment variable {api_key_env!r} is not set")
        self._api_key = key
        self.model = model
        self.model_id = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_max_s = backoff_max_s
        self.timeout_s = timeout_s
        self.prompt_variant: Literal["v1", "v2"] = prompt_variant
        self.provider_name = provider_name
        self.last_call: ModelCallInfo | None = None
        self.reported_models: list[str] = []
        self._post: PostFn = post if post is not None else urllib_post
        self._sleep = sleep
        self._clock = clock
        self._episode_seed = seed
        self._last_sent: float | None = None

    @property
    def agent_id(self) -> str:
        """Identifier written to trajectories, for example ``openai_compat:<model>``."""
        return f"openai_compat:{self.model}"

    @property
    def metadata(self) -> dict[str, Any]:
        """Run metadata for ``env.json``. Contains the endpoint, models and settings; no key."""
        return {
            "kind": "openai_compat",
            "provider": self.provider_name,
            "base_url": self.base_url,
            "model_requested": self.model,
            "model_reported": list(self.reported_models),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "min_interval_s": self.min_interval_s,
            "max_retries": self.max_retries,
            "prompt_variant": self.prompt_variant,
        }

    def begin_episode(self, seed: int) -> None:
        """Set the episode seed; the runner calls this before each episode.

        Args:
            seed: Episode seed from ``runner.episode_seed``.
        """
        self._episode_seed = seed

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Ask the hosted model for the next action; record tokens and latency.

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available.

        Returns:
            The parsed action. Invalid model output becomes a schema-error action.

        Raises:
            ProviderError: If the request fails with a non-retryable status or retries run
                out. The message never contains the API key.
        """
        names = [spec.name for spec in tools]
        messages = build_messages(task, history, tools, self.prompt_variant)
        call_seed = (self._episode_seed + len(history)) % SEED_MODULUS
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "seed": call_seed,
        }
        started = self._clock()
        response = self._request(json.dumps(body).encode("utf-8"))
        latency = self._clock() - started
        text, prompt_tokens, completion_tokens = content_and_usage(response)
        reported = response.get("model")
        reported_id = reported if isinstance(reported, str) else None
        if reported_id is not None and reported_id not in self.reported_models:
            self.reported_models.append(reported_id)
        self.last_call = ModelCallInfo(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            latency_s=latency,
            model_reported=reported_id,
        )
        return parse_action(text, names, call_id=f"c{len(history)}")

    def _scrub(self, text: str) -> str:
        """Remove the API key from ``text`` and truncate it for safe error messages."""
        cleaned = text.replace(self._api_key, REDACTED)
        return cleaned[:MAX_ERROR_SNIPPET_CHARS]

    def _headers(self) -> dict[str, str]:
        """Build request headers. The key appears only here, never in logs or errors."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "passk",
        }

    def _pace(self) -> None:
        """Sleep so that consecutive request starts are at least ``min_interval_s`` apart."""
        if self._last_sent is None or self.min_interval_s <= 0:
            return
        wait = self.min_interval_s - (self._clock() - self._last_sent)
        if wait > 0:
            self._sleep(wait)

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        """Return the delay before retry number ``attempt + 1``."""
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except ValueError:
                seconds = -1.0
            if seconds >= 0:
                return min(seconds, MAX_RETRY_AFTER_S)
        return min(self.backoff_base_s * (2**attempt), self.backoff_max_s)

    def _provider_message(self, body: bytes) -> str:
        """Extract a short, key-free error description from a provider error body."""
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ""
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return self._scrub(error["message"])
        return ""

    def _request(self, payload: bytes) -> dict[str, Any]:
        """POST ``payload`` with pacing and retries; return the parsed JSON response object.

        Raises:
            ProviderError: On a non-retryable status, or after ``max_retries`` retries.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = self._headers()
        attempt = 0
        while True:
            self._pace()
            self._last_sent = self._clock()
            try:
                response = self._post(url, headers, payload, self.timeout_s)
            except OSError as exc:
                if attempt >= self.max_retries:
                    raise ProviderError(
                        f"network error after {attempt + 1} attempt(s): {type(exc).__name__}"
                    ) from None
                self._sleep(self._backoff(attempt, None))
                attempt += 1
                continue
            if 200 <= response.status < 300:
                return self._parse_body(response.body)
            if not _is_retryable(response.status):
                detail = self._provider_message(response.body)
                raise ProviderError(
                    f"provider returned HTTP {response.status}" + (f": {detail}" if detail else "")
                ) from None
            if attempt >= self.max_retries:
                raise ProviderError(
                    f"provider returned HTTP {response.status} after {attempt + 1} attempt(s)"
                ) from None
            self._sleep(self._backoff(attempt, response.headers.get("retry-after")))
            attempt += 1

    def _parse_body(self, body: bytes) -> dict[str, Any]:
        """Parse a successful response body.

        Raises:
            ProviderError: If the body is not a JSON object or carries an ``error`` field.
        """
        try:
            parsed = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ProviderError("provider returned a body that is not JSON") from None
        if not isinstance(parsed, dict):
            raise ProviderError("provider returned a JSON value that is not an object") from None
        if "error" in parsed:
            detail = self._provider_message(body)
            raise ProviderError(
                "provider returned an error object" + (f": {detail}" if detail else "")
            ) from None
        return parsed
