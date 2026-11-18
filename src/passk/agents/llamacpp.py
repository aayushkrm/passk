"""Local GGUF agent using llama-cpp-python.

``llama_cpp`` is imported lazily inside the constructor, so importing this module never
requires the native build. Install it with ``pip install -e '.[llama]'``.

The agent never calls ``llama_print_system_info`` (it triggers a slow Metal init) and always
runs with ``n_gpu_layers=0`` (CPU only).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from passk.agents._common import (
    ModelCallInfo,
    action_json_schema,
    build_messages,
    content_and_usage,
    parse_action,
)
from passk.schema import AgentAction, Step, Task, ToolSpec

__all__ = ["LlamaCppAgent", "action_json_schema", "build_messages"]

SEED_MODULUS = 2**31 - 1
"""Per-call seeds are reduced modulo this value (a positive 31-bit integer)."""


def _import_llama() -> Callable[..., Any]:
    """Import and return ``llama_cpp.Llama``.

    Raises:
        ImportError: If llama-cpp-python is not installed.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed; install it with `uv pip install -e '.[llama]'`"
        ) from exc
    return Llama


def _sha256_of(path: Path) -> str | None:
    """Return the SHA-256 of a model file, or None when the file is absent.

    The file is read once per agent (about 1 GB for the Q4_K_M models, a few seconds).
    A missing file is not an error here: the constructor reports it when the model loads.
    """
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class LlamaCppAgent:
    """Agent backed by ``llama_cpp.Llama.create_chat_completion`` with a JSON schema.

    Each model call receives the full message list (stateless), so trials do not share
    context. Per-call seed = episode seed + number of past steps, mod 2**31 - 1.
    """

    provider_name = "llama_cpp"

    def __init__(
        self,
        model_path: Path | str,
        repo_id: str | None = None,
        n_ctx: int = 4096,
        n_threads: int = 4,
        temperature: float = 0.7,
        seed: int = 0,
        prompt_variant: Literal["v1", "v2"] = "v1",
        max_tokens: int = 256,
        llama_factory: Callable[..., Any] | None = None,
        system_suffix: str = "",
    ) -> None:
        """Load the model.

        Args:
            model_path: Path to the GGUF file (weights live outside the repo).
            n_ctx: Context window in tokens.
            n_threads: CPU threads for inference.
            temperature: Sampling temperature.
            seed: Default sampling seed, used until ``begin_episode`` is called.
            prompt_variant: ``v1`` includes the policy section; ``v2`` is identical without it.
            max_tokens: Maximum generated tokens per call.
            llama_factory: Callable with the ``Llama`` constructor signature. Tests inject a
                fake here. Defaults to ``llama_cpp.Llama``, imported lazily.

        Raises:
            ImportError: If llama-cpp-python is not installed and no factory is given.
        """
        self.model_path = Path(model_path).expanduser()
        self.model_id = self.model_path.stem
        self.repo_id = repo_id
        self.system_suffix = system_suffix
        self.model_sha256 = _sha256_of(self.model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.temperature = temperature
        self.prompt_variant: Literal["v1", "v2"] = prompt_variant
        self.max_tokens = max_tokens
        self.last_call: ModelCallInfo | None = None
        self._episode_seed = seed
        factory = llama_factory if llama_factory is not None else _import_llama()
        self._llm = factory(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            seed=seed,
            verbose=False,
        )

    @property
    def agent_id(self) -> str:
        """Identifier written to trajectories, for example ``llamacpp:qwen2.5-0.5b-...``."""
        return f"llamacpp:{self.model_id}"

    @property
    def metadata(self) -> dict[str, Any]:
        """Run metadata for ``env.json``. Contains no paths and no secrets."""
        return {
            "kind": "llamacpp",
            "model": self.model_id,
            "repo_id": self.repo_id,
            "model_file": self.model_path.name,
            "model_sha256": self.model_sha256,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "temperature": self.temperature,
            "prompt_variant": self.prompt_variant,
            "system_suffix": self.system_suffix,
            "n_gpu_layers": 0,
        }

    def begin_episode(self, seed: int) -> None:
        """Set the episode seed; the runner calls this before each episode.

        Args:
            seed: Episode seed from ``runner.episode_seed``.
        """
        self._episode_seed = seed

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Ask the model for the next action; record tokens and latency in ``last_call``.

        Invalid model output does not raise. It becomes a schema-error action (see
        ``passk.agents._common``).

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available.

        Returns:
            The parsed action.
        """
        names = [spec.name for spec in tools]
        messages = build_messages(task, history, tools, self.prompt_variant)
        if self.system_suffix:
            # Qwen3's thinking switch (/no_think) is a directive in the system message.
            messages[0] = {
                **messages[0],
                "content": f"{messages[0]['content']} {self.system_suffix}",
            }
        call_seed = (self._episode_seed + len(history)) % SEED_MODULUS
        started = time.perf_counter()
        response = self._llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object", "schema": action_json_schema(names)},
            temperature=self.temperature,
            seed=call_seed,
            max_tokens=self.max_tokens,
        )
        latency = time.perf_counter() - started
        text, prompt_tokens, completion_tokens = content_and_usage(response)
        self.last_call = ModelCallInfo(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            latency_s=latency,
        )
        return parse_action(text, names, call_id=f"c{len(history)}")
