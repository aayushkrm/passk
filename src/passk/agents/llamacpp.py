"""Local GGUF agent using llama-cpp-python.

``llama_cpp`` is imported lazily inside the constructor, so importing this module never
requires the native build. Install it with ``pip install -e '.[llama]'``.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from passk.schema import AgentAction, Step, Task, ToolSpec


def action_json_schema(tool_names: Sequence[str]) -> dict[str, Any]:
    """Build the JSON schema that constrains the model's action output.

    Shape: ``{"kind": "tool_call" | "final", "tool": enum(tool_names),
    "arguments": object, "message": string}``. Kept flat to limit grammar compile time.

    Args:
        tool_names: Names of the tools the model may call.

    Returns:
        A JSON schema object.
    """
    raise NotImplementedError("action_json_schema: scheduled for work package P-D")


def build_messages(
    task: Task,
    history: Sequence[Step],
    tools: Sequence[ToolSpec],
    prompt_variant: Literal["v1", "v2"],
) -> list[dict[str, str]]:
    """Render the chat messages for one model call.

    Args:
        task: Task being solved.
        history: Steps taken so far.
        tools: Tools available to the agent.
        prompt_variant: ``v1`` includes the policy section; ``v2`` has terse tool
            descriptions and no policy section.

    Returns:
        Chat messages in the OpenAI-style ``{"role", "content"}`` format.
    """
    raise NotImplementedError("build_messages: scheduled for work package P-D")


class LlamaCppAgent:
    """Agent backed by ``llama_cpp.Llama.create_chat_completion`` with a JSON schema."""

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 4096,
        n_threads: int = 4,
        temperature: float = 0.7,
        seed: int = 0,
        prompt_variant: Literal["v1", "v2"] = "v1",
    ) -> None:
        """Load the model.

        Args:
            model_path: Path to the GGUF file (weights live outside the repo).
            n_ctx: Context window in tokens.
            n_threads: CPU threads for inference.
            temperature: Sampling temperature.
            seed: Sampling seed.
            prompt_variant: Prompt variant, see ``build_messages``.

        Raises:
            ImportError: If llama-cpp-python is not installed.
        """
        raise NotImplementedError("LlamaCppAgent: scheduled for work package P-D")

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Ask the model for the next action; record tokens and latency on the step.

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available.

        Returns:
            The parsed action.
        """
        raise NotImplementedError("LlamaCppAgent.act: scheduled for work package P-D")
