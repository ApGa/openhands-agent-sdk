"""Programmatic tool calling through a persistent embedded IPython shell."""

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field
from rich.text import Text

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


TOOL_DESCRIPTION = """Execute Python code in a persistent IPython environment.

Use this when you need Python control flow, variables, loops, or helper functions
to inspect and compose other OpenHands tools. The Python namespace persists across
calls to this tool.

All active OpenHands tools are available as Python callables:
- Call identifier-safe tools directly, e.g. `terminal(command="pwd")`.
- Use `tools.<tool_name>(...)` for identifier-safe names.
- Use `tools["tool-name"](...)` for names that are not valid Python identifiers.
- Call `tools.available()` to list callable tools.
- Use `await atools.<tool_name>(...)` or `await acall_tool("tool_name", ...)`
  when composing calls from async Python code.
- `asyncio` is preloaded for concurrent orchestration.

Tool functions accept keyword arguments matching the tool schema and return the
tool's typed Observation object. For example:

```python
result = terminal(command="pwd")
print(result.text)

results = await asyncio.gather(
    atools.glob(pattern="**/*.py"),
    atools.grep(pattern="ProgrammaticToolCallingTool"),
)
```

The programmatic tool cannot call itself recursively.
"""

ORCHESTRATION_ONLY_DESCRIPTION = """

This Python runtime is outside the task environment and is available only for
orchestrating OpenHands tools. Do not use local filesystem, operating-system,
process, shell, or network APIs here. Use the tools exposed through `atools` for
those operations. Policy errors include a route based on metadata declared by the
currently available tools when possible. This behavioral guard catches common
local-access patterns; it is not a security sandbox. Local access remains
unsupported even when a Python escape is not intercepted.
"""

DEFAULT_PROGRAMMATIC_TOOL_CALLING_TIMEOUT_SECONDS = 300.0


class ProgrammaticToolCallingMode(StrEnum):
    UNRESTRICTED = "unrestricted"
    ORCHESTRATION_ONLY = "orchestration_only"


class ProgrammaticToolCallingErrorKind(StrEnum):
    POLICY_VIOLATION = "orchestration_policy_violation"
    NAME_ERROR = "name_error"
    MODULE_NOT_FOUND = "module_not_found"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_ERROR = "tool_error"
    PYTHON_ERROR = "python_error"


class ProgrammaticToolCallingAction(Action):
    """Schema for executing Python code in the programmatic tool environment."""

    code: str = Field(
        description=(
            "Python code to execute. State is preserved between calls, so "
            "variables and helper functions from previous executions remain "
            "available. OpenHands tools are exposed as Python callables in this "
            "namespace."
        )
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Python: ", style="bold")
        content.append(self.code)
        return content


class ProgrammaticToolCallingObservation(Observation):
    """Observation from a programmatic tool-calling execution."""

    execution_count: int = Field(
        description="Number of Python executions run by this tool instance."
    )
    error_kind: ProgrammaticToolCallingErrorKind | None = Field(
        default=None,
        description="Machine-readable category for an execution error.",
    )
    policy_violation: bool = Field(
        default=False,
        description="Whether orchestration-only policy rejected the operation.",
    )
    missing_symbol: str | None = Field(
        default=None,
        description="Missing Python symbol, module, or OpenHands tool name.",
    )
    suggested_routes: tuple[str, ...] = Field(
        default=(),
        description="Task-environment tool invocations suggested to the agent.",
    )
    failed_tool_names: tuple[str, ...] = Field(
        default=(),
        description="Nested OpenHands tools that returned or raised errors.",
    )


class ProgrammaticToolCallingTool(
    ToolDefinition[
        ProgrammaticToolCallingAction,
        ProgrammaticToolCallingObservation,
    ]
):
    """ToolDefinition for persistent Python-based tool orchestration."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        mode: ProgrammaticToolCallingMode | str = (
            ProgrammaticToolCallingMode.UNRESTRICTED
        ),
        execution_timeout_seconds: float = (
            DEFAULT_PROGRAMMATIC_TOOL_CALLING_TIMEOUT_SECONDS
        ),
    ) -> Sequence["ProgrammaticToolCallingTool"]:
        _ = conv_state
        from openhands.tools.programmatic_tool_calling.impl import (
            ProgrammaticToolCallingExecutor,
        )

        execution_mode = ProgrammaticToolCallingMode(mode)
        executor = ProgrammaticToolCallingExecutor(
            tool_name=cls.name,
            mode=execution_mode,
            execution_timeout_seconds=execution_timeout_seconds,
        )
        description = TOOL_DESCRIPTION
        if execution_mode is ProgrammaticToolCallingMode.ORCHESTRATION_ONLY:
            description += ORCHESTRATION_ONLY_DESCRIPTION
        description += (
            "\nEach program must finish within "
            f"{executor.execution_timeout_seconds:g} seconds. "
            "Keep loops bounded with an explicit iteration limit or terminating "
            "condition.\n"
        )

        return [
            cls(
                description=description,
                action_type=ProgrammaticToolCallingAction,
                observation_type=ProgrammaticToolCallingObservation,
                annotations=ToolAnnotations(
                    title="programmatic_tool_calling",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


register_tool(ProgrammaticToolCallingTool.name, ProgrammaticToolCallingTool)
