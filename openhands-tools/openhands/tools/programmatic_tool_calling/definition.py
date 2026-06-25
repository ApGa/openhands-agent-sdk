"""Programmatic tool calling through a persistent embedded IPython shell."""

from collections.abc import Sequence
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
    ) -> Sequence["ProgrammaticToolCallingTool"]:
        _ = conv_state
        from openhands.tools.programmatic_tool_calling.impl import (
            ProgrammaticToolCallingExecutor,
        )

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=ProgrammaticToolCallingAction,
                observation_type=ProgrammaticToolCallingObservation,
                annotations=ToolAnnotations(
                    title="programmatic_tool_calling",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=ProgrammaticToolCallingExecutor(tool_name=cls.name),
            )
        ]


register_tool(ProgrammaticToolCallingTool.name, ProgrammaticToolCallingTool)
