from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from pydantic import Field

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolDefinition,
    ToolExecutor,
)
from openhands.tools.programmatic_tool_calling import (
    ProgrammaticToolCallingAction,
    ProgrammaticToolCallingExecutor,
    ProgrammaticToolCallingObservation,
    ProgrammaticToolCallingTool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


class EchoAction(Action):
    text: str
    repeat: int = Field(default=1)


class EchoObservation(Observation):
    echoed: str


class EchoExecutor(ToolExecutor[EchoAction, EchoObservation]):
    def __init__(self) -> None:
        self.calls: list[EchoAction] = []

    def __call__(self, action: EchoAction, conversation=None) -> EchoObservation:
        self.calls.append(action)
        text = action.text * action.repeat
        return EchoObservation.from_text(text, echoed=text)


class EchoTool(ToolDefinition[EchoAction, EchoObservation]):
    name = "echo"

    @classmethod
    def create(cls, conv_state: ConversationState) -> Sequence[EchoTool]:
        return []


class SleepAction(Action):
    label: str
    delay: float = Field(default=0.2, ge=0)


class SleepObservation(Observation):
    label: str


class SleepExecutor(ToolExecutor[SleepAction, SleepObservation]):
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, action: SleepAction, conversation=None) -> SleepObservation:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(action.label)
        try:
            time.sleep(action.delay)
            return SleepObservation.from_text(action.label, label=action.label)
        finally:
            with self._lock:
                self.active -= 1


class ConcurrentSleepTool(ToolDefinition[SleepAction, SleepObservation]):
    name = "concurrent_sleep"

    @classmethod
    def create(cls, conv_state: ConversationState) -> Sequence[ConcurrentSleepTool]:
        return []

    def declared_resources(self, action: Action) -> DeclaredResources:  # noqa: ARG002
        return DeclaredResources(keys=(), declared=True)


class LockedSleepTool(ToolDefinition[SleepAction, SleepObservation]):
    name = "locked_sleep"

    @classmethod
    def create(cls, conv_state: ConversationState) -> Sequence[LockedSleepTool]:
        return []


@pytest.fixture
def echo_executor() -> EchoExecutor:
    return EchoExecutor()


@pytest.fixture
def concurrent_sleep_executor() -> SleepExecutor:
    return SleepExecutor()


@pytest.fixture
def locked_sleep_executor() -> SleepExecutor:
    return SleepExecutor()


@pytest.fixture
def conversation(
    echo_executor: EchoExecutor,
    concurrent_sleep_executor: SleepExecutor,
    locked_sleep_executor: SleepExecutor,
):
    echo_tool = EchoTool(
        description="Echo text.",
        action_type=EchoAction,
        observation_type=EchoObservation,
        executor=echo_executor,
    )
    concurrent_sleep_tool = ConcurrentSleepTool(
        description="Sleep without shared resources.",
        action_type=SleepAction,
        observation_type=SleepObservation,
        executor=concurrent_sleep_executor,
    )
    locked_sleep_tool = LockedSleepTool(
        description="Sleep with default tool-level locking.",
        action_type=SleepAction,
        observation_type=SleepObservation,
        executor=locked_sleep_executor,
    )
    programmatic_tool = ProgrammaticToolCallingTool(
        description="Run persistent Python code that can call other tools.",
        action_type=ProgrammaticToolCallingAction,
        observation_type=ProgrammaticToolCallingObservation,
        executor=ProgrammaticToolCallingExecutor(
            tool_name=ProgrammaticToolCallingTool.name
        ),
    )
    agent = SimpleNamespace(
        tools_map={
            echo_tool.name: echo_tool,
            concurrent_sleep_tool.name: concurrent_sleep_tool,
            locked_sleep_tool.name: locked_sleep_tool,
            programmatic_tool.name: programmatic_tool,
        }
    )
    return SimpleNamespace(agent=agent)


@pytest.fixture
def executor() -> ProgrammaticToolCallingExecutor:
    return ProgrammaticToolCallingExecutor(tool_name=ProgrammaticToolCallingTool.name)


def run_code(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
    code: str,
):
    return executor(ProgrammaticToolCallingAction(code=code), conversation)


def test_programmatic_tool_calling_preserves_python_state(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    first = run_code(executor, conversation, "counter = 40\ncounter")
    second = run_code(executor, conversation, "counter += 2\ncounter")

    assert first.is_error is False
    assert "40" in first.text
    assert second.is_error is False
    assert "42" in second.text
    assert second.execution_count == 2


def test_programmatic_tool_calling_exposes_tools_as_python_functions(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
    echo_executor: EchoExecutor,
) -> None:
    obs = run_code(
        executor,
        conversation,
        'result = echo(text="ha", repeat=2)\nresult.echoed',
    )

    assert obs.is_error is False
    assert "haha" in obs.text
    assert "Tool calls:" in obs.text
    assert echo_executor.calls == [EchoAction(text="ha", repeat=2)]


def test_programmatic_tool_calling_exposes_tools_namespace(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    obs = run_code(
        executor,
        conversation,
        'result = tools.echo(text="ok")\n(tools.available(), result.text)',
    )

    assert obs.is_error is False
    assert "echo" in obs.text
    assert ProgrammaticToolCallingTool.name not in obs.text
    assert "ok" in obs.text


def test_programmatic_tool_calling_supports_top_level_await(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    obs = run_code(
        executor,
        conversation,
        'await asyncio.sleep(0)\n"async-ok"',
    )

    assert obs.is_error is False
    assert "async-ok" in obs.text


def test_programmatic_tool_calling_exposes_async_tools(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
    echo_executor: EchoExecutor,
) -> None:
    obs = run_code(
        executor,
        conversation,
        'result = await atools.echo(text="async-ok")\nresult.echoed',
    )

    assert obs.is_error is False
    assert "async-ok" in obs.text
    assert echo_executor.calls == [EchoAction(text="async-ok")]


def test_programmatic_tool_calling_runs_async_tools_concurrently(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
    concurrent_sleep_executor: SleepExecutor,
) -> None:
    obs = run_code(
        executor,
        conversation,
        (
            "first, second = await asyncio.gather(\n"
            '    atools.concurrent_sleep(label="first"),\n'
            '    atools.concurrent_sleep(label="second"),\n'
            ")\n"
            "(first.label, second.label)"
        ),
    )

    assert obs.is_error is False
    assert "first" in obs.text
    assert "second" in obs.text
    assert concurrent_sleep_executor.max_active == 2


def test_programmatic_tool_calling_serializes_undeclared_resource_tools(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
    locked_sleep_executor: SleepExecutor,
) -> None:
    obs = run_code(
        executor,
        conversation,
        (
            "first, second = await asyncio.gather(\n"
            '    atools.locked_sleep(label="first"),\n'
            '    atools.locked_sleep(label="second"),\n'
            ")\n"
            "(first.label, second.label)"
        ),
    )

    assert obs.is_error is False
    assert "first" in obs.text
    assert "second" in obs.text
    assert locked_sleep_executor.max_active == 1


def test_programmatic_tool_calling_rejects_recursive_self_call(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    obs = run_code(
        executor,
        conversation,
        f'tools["{ProgrammaticToolCallingTool.name}"](code="1 + 1")',
    )

    assert obs.is_error is True
    assert "cannot call itself" in obs.text


def test_programmatic_tool_calling_rejects_positional_tool_args(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    obs = run_code(executor, conversation, 'echo("bad")')

    assert obs.is_error is True
    assert "keyword arguments" in obs.text


def test_default_preset_keeps_programmatic_tool_calling_opt_in() -> None:
    from openhands.tools.preset.default import get_default_tools

    default_tools = get_default_tools(enable_browser=False)
    opt_in_tools = get_default_tools(
        enable_browser=False,
        enable_programmatic_tool_calling=True,
    )

    assert ProgrammaticToolCallingTool.name not in {t.name for t in default_tools}
    assert ProgrammaticToolCallingTool.name in {t.name for t in opt_in_tools}
