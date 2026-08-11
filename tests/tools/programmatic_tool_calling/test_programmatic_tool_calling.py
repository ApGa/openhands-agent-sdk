from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from collections.abc import Sequence
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import Field

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    Tool,
    ToolDefinition,
    ToolExecutor,
    resolve_tool,
)
from openhands.tools.programmatic_tool_calling import (
    DEFAULT_PROGRAMMATIC_TOOL_CALLING_MAX_TOOL_CALLS,
    ProgrammaticToolCallingAction,
    ProgrammaticToolCallingErrorKind,
    ProgrammaticToolCallingExecutor,
    ProgrammaticToolCallingMode,
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


class CallTool(EchoTool):
    name = "call_tool"


class GetToolDetailsTool(EchoTool):
    name = "get_tool_details"


class ViewFileTool(EchoTool):
    name = "view_file"


class CreateFileTool(EchoTool):
    name = "create_file"


class FailingEchoExecutor(ToolExecutor[EchoAction, EchoObservation]):
    def __call__(self, action: EchoAction, conversation=None) -> EchoObservation:
        return EchoObservation.from_text(
            f"failed: {action.text}",
            is_error=True,
            echoed="",
        )


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


@pytest.fixture
def orchestration_executor() -> ProgrammaticToolCallingExecutor:
    return ProgrammaticToolCallingExecutor(
        tool_name=ProgrammaticToolCallingTool.name,
        mode=ProgrammaticToolCallingMode.ORCHESTRATION_ONLY,
    )


@pytest.fixture
def catalog_conversation(conversation):
    call_tool = CallTool(
        description="Invoke a tool from the task catalog.",
        action_type=EchoAction,
        observation_type=EchoObservation,
        executor=EchoExecutor(),
        meta={
            "openhands.dev/tool-routing": {
                "version": 1,
                "execution_domain": "task",
                "capabilities": ["tool.dispatch"],
                "invocation": {
                    "kind": "dispatcher",
                    "name_argument": "name",
                    "arguments_argument": "arguments",
                    "discovery_tool": "get_tool_details",
                    "targets": [
                        {
                            "name": "workspace_operations",
                            "capabilities": [
                                "python.execute",
                                "filesystem.read",
                                "filesystem.write",
                                "system.execute",
                            ],
                        }
                    ],
                },
            }
        },
    )
    get_tool_details = GetToolDetailsTool(
        description="Inspect a catalog tool.",
        action_type=EchoAction,
        observation_type=EchoObservation,
        executor=EchoExecutor(),
    )
    tools_map = {
        **conversation.agent.tools_map,
        call_tool.name: call_tool,
        get_tool_details.name: get_tool_details,
    }
    return SimpleNamespace(agent=SimpleNamespace(tools_map=tools_map))


@pytest.fixture
def ranked_routing_conversation(catalog_conversation):
    def direct_tool(tool_type, capability: str):
        return tool_type(
            description=f"Handle {capability}.",
            action_type=EchoAction,
            observation_type=EchoObservation,
            executor=EchoExecutor(),
            meta={
                "openhands.dev/tool-routing": {
                    "version": 1,
                    "execution_domain": "task",
                    "capabilities": [capability],
                    "invocation": {"kind": "direct"},
                }
            },
        )

    view_file = direct_tool(ViewFileTool, "filesystem.read")
    create_file = direct_tool(CreateFileTool, "filesystem.write")
    tools_map = {
        **catalog_conversation.agent.tools_map,
        view_file.name: view_file,
        create_file.name: create_file,
    }
    return SimpleNamespace(agent=SimpleNamespace(tools_map=tools_map))


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
    assert "Tool calls:" not in obs.text
    assert echo_executor.calls == [EchoAction(text="ha", repeat=2)]


def test_programmatic_tool_calling_keeps_unprinted_tool_results_quiet(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
    echo_executor: EchoExecutor,
) -> None:
    obs = run_code(
        executor,
        conversation,
        'result = echo(text="quiet")',
    )

    assert obs.is_error is False
    assert obs.text == "Executed successfully with no output."
    assert echo_executor.calls == [EchoAction(text="quiet")]


def test_programmatic_tool_calling_includes_printed_tool_results(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
    echo_executor: EchoExecutor,
) -> None:
    obs = run_code(
        executor,
        conversation,
        'result = echo(text="printed")\nprint(result.text)',
    )

    assert obs.is_error is False
    assert obs.text == "printed"
    assert echo_executor.calls == [EchoAction(text="printed")]


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


@pytest.mark.parametrize("async_call", [False, True])
def test_default_budget_does_not_time_out_long_nested_tool_call(
    conversation,
    concurrent_sleep_executor: SleepExecutor,
    async_call: bool,
) -> None:
    executor = ProgrammaticToolCallingExecutor(
        tool_name=ProgrammaticToolCallingTool.name,
        max_tool_calls_per_execution=1,
    )
    call = "await atools.concurrent_sleep" if async_call else "tools.concurrent_sleep"
    try:
        started = time.monotonic()
        obs = run_code(
            executor,
            conversation,
            (f"result = {call}(label='delegated-subtree', delay=0.15)\nresult.label"),
        )

        assert time.monotonic() - started >= 0.1
        assert obs.is_error is False
        assert "delegated-subtree" in obs.text
        assert concurrent_sleep_executor.calls == ["delegated-subtree"]
        assert obs.tool_call_limit == 1
        assert obs.tool_call_attempts == 1
        assert obs.tool_calls_admitted == 1
        assert obs.tool_calls_completed == 1
        assert obs.tool_calls_rejected == 0
        assert obs.tool_call_limit_reached is False
        assert "nested-call telemetry" not in obs.text
        assert obs.model_dump()["tool_call_attempts"] == 1
        llm_text = "".join(
            getattr(content, "text", "") for content in obs.to_llm_content
        )
        assert llm_text == obs.text
        assert "tool_call_attempts" not in llm_text
    finally:
        executor.close()


def test_tool_call_budget_stops_sequential_runaway_and_resets_next_cell(
    conversation,
    echo_executor: EchoExecutor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = ProgrammaticToolCallingExecutor(
        tool_name=ProgrammaticToolCallingTool.name,
        max_tool_calls_per_execution=3,
    )
    try:
        caplog.set_level(
            logging.WARNING,
            logger="openhands.tools.programmatic_tool_calling.impl",
        )
        obs = run_code(
            executor,
            conversation,
            (
                "while True:\n"
                "    try:\n"
                "        echo(text='retry')\n"
                "    except Exception:\n"
                "        pass"
            ),
        )

        assert obs.is_error is True
        assert obs.error_kind is ProgrammaticToolCallingErrorKind.PYTHON_ERROR
        assert "per-cell budget of 3 direct OpenHands tool-call attempts" in obs.text
        assert "Use bounded loops" in obs.text
        assert obs.failed_tool_names == ("echo",)
        assert obs.tool_call_limit == 3
        assert obs.tool_call_attempts == 4
        assert obs.tool_calls_admitted == 3
        assert obs.tool_calls_completed == 3
        assert obs.tool_calls_rejected == 1
        assert obs.tool_call_limit_reached is True
        assert len(echo_executor.calls) == 3
        cap_record = next(
            record
            for record in caplog.records
            if getattr(record, "ptc_tool_call_limit_reached", False)
        )
        assert cap_record.ptc_tool_call_limit == 3
        assert cap_record.ptc_tool_call_attempts == 4
        assert cap_record.ptc_tool_calls_admitted == 3
        assert cap_record.ptc_tool_calls_completed == 3
        assert cap_record.ptc_tool_calls_rejected == 1

        follow_up = run_code(executor, conversation, "echo(text='next-cell').text")
        assert follow_up.is_error is False
        assert "next-cell" in follow_up.text
        assert follow_up.tool_call_attempts == 1
        assert follow_up.tool_calls_admitted == 1
        assert follow_up.tool_calls_completed == 1
        assert follow_up.tool_calls_rejected == 0
        assert follow_up.tool_call_limit_reached is False
        assert len(echo_executor.calls) == 4
    finally:
        executor.close()


def test_tool_call_budget_handles_malformed_tool_name_without_secondary_error(
    conversation,
) -> None:
    executor = ProgrammaticToolCallingExecutor(
        tool_name=ProgrammaticToolCallingTool.name,
        max_tool_calls_per_execution=1,
    )
    try:
        obs = run_code(
            executor,
            conversation,
            (
                "while True:\n"
                "    try:\n"
                "        call_tool({'name': 'echo'})\n"
                "    except Exception:\n"
                "        pass"
            ),
        )

        assert obs.is_error is True
        assert obs.error_kind is ProgrammaticToolCallingErrorKind.PYTHON_ERROR
        assert "per-cell budget of 1 direct OpenHands tool-call attempt" in obs.text
        assert "unhashable type" not in obs.text
        assert obs.failed_tool_names == ("<invalid dict tool name>",)
        assert obs.tool_call_attempts == 2
        assert obs.tool_calls_admitted == 1
        assert obs.tool_calls_completed == 1
        assert obs.tool_calls_rejected == 1
        assert obs.tool_call_limit_reached is True
    finally:
        executor.close()


@pytest.mark.parametrize("async_call", [False, True])
def test_missing_tool_runaway_is_covered_by_tool_call_budget(
    conversation,
    async_call: bool,
) -> None:
    executor = ProgrammaticToolCallingExecutor(
        tool_name=ProgrammaticToolCallingTool.name,
        max_tool_calls_per_execution=2,
    )
    call = 'await atools["missing_tool"]()' if async_call else 'tools["missing_tool"]()'
    try:
        obs = run_code(
            executor,
            conversation,
            (
                "while True:\n"
                "    try:\n"
                f"        {call}\n"
                "    except Exception:\n"
                "        pass"
            ),
        )

        assert obs.is_error is True
        assert obs.error_kind is ProgrammaticToolCallingErrorKind.PYTHON_ERROR
        assert "ToolNotFoundError" in obs.text
        assert "per-cell budget of 2 direct OpenHands tool-call attempts" in obs.text
        assert obs.failed_tool_names == ("missing_tool",)
        assert obs.tool_call_attempts == 3
        assert obs.tool_calls_admitted == 2
        assert obs.tool_calls_completed == 2
        assert obs.tool_calls_rejected == 1
        assert obs.tool_call_limit_reached is True
    finally:
        executor.close()


def test_concurrent_tool_call_budget_drains_admitted_calls_before_return(
    conversation,
    concurrent_sleep_executor: SleepExecutor,
) -> None:
    executor = ProgrammaticToolCallingExecutor(
        tool_name=ProgrammaticToolCallingTool.name,
        max_tool_calls_per_execution=2,
    )
    try:
        started = time.monotonic()
        obs = run_code(
            executor,
            conversation,
            (
                "await asyncio.gather(*(\n"
                "    atools.concurrent_sleep(label=f'call-{i}', delay=0.15)\n"
                "    for i in range(3)\n"
                "))"
            ),
        )
        elapsed = time.monotonic() - started

        assert obs.is_error is True
        assert obs.error_kind is ProgrammaticToolCallingErrorKind.PYTHON_ERROR
        assert "per-cell budget of 2 direct OpenHands tool-call attempts" in obs.text
        assert obs.failed_tool_names == ("concurrent_sleep",)
        assert obs.tool_call_limit == 2
        assert obs.tool_call_attempts == 3
        assert obs.tool_calls_admitted == 2
        assert obs.tool_calls_completed == 2
        assert obs.tool_calls_rejected == 1
        assert obs.tool_call_limit_reached is True
        assert elapsed >= 0.1
        assert concurrent_sleep_executor.active == 0
        assert len(concurrent_sleep_executor.calls) == 2

        calls_at_return = list(concurrent_sleep_executor.calls)
        time.sleep(0.1)
        assert concurrent_sleep_executor.calls == calls_at_return

        follow_up = run_code(executor, conversation, '"clean-next-cell"')
        assert follow_up.is_error is False
        assert "clean-next-cell" in follow_up.text
        assert follow_up.tool_call_attempts == 0
        assert follow_up.tool_calls_admitted == 0
        assert follow_up.tool_calls_completed == 0
        assert follow_up.tool_calls_rejected == 0
        assert follow_up.tool_call_limit_reached is False
    finally:
        executor.close()


def test_cancelled_async_wrapper_drains_admitted_worker_before_cell_returns(
    conversation,
    concurrent_sleep_executor: SleepExecutor,
) -> None:
    executor = ProgrammaticToolCallingExecutor(
        tool_name=ProgrammaticToolCallingTool.name,
        max_tool_calls_per_execution=1,
    )
    try:
        started = time.monotonic()
        obs = run_code(
            executor,
            conversation,
            (
                "async def run_tool():\n"
                "    return await atools.concurrent_sleep(\n"
                "        label='cancelled-wrapper', delay=0.15\n"
                "    )\n"
                "task = asyncio.create_task(run_tool())\n"
                "await asyncio.sleep(0.02)\n"
                "task.cancel()\n"
                "try:\n"
                "    await task\n"
                "except asyncio.CancelledError:\n"
                "    pass\n"
                '"wrapper-cancelled"'
            ),
        )

        assert time.monotonic() - started >= 0.1
        assert obs.is_error is False
        assert "wrapper-cancelled" in obs.text
        assert concurrent_sleep_executor.active == 0
        assert concurrent_sleep_executor.calls == ["cancelled-wrapper"]
        assert obs.tool_call_attempts == 1
        assert obs.tool_calls_admitted == 1
        assert obs.tool_calls_completed == 1
        assert obs.tool_calls_rejected == 0

        calls_at_return = list(concurrent_sleep_executor.calls)
        time.sleep(0.1)
        assert concurrent_sleep_executor.calls == calls_at_return
    finally:
        executor.close()


def test_programmatic_tool_calling_cancels_tasks_abandoned_by_asyncio_run(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
    echo_executor: EchoExecutor,
) -> None:
    obs = run_code(
        executor,
        conversation,
        (
            "async def delayed_echo():\n"
            "    await asyncio.sleep(0.05)\n"
            '    echo(text="leaked")\n'
            "asyncio.run(asyncio.gather(delayed_echo()))"
        ),
    )

    assert obs.is_error is True
    assert obs.error_kind is ProgrammaticToolCallingErrorKind.PYTHON_ERROR
    assert "asyncio.run() cannot be called from a running event loop" in obs.text
    assert "Canceled 1 background asyncio task" in obs.text
    assert "Use top-level `await`" in obs.text

    follow_up = run_code(
        executor,
        conversation,
        'await asyncio.sleep(0.1)\n"next-cell"',
    )

    assert follow_up.is_error is False
    assert "next-cell" in follow_up.text
    assert echo_executor.calls == []


def test_programmatic_tool_calling_marks_unawaited_tasks_as_errors(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    obs = run_code(
        executor,
        conversation,
        (
            "async def background():\n"
            "    await asyncio.sleep(60)\n"
            "asyncio.create_task(background())\n"
            '"scheduled"'
        ),
    )

    assert obs.is_error is True
    assert obs.error_kind is ProgrammaticToolCallingErrorKind.PYTHON_ERROR
    assert "Canceled 1 background asyncio task" in obs.text
    assert "await asyncio.gather(...)" in obs.text
    assert not executor._loop.is_closed()
    assert not executor._loop.run_until_complete(_pending_tasks())


async def _pending_tasks() -> set:
    current = asyncio.current_task()
    return {task for task in asyncio.all_tasks() if task is not current}


def test_programmatic_tool_calling_consumes_abandoned_gather_errors(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    loop_errors: list[dict] = []
    executor._loop.set_exception_handler(
        lambda _loop, context: loop_errors.append(context)
    )

    obs = run_code(
        executor,
        conversation,
        (
            "async def missing_tool():\n"
            '    tools["does_not_exist"]()\n'
            "asyncio.run(asyncio.gather(missing_tool()))"
        ),
    )

    assert obs.is_error is True
    assert "asyncio.run() cannot be called from a running event loop" in obs.text
    assert "ToolNotFoundError" in obs.text
    assert obs.failed_tool_names == ("does_not_exist",)

    run_code(executor, conversation, '"release-previous-traceback"')
    gc.collect()

    assert not any(
        "exception was never retrieved" in context.get("message", "").lower()
        for context in loop_errors
    )


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


def test_bare_call_tool_rejects_request_mapping_with_dispatcher_guidance(
    executor: ProgrammaticToolCallingExecutor,
    catalog_conversation,
) -> None:
    obs = run_code(
        executor,
        catalog_conversation,
        'await call_tool({"name": "catalog.echo", "arguments": {}})',
    )

    assert obs.is_error is True
    assert obs.error_kind is ProgrammaticToolCallingErrorKind.PYTHON_ERROR
    assert obs.failed_tool_names == ("<invalid dict tool name>",)
    assert "An OpenHands tool name must be a string; received dict" in obs.text
    assert "`tools.call_tool({...})`" in obs.text
    assert "`await atools.call_tool({...})`" in obs.text
    assert "unhashable type" not in obs.text


def test_bare_call_tool_uses_generic_guidance_without_dispatcher(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    obs = run_code(
        executor,
        conversation,
        'call_tool({"name": "echo", "arguments": {}})',
    )

    assert obs.is_error is True
    assert obs.failed_tool_names == ("<invalid dict tool name>",)
    assert 'call_tool("<tool_name>", **arguments)' in obs.text
    nested_feedback = obs.text.split("Nested tool call errors:\n", 1)[1]
    assert "`tools.call_tool({...})`" not in nested_feedback


@pytest.mark.parametrize(
    "code",
    [
        'open("/workspace/example.txt")',
        "import os\nos.listdir('.')",
        "import builtins as b\nlocal_open = b.open",
        "import builtins as b\ngetattr(b, 'open')('/workspace/example.txt')",
        'import io as stream_io\nstream_io.open("/workspace/example.txt")',
        "import sys\nsys.modules['builtins'].open('/workspace/example.txt')",
        "!pwd",
    ],
)
def test_orchestration_only_blocks_local_environment_operations(
    orchestration_executor: ProgrammaticToolCallingExecutor,
    conversation,
    code: str,
) -> None:
    obs = run_code(orchestration_executor, conversation, code)

    assert obs.is_error is True
    assert obs.error_kind is ProgrammaticToolCallingErrorKind.POLICY_VIOLATION
    assert obs.policy_violation is True
    assert "outside the task environment" in obs.text
    assert "Currently available tools" in obs.text


def test_orchestration_only_runtime_guard_blocks_saved_open_reference(
    orchestration_executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    saved = run_code(orchestration_executor, conversation, "local_open = open")
    obs = run_code(
        orchestration_executor,
        conversation,
        'local_open("/workspace/example.txt")',
    )

    assert saved.is_error is False
    assert obs.is_error is True
    assert obs.policy_violation is True


def test_orchestration_only_uses_declared_catalog_route(
    orchestration_executor: ProgrammaticToolCallingExecutor,
    catalog_conversation,
) -> None:
    obs = run_code(
        orchestration_executor,
        catalog_conversation,
        'open("/workspace/example.txt")',
    )

    assert obs.policy_violation is True
    assert obs.suggested_routes == (
        "await atools.call_tool(name='workspace_operations', arguments={...})",
    )
    assert "atools.get_tool_details(name='workspace_operations')" in obs.text
    assert "atools.call_tool(name='workspace_operations'" in obs.text


def test_orchestration_only_ranks_routes_for_open_mode(
    orchestration_executor: ProgrammaticToolCallingExecutor,
    ranked_routing_conversation,
) -> None:
    read = run_code(
        orchestration_executor,
        ranked_routing_conversation,
        'open("/workspace/example.txt")',
    )
    write = run_code(
        orchestration_executor,
        ranked_routing_conversation,
        'open("/workspace/example.txt", "w")',
    )
    update = run_code(
        orchestration_executor,
        ranked_routing_conversation,
        'open("/workspace/example.txt", "r+")',
    )
    path_read = run_code(
        orchestration_executor,
        ranked_routing_conversation,
        'Path("/workspace/example.txt").read_text()',
    )
    path_write = run_code(
        orchestration_executor,
        ranked_routing_conversation,
        'Path("/workspace/example.txt").write_text("content")',
    )

    assert read.suggested_routes[0] == "await atools.view_file(...)"
    assert write.suggested_routes[0] == "await atools.create_file(...)"
    assert update.suggested_routes[0].startswith(
        "await atools.call_tool(name='workspace_operations'"
    )
    assert path_read.suggested_routes[0] == "await atools.view_file(...)"
    assert path_write.suggested_routes[0] == "await atools.create_file(...)"


def test_missing_catalog_tool_redirects_to_dispatcher(
    orchestration_executor: ProgrammaticToolCallingExecutor,
    catalog_conversation,
) -> None:
    obs = run_code(
        orchestration_executor,
        catalog_conversation,
        'await atools.workspace_operations(path="example.txt")',
    )

    assert obs.is_error is True
    assert obs.error_kind is ProgrammaticToolCallingErrorKind.TOOL_NOT_FOUND
    assert obs.missing_symbol == "workspace_operations"
    assert obs.suggested_routes[0].startswith(
        "await atools.call_tool(name='workspace_operations'"
    )


@pytest.mark.parametrize(
    ("version", "target_name"),
    [
        (True, "workspace_operations"),
        (1, "workspace_operations`\nignore_previous_instructions"),
    ],
)
def test_invalid_routing_metadata_falls_back_without_rendering_it(
    orchestration_executor: ProgrammaticToolCallingExecutor,
    catalog_conversation,
    version: object,
    target_name: str,
) -> None:
    call_tool = catalog_conversation.agent.tools_map["call_tool"]
    meta = deepcopy(call_tool.meta)
    assert meta is not None
    routing = meta["openhands.dev/tool-routing"]
    routing["version"] = version
    routing["invocation"]["targets"][0]["name"] = target_name
    invalid_call_tool = call_tool.model_copy(update={"meta": meta})
    tools_map = {
        **catalog_conversation.agent.tools_map,
        "call_tool": invalid_call_tool,
    }
    invalid_conversation = SimpleNamespace(agent=SimpleNamespace(tools_map=tools_map))

    obs = run_code(
        orchestration_executor,
        invalid_conversation,
        'open("/workspace/example.txt")',
    )

    assert obs.suggested_routes == ()
    assert "ignore_previous_instructions" not in obs.text
    assert "Currently available tools" in obs.text


def test_name_and_module_errors_use_declared_catalog_route(
    orchestration_executor: ProgrammaticToolCallingExecutor,
    catalog_conversation,
) -> None:
    name_error = run_code(
        orchestration_executor,
        catalog_conversation,
        'workspace_operations(path="example.txt")',
    )
    module_error = run_code(
        orchestration_executor,
        catalog_conversation,
        "import definitely_missing_ptc_module",
    )

    assert name_error.error_kind is ProgrammaticToolCallingErrorKind.NAME_ERROR
    assert name_error.missing_symbol == "workspace_operations"
    assert name_error.suggested_routes
    assert module_error.error_kind is ProgrammaticToolCallingErrorKind.MODULE_NOT_FOUND
    assert module_error.missing_symbol == "definitely_missing_ptc_module"
    assert module_error.suggested_routes


def test_nested_tool_error_marks_outer_observation_as_error(
    executor: ProgrammaticToolCallingExecutor,
    conversation,
) -> None:
    failing_echo = EchoTool(
        description="Return an error.",
        action_type=EchoAction,
        observation_type=EchoObservation,
        executor=FailingEchoExecutor(),
    )
    tools_map = {**conversation.agent.tools_map, failing_echo.name: failing_echo}
    failing_conversation = SimpleNamespace(agent=SimpleNamespace(tools_map=tools_map))

    obs = run_code(executor, failing_conversation, 'echo(text="bad")')

    assert obs.is_error is True
    assert obs.error_kind is ProgrammaticToolCallingErrorKind.TOOL_ERROR
    assert obs.failed_tool_names == ("echo",)
    assert "Nested tool call errors:\necho: failed: bad" in obs.text


def test_caught_nested_exceptions_still_mark_outer_observation_as_error(
    executor: ProgrammaticToolCallingExecutor,
    catalog_conversation,
) -> None:
    obs = run_code(
        executor,
        catalog_conversation,
        (
            "try:\n"
            '    await atools.workspace_operations(path="example.txt")\n'
            "except LookupError:\n"
            "    pass\n"
            "try:\n"
            '    echo(unexpected="argument")\n'
            "except Exception:\n"
            "    pass\n"
            '"continued"'
        ),
    )

    assert obs.is_error is True
    assert obs.error_kind is ProgrammaticToolCallingErrorKind.TOOL_ERROR
    assert obs.failed_tool_names == ("workspace_operations", "echo")
    assert "atools.call_tool(name='workspace_operations'" in obs.text
    assert "validation error" in obs.text


def test_tool_factory_configures_orchestration_only_mode() -> None:
    tool = resolve_tool(
        Tool(
            name=ProgrammaticToolCallingTool.name,
            params={
                "mode": "orchestration_only",
                "max_tool_calls_per_execution": 17,
            },
        ),
        cast("ConversationState", SimpleNamespace()),
    )[0]

    assert isinstance(tool.executor, ProgrammaticToolCallingExecutor)
    assert tool.executor.mode is ProgrammaticToolCallingMode.ORCHESTRATION_ONLY
    assert tool.executor.max_tool_calls_per_execution == 17
    assert "outside the task environment" in tool.description
    assert "at most 17 direct OpenHands tool calls" in tool.description


def test_programmatic_tool_calling_default_tool_call_budget(
    executor: ProgrammaticToolCallingExecutor,
) -> None:
    assert (
        executor.max_tool_calls_per_execution
        == DEFAULT_PROGRAMMATIC_TOOL_CALLING_MAX_TOOL_CALLS
        == 1024
    )


@pytest.mark.parametrize(
    "max_tool_calls_per_execution",
    [0, -1, 1.5, True, "5", None],
)
def test_programmatic_tool_calling_rejects_invalid_tool_call_budget(
    max_tool_calls_per_execution,
) -> None:
    with pytest.raises(
        ValueError,
        match="max_tool_calls_per_execution must be a positive integer",
    ):
        ProgrammaticToolCallingExecutor(
            tool_name=ProgrammaticToolCallingTool.name,
            max_tool_calls_per_execution=max_tool_calls_per_execution,
        )


def test_default_preset_keeps_programmatic_tool_calling_opt_in() -> None:
    from openhands.tools.preset.default import get_default_tools

    default_tools = get_default_tools(enable_browser=False)
    opt_in_tools = get_default_tools(
        enable_browser=False,
        enable_programmatic_tool_calling=True,
    )

    assert ProgrammaticToolCallingTool.name not in {t.name for t in default_tools}
    assert ProgrammaticToolCallingTool.name in {t.name for t in opt_in_tools}
