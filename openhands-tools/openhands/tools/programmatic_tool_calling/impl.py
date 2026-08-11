from __future__ import annotations

import asyncio
import io
import keyword
import logging
import re
import sys
import threading
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config.loader import Config

from openhands.sdk.conversation.resource_lock_manager import ResourceLockManager
from openhands.sdk.tool import Observation, ToolExecutor
from openhands.sdk.utils import maybe_truncate
from openhands.tools.programmatic_tool_calling.definition import (
    DEFAULT_PROGRAMMATIC_TOOL_CALLING_MAX_TOOL_CALLS,
    ProgrammaticToolCallingAction,
    ProgrammaticToolCallingErrorKind,
    ProgrammaticToolCallingMode,
    ProgrammaticToolCallingObservation,
)
from openhands.tools.programmatic_tool_calling.policy import (
    OrchestrationPolicyError,
    capabilities_for_missing_symbol,
    orchestration_only_bindings,
    validate_orchestration_code,
)
from openhands.tools.programmatic_tool_calling.routing import (
    ToolRoute,
    ToolRoutingIndex,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.tool import Action, ToolDefinition


MAX_PROGRAMMATIC_TOOL_OUTPUT_SIZE = 50_000
logger = logging.getLogger(__name__)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_IPYTHON_OUT_PROMPT_RE = re.compile(r"^Out\[\d+\]: ?")
_RESERVED_NAMES = frozenset(
    {
        "acall_tool",
        "atools",
        "call_tool",
        "tools",
    }
)


class ToolNotFoundError(LookupError):
    def __init__(self, tool_name: str, available_tool_names: Sequence[str]):
        self.tool_name = tool_name
        super().__init__(
            f"Tool '{tool_name}' not found. Available tools: "
            f"{list(available_tool_names)}"
        )


class _ToolCallBudgetExceededError(BaseException):
    def __init__(self, max_tool_calls: int, tool_name: object):
        self.max_tool_calls = max_tool_calls
        self.tool_name = tool_name
        super().__init__(
            "Programmatic tool execution exceeded its per-cell budget of "
            f"{max_tool_calls} direct OpenHands tool-call attempts."
        )


class _ToolCallAdmissionsClosedError(BaseException):
    def __init__(self, tool_name: object):
        self.tool_name = tool_name
        super().__init__(
            "This programmatic tool cell is already finishing and no longer "
            "accepts direct OpenHands tool calls."
        )


@dataclass(frozen=True, slots=True)
class _CellCallTelemetry:
    limit: int
    attempts: int
    admitted: int
    completed: int
    rejected: int
    limit_reached: bool


class _CellCallState:
    """Per-cell admissions and liveness state shared with nested-call workers."""

    __slots__ = (
        "_accepting",
        "_active",
        "_admitted",
        "_attempts",
        "_completed",
        "_condition",
        "_limit_reached",
        "_max_tool_calls",
        "_rejected",
    )

    def __init__(self, max_tool_calls: int):
        self._max_tool_calls = max_tool_calls
        self._condition = threading.Condition()
        self._accepting = True
        self._attempts = 0
        self._admitted = 0
        self._completed = 0
        self._rejected = 0
        self._active = 0
        self._limit_reached = False

    @property
    def exhausted(self) -> bool:
        with self._condition:
            return self._limit_reached

    def begin(self, tool_name: object) -> None:
        with self._condition:
            self._attempts += 1
            if not self._accepting:
                self._rejected += 1
                raise _ToolCallAdmissionsClosedError(tool_name)
            if self._admitted >= self._max_tool_calls:
                self._rejected += 1
                self._limit_reached = True
                raise _ToolCallBudgetExceededError(
                    self._max_tool_calls,
                    tool_name,
                )
            self._admitted += 1
            self._active += 1

    def finish(self) -> None:
        with self._condition:
            self._active -= 1
            self._completed += 1
            if not self._active:
                self._condition.notify_all()

    def close_admissions(self) -> None:
        with self._condition:
            self._accepting = False

    def wait_until_idle(self) -> None:
        with self._condition:
            while self._active:
                self._condition.wait()

    def telemetry(self) -> _CellCallTelemetry:
        with self._condition:
            return _CellCallTelemetry(
                limit=self._max_tool_calls,
                attempts=self._attempts,
                admitted=self._admitted,
                completed=self._completed,
                rejected=self._rejected,
                limit_reached=self._limit_reached,
            )


@dataclass(frozen=True, slots=True)
class _NestedToolError:
    tool_name: str
    text: str


@dataclass(frozen=True, slots=True)
class _ExecutionError:
    kind: ProgrammaticToolCallingErrorKind | None = None
    policy_violation: bool = False
    missing_symbol: str | None = None
    guidance: str | None = None
    suggested_routes: tuple[str, ...] = ()


class _ToolFunction:
    def __init__(self, executor: ProgrammaticToolCallingExecutor, tool_name: str):
        self._executor = executor
        self._tool_name = tool_name
        self.__name__ = tool_name
        self.__qualname__ = tool_name
        self.__doc__ = f"Call the OpenHands '{tool_name}' tool."

    def __call__(self, *args: Any, **kwargs: Any) -> Observation:
        return self._executor.call_tool(self._tool_name, args, kwargs)

    def __repr__(self) -> str:
        return f"<OpenHands tool function {self._tool_name}>"


class _AsyncToolFunction:
    def __init__(self, executor: ProgrammaticToolCallingExecutor, tool_name: str):
        self._executor = executor
        self._tool_name = tool_name
        self.__name__ = tool_name
        self.__qualname__ = tool_name
        self.__doc__ = f"Call the OpenHands '{tool_name}' tool asynchronously."

    async def __call__(self, *args: Any, **kwargs: Any) -> Observation:
        return await self._executor.acall_tool(self._tool_name, args, kwargs)

    def __repr__(self) -> str:
        return f"<OpenHands async tool function {self._tool_name}>"


class _ToolNamespace:
    def __init__(self, executor: ProgrammaticToolCallingExecutor):
        self._executor = executor

    def __getattr__(self, tool_name: str) -> _ToolFunction:
        if tool_name.startswith("_"):
            raise AttributeError(tool_name)
        return self[tool_name]

    def __getitem__(self, tool_name: str) -> _ToolFunction:
        return _ToolFunction(self._executor, tool_name)

    def __dir__(self) -> list[str]:
        return [
            name
            for name in self.available()
            if name.isidentifier() and not keyword.iskeyword(name)
        ]

    def available(self) -> list[str]:
        return self._executor.available_tool_names()

    def __repr__(self) -> str:
        names = ", ".join(self.available())
        return f"<OpenHands tools: {names}>"


class _AsyncToolNamespace:
    def __init__(self, executor: ProgrammaticToolCallingExecutor):
        self._executor = executor

    def __getattr__(self, tool_name: str) -> _AsyncToolFunction:
        if tool_name.startswith("_"):
            raise AttributeError(tool_name)
        return self[tool_name]

    def __getitem__(self, tool_name: str) -> _AsyncToolFunction:
        return _AsyncToolFunction(self._executor, tool_name)

    def __dir__(self) -> list[str]:
        return [
            name
            for name in self.available()
            if name.isidentifier() and not keyword.iskeyword(name)
        ]

    def available(self) -> list[str]:
        return self._executor.available_tool_names()

    def __repr__(self) -> str:
        names = ", ".join(self.available())
        return f"<OpenHands async tools: {names}>"


class ProgrammaticToolCallingExecutor(
    ToolExecutor[
        ProgrammaticToolCallingAction,
        ProgrammaticToolCallingObservation,
    ]
):
    """Execute Python code in a persistent IPython namespace with tool callables."""

    def __init__(
        self,
        tool_name: str,
        mode: ProgrammaticToolCallingMode | str = (
            ProgrammaticToolCallingMode.UNRESTRICTED
        ),
        max_tool_calls_per_execution: int = (
            DEFAULT_PROGRAMMATIC_TOOL_CALLING_MAX_TOOL_CALLS
        ),
    ):
        if (
            isinstance(max_tool_calls_per_execution, bool)
            or not isinstance(max_tool_calls_per_execution, int)
            or max_tool_calls_per_execution <= 0
        ):
            raise ValueError("max_tool_calls_per_execution must be a positive integer")
        self._tool_name = tool_name
        self._mode = ProgrammaticToolCallingMode(mode)
        self._max_tool_calls_per_execution = max_tool_calls_per_execution
        self._shell = self._create_shell()
        self._loop = asyncio.new_event_loop()
        self._tool_thread_pool = ThreadPoolExecutor(
            thread_name_prefix="openhands-ptc-tool"
        )
        self._lock = threading.RLock()
        self._nested_error_lock = threading.Lock()
        self._cell_call_state: _CellCallState | None = None
        self._resource_lock_manager = ResourceLockManager()
        self._conversation: LocalConversation | None = None
        self._routing_index = ToolRoutingIndex(routes=(), available_tool_names=())
        self._nested_tool_errors: list[_NestedToolError] = []
        self._execution_count = 0

    def __call__(
        self,
        action: ProgrammaticToolCallingAction,
        conversation: LocalConversation | None = None,
    ) -> ProgrammaticToolCallingObservation:
        with self._lock:
            execution_count = self._execution_count
        if conversation is None:
            return ProgrammaticToolCallingObservation.from_text(
                "programmatic_tool_calling requires a LocalConversation context.",
                is_error=True,
                execution_count=execution_count,
                error_kind=ProgrammaticToolCallingErrorKind.PYTHON_ERROR,
                tool_call_limit=self._max_tool_calls_per_execution,
            )

        code = action.code.strip()
        if not code:
            return ProgrammaticToolCallingObservation.from_text(
                "No Python code was provided.",
                is_error=True,
                execution_count=execution_count,
                error_kind=ProgrammaticToolCallingErrorKind.PYTHON_ERROR,
                tool_call_limit=self._max_tool_calls_per_execution,
            )

        with self._lock:
            orphaned_task_count = self._cancel_and_drain_pending_tasks()
            cell_call_state = _CellCallState(self._max_tool_calls_per_execution)
            self._cell_call_state = cell_call_state
            self._conversation = conversation
            self._install_tool_namespace(conversation)
            with self._nested_error_lock:
                self._nested_tool_errors.clear()
            self._execution_count += 1
            execution_count = self._execution_count

            stdout = io.StringIO()
            stderr = io.StringIO()
            execution_error = _ExecutionError()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result, cell_orphaned_task_count = self._run_cell(code)
                orphaned_task_count += cell_orphaned_task_count

                python_error = getattr(result, "error_before_exec", None) or getattr(
                    result, "error_in_exec", None
                )
                self._consume_future_exceptions_from_traceback(python_error)
                execution_error = self._classify_error(python_error)
                orphaned_task_guidance = self._orphaned_task_guidance(
                    orphaned_task_count
                )
                nested_tool_errors = self._nested_errors_snapshot()
                output = self._format_output(
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue(),
                    result=getattr(result, "result", None),
                    error=python_error,
                    guidance=self._join_guidance(
                        execution_error.guidance,
                        orphaned_task_guidance,
                    ),
                    nested_tool_errors=nested_tool_errors,
                )
                is_error = (
                    not getattr(result, "success", False)
                    or bool(nested_tool_errors)
                    or bool(orphaned_task_count)
                )
                if nested_tool_errors and execution_error.kind is None:
                    execution_error = _ExecutionError(
                        kind=(
                            ProgrammaticToolCallingErrorKind.PYTHON_ERROR
                            if cell_call_state.exhausted
                            else ProgrammaticToolCallingErrorKind.TOOL_ERROR
                        )
                    )
                if orphaned_task_count and execution_error.kind is None:
                    execution_error = _ExecutionError(
                        kind=ProgrammaticToolCallingErrorKind.PYTHON_ERROR
                    )
            except BaseException as exc:
                orphaned_task_count += self._cancel_and_drain_pending_tasks()
                execution_error = self._classify_error(exc)
                self._consume_future_exceptions_from_traceback(exc)
                output = self._format_caught_exception(
                    exc,
                    self._join_guidance(
                        execution_error.guidance,
                        self._orphaned_task_guidance(orphaned_task_count),
                    ),
                )
                nested_tool_errors = self._nested_errors_snapshot()
                is_error = True
            finally:
                # _run_cell closes admissions, cancels async wrappers, and waits
                # for every admitted worker before control can reach this point.
                self._conversation = None
                self._cell_call_state = None
                self._routing_index = ToolRoutingIndex(
                    routes=(), available_tool_names=()
                )

        telemetry = cell_call_state.telemetry()
        self._log_cell_telemetry(execution_count, telemetry)
        failed_tool_names = tuple(
            dict.fromkeys(error.tool_name for error in nested_tool_errors)
        )

        return ProgrammaticToolCallingObservation.from_text(
            maybe_truncate(output, truncate_after=MAX_PROGRAMMATIC_TOOL_OUTPUT_SIZE),
            is_error=is_error,
            execution_count=execution_count,
            error_kind=execution_error.kind,
            policy_violation=execution_error.policy_violation,
            missing_symbol=execution_error.missing_symbol,
            suggested_routes=execution_error.suggested_routes,
            failed_tool_names=failed_tool_names,
            tool_call_limit=telemetry.limit,
            tool_call_attempts=telemetry.attempts,
            tool_calls_admitted=telemetry.admitted,
            tool_calls_completed=telemetry.completed,
            tool_calls_rejected=telemetry.rejected,
            tool_call_limit_reached=telemetry.limit_reached,
        )

    @property
    def mode(self) -> ProgrammaticToolCallingMode:
        return self._mode

    @property
    def max_tool_calls_per_execution(self) -> int:
        return self._max_tool_calls_per_execution

    def close(self) -> None:
        with self._lock:
            cell_call_state = self._cell_call_state
            if cell_call_state is not None:
                cell_call_state.close_admissions()
            if not self._loop.is_closed():
                self._cancel_and_drain_pending_tasks()
                if cell_call_state is not None:
                    cell_call_state.wait_until_idle()
                    self._flush_loop_callbacks()
                self._conversation = None
                self._cell_call_state = None
                self._tool_thread_pool.shutdown(wait=True, cancel_futures=False)
                self._loop.close()

    def call_tool(
        self,
        tool_name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Observation:
        conversation, cell_call_state = self._current_call_context()
        try:
            cell_call_state.begin(tool_name)
        except (
            _ToolCallBudgetExceededError,
            _ToolCallAdmissionsClosedError,
        ) as exc:
            self._record_nested_exception(tool_name, exc)
            raise
        try:
            return self._execute_tool_call(
                conversation,
                tool_name,
                args,
                kwargs,
            )
        finally:
            cell_call_state.finish()

    async def acall_tool(
        self,
        tool_name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Observation:
        conversation, cell_call_state = self._current_call_context()
        try:
            cell_call_state.begin(tool_name)
        except (
            _ToolCallBudgetExceededError,
            _ToolCallAdmissionsClosedError,
        ) as exc:
            self._record_nested_exception(tool_name, exc)
            raise

        try:
            concurrent_worker = self._tool_thread_pool.submit(
                self._execute_tool_call,
                conversation,
                tool_name,
                args,
                kwargs,
            )
        except BaseException:
            cell_call_state.finish()
            raise

        try:
            # asyncio.wrap_future installs its thread-safe loop notification on
            # the concurrent Future. Add the admission-release callback after it,
            # so wait_until_idle cannot return before notification is enqueued.
            worker = asyncio.wrap_future(concurrent_worker, loop=self._loop)
        finally:
            concurrent_worker.add_done_callback(
                lambda _future: cell_call_state.finish()
            )
        worker.add_done_callback(self._consume_async_worker_exception)

        # Cancellation of the asyncio wrapper must not cancel a queued or
        # running environment mutation. The per-cell state keeps the admission
        # live, and cell cleanup waits for the concurrent worker to finish.
        return await asyncio.shield(worker)

    def _execute_tool_call(
        self,
        conversation: LocalConversation,
        tool_name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Observation:
        if tool_name == self._tool_name:
            raise ValueError("programmatic_tool_calling cannot call itself.")

        try:
            tool = self._get_tool(conversation, tool_name)
            if tool.executor is None:
                raise NotImplementedError(f"Tool '{tool_name}' has no executor")

            arguments = self._normalize_tool_arguments(args, kwargs)
            action = tool.action_from_arguments(arguments)
            lock_keys = self._resolve_lock_keys(tool, action)

            with self._lock_tool_resources(lock_keys):
                observation = tool(action, conversation)
        except Exception as exc:
            self._record_nested_exception(tool_name, exc)
            raise
        if observation.is_error:
            self._record_nested_tool_error(tool_name, observation.text)
        return observation

    @staticmethod
    def _consume_async_worker_exception(worker: asyncio.Future) -> None:
        if worker.cancelled():
            return
        try:
            worker.exception()
        except BaseException:
            # Retrieval prevents a detached, shielded worker from producing an
            # "exception was never retrieved" warning after its wrapper is
            # cancelled. The awaiting user task still receives the exception.
            pass

    def _current_call_context(self) -> tuple[LocalConversation, _CellCallState]:
        conversation = self._conversation
        cell_call_state = self._cell_call_state
        if conversation is None or cell_call_state is None:
            raise RuntimeError("OpenHands tools can only be called during execution.")
        return conversation, cell_call_state

    def available_tool_names(self) -> list[str]:
        conversation = self._conversation
        if conversation is None:
            return []
        return [
            name for name in conversation.agent.tools_map if name != self._tool_name
        ]

    def _create_shell(self) -> InteractiveShellEmbed:
        original_excepthook = sys.excepthook
        config = Config()
        config.HistoryManager.enabled = False
        shell = InteractiveShellEmbed(config=config)
        sys.excepthook = original_excepthook
        shell.user_ns["tools"] = _ToolNamespace(self)
        shell.user_ns["atools"] = _AsyncToolNamespace(self)
        shell.user_ns["call_tool"] = self._call_tool_by_name
        shell.user_ns["acall_tool"] = self._acall_tool_by_name
        shell.user_ns.setdefault("asyncio", asyncio)
        return shell

    def _run_cell(self, code: str) -> tuple[Any, int]:
        cell_call_state = self._cell_call_state
        if cell_call_state is None:
            raise RuntimeError("No per-cell tool-call state is installed.")
        try:
            preprocessing_exc_tuple = None
            try:
                transformed_cell = self._shell.transform_cell(code)
            except Exception:
                transformed_cell = None
                preprocessing_exc_tuple = sys.exc_info()
            if self._mode is ProgrammaticToolCallingMode.ORCHESTRATION_ONLY:
                validate_orchestration_code(
                    transformed_cell if transformed_cell is not None else code,
                    self.available_tool_names(),
                )
            result = self._loop.run_until_complete(
                self._shell.run_cell_async(
                    code,
                    store_history=True,
                    transformed_cell=transformed_cell,
                    preprocessing_exc_tuple=preprocessing_exc_tuple,
                )
            )
        finally:
            # First reject any worker that has not yet entered a direct tool call,
            # then cancel asyncio wrappers and retrieve their results. Cancelling
            # the wrapper cannot stop its concurrent worker, so wait for every
            # admitted call while the current conversation remains installed.
            # Only then may the caller clear it or begin another cell.
            cell_call_state.close_admissions()
            orphaned_task_count = self._cancel_and_drain_pending_tasks()
            cell_call_state.wait_until_idle()
            self._flush_loop_callbacks()
        return result, orphaned_task_count

    def _cancel_and_drain_pending_tasks(self) -> int:
        """Cancel tasks left on this executor's private loop and retrieve results."""
        if self._loop.is_closed():
            return 0

        pending = tuple(asyncio.all_tasks(self._loop))
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        return len(pending)

    def _flush_loop_callbacks(self) -> None:
        """Deliver worker completion and exception-retrieval callbacks."""
        if self._loop.is_closed():
            return
        # Completing the wrapped Future schedules its done callbacks, so two
        # zero-delay turns are intentional.
        self._loop.run_until_complete(asyncio.sleep(0))
        self._loop.run_until_complete(asyncio.sleep(0))

    @staticmethod
    def _log_cell_telemetry(
        execution_count: int,
        telemetry: _CellCallTelemetry,
    ) -> None:
        log = logger.warning if telemetry.limit_reached else logger.debug
        log(
            "Programmatic tool cell %d nested-call telemetry: "
            "limit=%d attempts=%d admitted=%d completed=%d rejected=%d "
            "limit_reached=%s",
            execution_count,
            telemetry.limit,
            telemetry.attempts,
            telemetry.admitted,
            telemetry.completed,
            telemetry.rejected,
            telemetry.limit_reached,
            extra={
                "ptc_execution_count": execution_count,
                "ptc_tool_call_limit": telemetry.limit,
                "ptc_tool_call_attempts": telemetry.attempts,
                "ptc_tool_calls_admitted": telemetry.admitted,
                "ptc_tool_calls_completed": telemetry.completed,
                "ptc_tool_calls_rejected": telemetry.rejected,
                "ptc_tool_call_limit_reached": telemetry.limit_reached,
            },
        )

    def _consume_future_exceptions_from_traceback(
        self,
        error: BaseException | None,
    ) -> None:
        """Retrieve abandoned same-loop Future errors retained by a traceback."""
        if error is None:
            return

        traceback_frame = error.__traceback__
        seen: set[int] = set()
        while traceback_frame is not None:
            for value in traceback_frame.tb_frame.f_locals.values():
                if not isinstance(value, asyncio.Future) or id(value) in seen:
                    continue
                seen.add(id(value))
                if value.get_loop() is not self._loop or not value.done():
                    continue
                try:
                    value.exception()
                except BaseException:
                    # Retrieving a cancelled Future raises CancelledError. The call
                    # still marks it handled, which is all cleanup requires here.
                    pass
            traceback_frame = traceback_frame.tb_next

    @staticmethod
    def _orphaned_task_guidance(orphaned_task_count: int) -> str | None:
        if not orphaned_task_count:
            return None
        noun = "task" if orphaned_task_count == 1 else "tasks"
        return (
            f"Canceled {orphaned_task_count} background asyncio {noun} left "
            "running by this cell. Await all asynchronous work before the cell "
            "ends. Use top-level `await` (for example, "
            "`await asyncio.gather(...)`) instead of `asyncio.run(...)` or an "
            "unawaited `asyncio.create_task(...)`."
        )

    @staticmethod
    def _join_guidance(*parts: str | None) -> str | None:
        guidance = [part for part in parts if part]
        return "\n".join(guidance) or None

    def _install_tool_namespace(self, conversation: LocalConversation) -> None:
        shell_ns = self._shell.user_ns
        self._routing_index = ToolRoutingIndex.from_tools(
            conversation.agent.tools_map,
            excluded_tool_name=self._tool_name,
        )
        shell_ns["tools"] = _ToolNamespace(self)
        shell_ns["atools"] = _AsyncToolNamespace(self)
        shell_ns["call_tool"] = self._call_tool_by_name
        shell_ns["acall_tool"] = self._acall_tool_by_name
        shell_ns.setdefault("asyncio", asyncio)
        if self._mode is ProgrammaticToolCallingMode.ORCHESTRATION_ONLY:
            shell_ns.update(orchestration_only_bindings())

        for tool_name in conversation.agent.tools_map:
            if tool_name == self._tool_name:
                continue
            if not tool_name.isidentifier() or keyword.iskeyword(tool_name):
                continue
            if tool_name in _RESERVED_NAMES:
                continue
            shell_ns[tool_name] = _ToolFunction(self, tool_name)

    def _call_tool_by_name(self, tool_name: str, **kwargs: Any) -> Observation:
        return self.call_tool(tool_name, (), kwargs)

    async def _acall_tool_by_name(self, tool_name: str, **kwargs: Any) -> Observation:
        return await self.acall_tool(tool_name, (), kwargs)

    def _lock_tool_resources(self, lock_keys: list[str]):
        if not lock_keys:
            return nullcontext()
        return self._resource_lock_manager.lock(*lock_keys)

    def _get_tool(
        self,
        conversation: LocalConversation,
        tool_name: str,
    ) -> ToolDefinition:
        self._require_string_tool_name(tool_name)
        if tool_name == self._tool_name:
            raise ValueError("programmatic_tool_calling cannot call itself.")

        tool = conversation.agent.tools_map.get(tool_name)
        if tool is None:
            raise ToolNotFoundError(tool_name, self.available_tool_names())
        return tool

    def _require_string_tool_name(self, tool_name: object) -> None:
        if isinstance(tool_name, str):
            return

        message = (
            "An OpenHands tool name must be a string; received "
            f"{type(tool_name).__name__}. The bare `call_tool(...)` and "
            "`acall_tool(...)` helpers expect a direct OpenHands tool name, for "
            'example `call_tool("<tool_name>", **arguments)`.'
        )
        if "call_tool" in self.available_tool_names():
            message += (
                " An OpenHands tool also named `call_tool` is available. To pass "
                "a request mapping to that tool, use `tools.call_tool({...})` or "
                "`await atools.call_tool({...})`."
            )
        raise TypeError(message)

    @staticmethod
    def _resolve_lock_keys(tool: ToolDefinition, action: Action) -> list[str]:
        resources = tool.declared_resources(action)
        if not resources.declared:
            return [f"tool:{tool.name}"]
        return list(resources.keys)

    def _normalize_tool_arguments(
        self,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if len(args) == 1 and isinstance(args[0], Mapping) and not kwargs:
            return dict(args[0])
        if args:
            raise TypeError(
                "OpenHands tool functions accept keyword arguments, or a single "
                "mapping positional argument."
            )
        return dict(kwargs)

    def _format_output(
        self,
        *,
        stdout: str,
        stderr: str,
        result: Any,
        error: BaseException | None,
        guidance: str | None,
        nested_tool_errors: Sequence[_NestedToolError],
    ) -> str:
        parts: list[str] = []

        clean_stdout = self._strip_ipython_out_prompt(self._strip_ansi(stdout)).strip()
        if clean_stdout:
            parts.append(clean_stdout)

        clean_stderr = self._strip_ansi(stderr).strip()
        if clean_stderr:
            parts.append(clean_stderr)

        if result is not None:
            parts.append(f"Result:\n{result!r}")

        if error is not None and not clean_stderr:
            parts.append("Traceback:\n" + "".join(traceback.format_exception(error)))

        if nested_tool_errors:
            nested = []
            for tool_error in nested_tool_errors:
                text = tool_error.text.strip() or "The tool returned an empty error."
                nested.append(f"{tool_error.tool_name}: {text}")
            parts.append("Nested tool call errors:\n" + "\n".join(nested))

        if guidance is not None:
            parts.append("Corrective guidance:\n" + guidance)

        if not parts:
            return "Executed successfully with no output."
        return "\n\n".join(parts)

    def _classify_error(self, error: BaseException | None) -> _ExecutionError:
        if error is None:
            return _ExecutionError()
        if isinstance(error, _ToolCallBudgetExceededError):
            symbol = error.tool_name if isinstance(error.tool_name, str) else ""
            routes = self._routing_index.for_symbol(symbol) if symbol else ()
            return _ExecutionError(
                kind=ProgrammaticToolCallingErrorKind.PYTHON_ERROR,
                guidance=(
                    "Use bounded loops and stop retrying or paginating when the "
                    "tool response indicates completion. Reuse prior results "
                    "instead of repeatedly issuing the same call."
                ),
                suggested_routes=self._suggested_routes(routes),
            )
        if isinstance(error, _ToolCallAdmissionsClosedError):
            symbol = error.tool_name if isinstance(error.tool_name, str) else ""
            routes = self._routing_index.for_symbol(symbol) if symbol else ()
            return _ExecutionError(
                kind=ProgrammaticToolCallingErrorKind.PYTHON_ERROR,
                guidance=(
                    "Await all scheduled tool calls in the current cell instead "
                    "of leaving background work running."
                ),
                suggested_routes=self._suggested_routes(routes),
            )
        if isinstance(error, OrchestrationPolicyError):
            routes = self._routing_index.for_capabilities(error.capabilities)
            return _ExecutionError(
                kind=ProgrammaticToolCallingErrorKind.POLICY_VIOLATION,
                policy_violation=True,
                guidance=self._routing_index.guidance_for_capabilities(
                    error.capabilities
                ),
                suggested_routes=self._suggested_routes(routes),
            )
        if isinstance(error, ToolNotFoundError):
            routes = self._routing_index.for_symbol(error.tool_name)
            return _ExecutionError(
                kind=ProgrammaticToolCallingErrorKind.TOOL_NOT_FOUND,
                missing_symbol=error.tool_name,
                guidance=self._routing_index.guidance_for_symbol(error.tool_name),
                suggested_routes=self._suggested_routes(routes),
            )
        if isinstance(error, ModuleNotFoundError):
            missing_symbol = error.name
            routes = self._routing_index.for_symbol(missing_symbol or "")
            if (
                not routes
                and self._mode is ProgrammaticToolCallingMode.ORCHESTRATION_ONLY
            ):
                routes = self._routing_index.for_capabilities(("python.execute",))
            guidance = self._guidance_for_routes(routes)
            return _ExecutionError(
                kind=ProgrammaticToolCallingErrorKind.MODULE_NOT_FOUND,
                missing_symbol=missing_symbol,
                guidance=guidance,
                suggested_routes=self._suggested_routes(routes),
            )
        if isinstance(error, NameError):
            missing_symbol = error.name
            capabilities = capabilities_for_missing_symbol(missing_symbol or "")
            if (
                capabilities
                and self._mode is ProgrammaticToolCallingMode.ORCHESTRATION_ONLY
            ):
                routes = self._routing_index.for_capabilities(capabilities)
                return _ExecutionError(
                    kind=ProgrammaticToolCallingErrorKind.POLICY_VIOLATION,
                    policy_violation=True,
                    missing_symbol=missing_symbol,
                    guidance=self._routing_index.guidance_for_capabilities(
                        capabilities
                    ),
                    suggested_routes=self._suggested_routes(routes),
                )
            routes = self._routing_index.for_symbol(missing_symbol or "")
            return _ExecutionError(
                kind=ProgrammaticToolCallingErrorKind.NAME_ERROR,
                missing_symbol=missing_symbol,
                guidance=self._guidance_for_routes(routes),
                suggested_routes=self._suggested_routes(routes),
            )
        return _ExecutionError(kind=ProgrammaticToolCallingErrorKind.PYTHON_ERROR)

    def _guidance_for_routes(self, routes: Sequence[ToolRoute]) -> str | None:
        if not routes:
            if self._mode is ProgrammaticToolCallingMode.ORCHESTRATION_ONLY:
                return self._routing_index.general_guidance()
            return None
        target = routes[0].target_name or routes[0].tool_name
        return self._routing_index.guidance_for_symbol(target)

    @staticmethod
    def _suggested_routes(routes: Sequence[ToolRoute]) -> tuple[str, ...]:
        return tuple(route.render() for route in routes[:3])

    def _format_caught_exception(
        self,
        error: BaseException,
        guidance: str | None,
    ) -> str:
        if isinstance(
            error,
            OrchestrationPolicyError
            | _ToolCallBudgetExceededError
            | _ToolCallAdmissionsClosedError,
        ):
            output = str(error)
        else:
            output = "Traceback:\n" + "".join(traceback.format_exception(error))
        if guidance is not None:
            output += "\n\nCorrective guidance:\n" + guidance
        return output

    def _nested_errors_snapshot(self) -> tuple[_NestedToolError, ...]:
        with self._nested_error_lock:
            return tuple(self._nested_tool_errors)

    def _record_nested_exception(
        self,
        tool_name: str,
        error: BaseException,
    ) -> None:
        execution_error = self._classify_error(error)
        text = f"{type(error).__name__}: {error}"
        if execution_error.guidance is not None:
            text += "\nCorrective guidance:\n" + execution_error.guidance
        self._record_nested_tool_error(tool_name, text)

    def _record_nested_tool_error(self, tool_name: object, text: str) -> None:
        if not isinstance(tool_name, str):
            tool_name = f"<invalid {type(tool_name).__name__} tool name>"
        with self._nested_error_lock:
            self._nested_tool_errors.append(
                _NestedToolError(tool_name=tool_name, text=text)
            )

    @staticmethod
    def _strip_ansi(text: str) -> str:
        return _ANSI_ESCAPE_RE.sub("", text)

    @staticmethod
    def _strip_ipython_out_prompt(text: str) -> str:
        lines = [
            line for line in text.splitlines() if not _IPYTHON_OUT_PROMPT_RE.match(line)
        ]
        return "\n".join(lines)
