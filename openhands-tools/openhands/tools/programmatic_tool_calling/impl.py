from __future__ import annotations

import asyncio
import io
import keyword
import re
import sys
import threading
import traceback
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config.loader import Config

from openhands.sdk.conversation.resource_lock_manager import ResourceLockManager
from openhands.sdk.tool import Observation, ToolExecutor
from openhands.sdk.utils import maybe_truncate
from openhands.tools.programmatic_tool_calling.definition import (
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
        self._executor.require_tool_available(tool_name)
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
        self._executor.require_tool_available(tool_name)
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
    ):
        self._tool_name = tool_name
        self._mode = ProgrammaticToolCallingMode(mode)
        self._shell = self._create_shell()
        self._loop = asyncio.new_event_loop()
        self._lock = threading.RLock()
        self._nested_error_lock = threading.Lock()
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
        if conversation is None:
            return ProgrammaticToolCallingObservation.from_text(
                "programmatic_tool_calling requires a LocalConversation context.",
                is_error=True,
                execution_count=self._execution_count,
                error_kind=ProgrammaticToolCallingErrorKind.PYTHON_ERROR,
            )

        code = action.code.strip()
        if not code:
            return ProgrammaticToolCallingObservation.from_text(
                "No Python code was provided.",
                is_error=True,
                execution_count=self._execution_count,
                error_kind=ProgrammaticToolCallingErrorKind.PYTHON_ERROR,
            )

        with self._lock:
            self._conversation = conversation
            self._install_tool_namespace(conversation)
            with self._nested_error_lock:
                self._nested_tool_errors.clear()
            self._execution_count += 1

            stdout = io.StringIO()
            stderr = io.StringIO()
            execution_error = _ExecutionError()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = self._run_cell(code)

                python_error = getattr(result, "error_before_exec", None) or getattr(
                    result, "error_in_exec", None
                )
                execution_error = self._classify_error(python_error)
                nested_tool_errors = self._nested_errors_snapshot()
                output = self._format_output(
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue(),
                    result=getattr(result, "result", None),
                    error=python_error,
                    guidance=execution_error.guidance,
                    nested_tool_errors=nested_tool_errors,
                )
                is_error = not getattr(result, "success", False) or bool(
                    nested_tool_errors
                )
                if nested_tool_errors and execution_error.kind is None:
                    execution_error = _ExecutionError(
                        kind=ProgrammaticToolCallingErrorKind.TOOL_ERROR
                    )
            except BaseException as exc:
                execution_error = self._classify_error(exc)
                output = self._format_caught_exception(exc, execution_error.guidance)
                nested_tool_errors = self._nested_errors_snapshot()
                is_error = True
            finally:
                self._conversation = None
                self._routing_index = ToolRoutingIndex(
                    routes=(), available_tool_names=()
                )

        failed_tool_names = tuple(
            dict.fromkeys(error.tool_name for error in nested_tool_errors)
        )

        return ProgrammaticToolCallingObservation.from_text(
            maybe_truncate(output, truncate_after=MAX_PROGRAMMATIC_TOOL_OUTPUT_SIZE),
            is_error=is_error,
            execution_count=self._execution_count,
            error_kind=execution_error.kind,
            policy_violation=execution_error.policy_violation,
            missing_symbol=execution_error.missing_symbol,
            suggested_routes=execution_error.suggested_routes,
            failed_tool_names=failed_tool_names,
        )

    @property
    def mode(self) -> ProgrammaticToolCallingMode:
        return self._mode

    def close(self) -> None:
        with self._lock:
            self._conversation = None
            if not self._loop.is_closed():
                self._loop.close()

    def call_tool(
        self,
        tool_name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Observation:
        conversation = self._conversation
        if conversation is None:
            raise RuntimeError("OpenHands tools can only be called during execution.")
        try:
            if tool_name == self._tool_name:
                raise ValueError("programmatic_tool_calling cannot call itself.")

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

    async def acall_tool(
        self,
        tool_name: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> Observation:
        return await asyncio.to_thread(
            self.call_tool,
            tool_name,
            args,
            kwargs,
        )

    def require_tool_available(self, tool_name: str) -> None:
        conversation = self._conversation
        if conversation is None:
            raise RuntimeError("OpenHands tools can only be called during execution.")
        try:
            self._get_tool(conversation, tool_name)
        except Exception as exc:
            self._record_nested_exception(tool_name, exc)
            raise

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

    def _run_cell(self, code: str) -> Any:
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
        return self._loop.run_until_complete(
            self._shell.run_cell_async(
                code,
                store_history=True,
                transformed_cell=transformed_cell,
                preprocessing_exc_tuple=preprocessing_exc_tuple,
            )
        )

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
            "example `call_tool(\"<tool_name>\", **arguments)`."
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
        if isinstance(error, OrchestrationPolicyError):
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
        error: Exception,
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
