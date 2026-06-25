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
    ProgrammaticToolCallingObservation,
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


@dataclass(slots=True)
class _ToolCallRecord:
    sequence: int
    tool_name: str
    arguments: dict[str, Any]
    observation: Observation


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

    def __init__(self, tool_name: str):
        self._tool_name = tool_name
        self._shell = self._create_shell()
        self._loop = asyncio.new_event_loop()
        self._lock = threading.RLock()
        self._tool_call_lock = threading.RLock()
        self._resource_lock_manager = ResourceLockManager()
        self._conversation: LocalConversation | None = None
        self._records: list[_ToolCallRecord] = []
        self._next_tool_call_sequence = 0
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
            )

        code = action.code.strip()
        if not code:
            return ProgrammaticToolCallingObservation.from_text(
                "No Python code was provided.",
                is_error=True,
                execution_count=self._execution_count,
            )

        with self._lock:
            self._conversation = conversation
            self._records = []
            self._next_tool_call_sequence = 0
            self._install_tool_namespace(conversation)
            self._execution_count += 1

            stdout = io.StringIO()
            stderr = io.StringIO()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = self._run_cell(code)

                output = self._format_output(
                    stdout=stdout.getvalue(),
                    stderr=stderr.getvalue(),
                    result=getattr(result, "result", None),
                    error=getattr(result, "error_before_exec", None)
                    or getattr(result, "error_in_exec", None),
                )
                is_error = not getattr(result, "success", False)
            except BaseException as exc:
                output = "Traceback:\n" + "".join(traceback.format_exception(exc))
                is_error = True
            finally:
                self._conversation = None

        return ProgrammaticToolCallingObservation.from_text(
            maybe_truncate(output, truncate_after=MAX_PROGRAMMATIC_TOOL_OUTPUT_SIZE),
            is_error=is_error,
            execution_count=self._execution_count,
        )

    def close(self) -> None:
        with self._lock:
            self._conversation = None
            self._records = []
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
        if tool_name == self._tool_name:
            raise ValueError("programmatic_tool_calling cannot call itself.")

        tool = self._get_tool(conversation, tool_name)
        if tool.executor is None:
            raise NotImplementedError(f"Tool '{tool_name}' has no executor")

        arguments = self._normalize_tool_arguments(args, kwargs)
        action = tool.action_from_arguments(arguments)
        lock_keys = self._resolve_lock_keys(tool, action)
        with self._tool_call_lock:
            sequence = self._next_tool_call_sequence
            self._next_tool_call_sequence += 1

        with self._lock_tool_resources(lock_keys):
            observation = tool(action, conversation)

        with self._tool_call_lock:
            self._records.append(
                _ToolCallRecord(
                    sequence=sequence,
                    tool_name=tool_name,
                    arguments=arguments,
                    observation=observation,
                )
            )
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
        self._get_tool(conversation, tool_name)

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
        shell_ns["tools"] = _ToolNamespace(self)
        shell_ns["atools"] = _AsyncToolNamespace(self)
        shell_ns["call_tool"] = self._call_tool_by_name
        shell_ns["acall_tool"] = self._acall_tool_by_name
        shell_ns.setdefault("asyncio", asyncio)

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
        if tool_name == self._tool_name:
            raise ValueError("programmatic_tool_calling cannot call itself.")

        tool = conversation.agent.tools_map.get(tool_name)
        if tool is None:
            available = self.available_tool_names()
            raise KeyError(
                f"Tool '{tool_name}' not found. Available tools: {available}"
            )
        return tool

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

        if self._records:
            parts.append(self._format_tool_call_records())

        if error is not None and not clean_stderr:
            parts.append("Traceback:\n" + "".join(traceback.format_exception(error)))

        if not parts:
            return "Executed successfully with no output."
        return "\n\n".join(parts)

    def _format_tool_call_records(self) -> str:
        lines = ["Tool calls:"]
        records = sorted(self._records, key=lambda record: record.sequence)
        for index, record in enumerate(records, start=1):
            text = record.observation.text
            if not text:
                text = repr(record.observation.model_dump(exclude={"content"}))
            text = maybe_truncate(text.strip(), truncate_after=2_000)
            lines.append(f"{index}. {record.tool_name}({record.arguments!r}) -> {text}")
        return "\n".join(lines)

    @staticmethod
    def _strip_ansi(text: str) -> str:
        return _ANSI_ESCAPE_RE.sub("", text)

    @staticmethod
    def _strip_ipython_out_prompt(text: str) -> str:
        lines = [
            line for line in text.splitlines() if not _IPYTHON_OUT_PROMPT_RE.match(line)
        ]
        return "\n".join(lines)
