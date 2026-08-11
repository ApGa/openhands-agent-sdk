from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any


_FILESYSTEM_CAPABILITIES = ("filesystem.read", "filesystem.write")
_MODULE_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "aiohttp": ("network.access",),
    "glob": _FILESYSTEM_CAPABILITIES,
    "http": ("network.access",),
    "importlib": ("python.execute",),
    "multiprocessing": ("system.execute",),
    "os": (*_FILESYSTEM_CAPABILITIES, "system.execute"),
    "pathlib": _FILESYSTEM_CAPABILITIES,
    "pty": ("system.execute",),
    "requests": ("network.access",),
    "shutil": _FILESYSTEM_CAPABILITIES,
    "socket": ("network.access",),
    "subprocess": ("system.execute",),
    "tempfile": _FILESYSTEM_CAPABILITIES,
    "urllib": ("network.access",),
}
_PREDEFINED_ALIASES = {
    "Path": "pathlib.Path",
    "PosixPath": "pathlib.PosixPath",
    "PurePath": "pathlib.PurePath",
    "PurePosixPath": "pathlib.PurePosixPath",
    "PureWindowsPath": "pathlib.PureWindowsPath",
    "WindowsPath": "pathlib.WindowsPath",
}
_BUILTIN_OPERATIONS: dict[str, tuple[str, ...]] = {
    "__import__": ("python.execute",),
    "breakpoint": ("system.execute",),
    "compile": ("python.execute",),
    "eval": ("python.execute",),
    "exec": ("python.execute",),
    "input": ("system.execute",),
    "open": _FILESYSTEM_CAPABILITIES,
}
_BLOCKED_METHODS: dict[str, tuple[str, ...]] = {
    "create_subprocess_exec": ("system.execute",),
    "create_subprocess_shell": ("system.execute",),
    "getoutput": ("system.execute",),
    "getstatusoutput": ("system.execute",),
    "popen": ("system.execute",),
    "read_bytes": ("filesystem.read",),
    "read_text": ("filesystem.read",),
    "run_cell_magic": ("system.execute",),
    "run_line_magic": ("system.execute",),
    "system": ("system.execute",),
    "write_bytes": ("filesystem.write",),
    "write_text": ("filesystem.write",),
}
_BLOCKED_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "builtins.__import__": ("python.execute",),
    "builtins.open": _FILESYSTEM_CAPABILITIES,
    "io.open": _FILESYSTEM_CAPABILITIES,
    "sys.modules": ("python.execute",),
}
_READ_OPERATIONS = frozenset(
    {
        "access",
        "exists",
        "getcwd",
        "getmtime",
        "getsize",
        "glob",
        "home",
        "is_dir",
        "is_file",
        "isdir",
        "isfile",
        "iterdir",
        "listdir",
        "lstat",
        "read_bytes",
        "read_text",
        "readlink",
        "resolve",
        "rglob",
        "scandir",
        "stat",
        "walk",
    }
)
_WRITE_OPERATIONS = frozenset(
    {
        "chmod",
        "chown",
        "hardlink_to",
        "link",
        "mkdir",
        "makedirs",
        "remove",
        "removedirs",
        "rename",
        "replace",
        "rmdir",
        "symlink",
        "symlink_to",
        "touch",
        "truncate",
        "unlink",
        "utime",
        "write_bytes",
        "write_text",
    }
)
_SYSTEM_OPERATIONS = frozenset(
    {
        "create_subprocess_exec",
        "create_subprocess_shell",
        "execl",
        "execle",
        "execlp",
        "execlpe",
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "fork",
        "getoutput",
        "getstatusoutput",
        "popen",
        "run",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
    }
)


class OrchestrationPolicyError(RuntimeError):
    def __init__(self, operation: str, capabilities: Sequence[str]):
        self.operation = operation
        self.capabilities = tuple(capabilities)
        super().__init__(
            f"Local operation '{operation}' is unsupported in orchestration-only "
            "PTC and was blocked because this Python runtime is outside the task "
            "environment."
        )


class BlockedLocalModule:
    def __init__(self, module_name: str, capabilities: Sequence[str]):
        self._module_name = module_name
        self._capabilities = tuple(capabilities)

    def __getattr__(self, attribute: str) -> Any:
        operation = f"{self._module_name}.{attribute}"
        raise OrchestrationPolicyError(
            operation,
            capabilities_for_operation(operation) or self._capabilities,
        )

    def __repr__(self) -> str:
        return f"<{self._module_name} disabled by orchestration-only PTC>"


def blocked_open(*args: Any, **kwargs: Any) -> Any:
    mode = kwargs.get("mode")
    if mode is None and len(args) >= 2:
        mode = args[1]
    raise OrchestrationPolicyError("open", _open_capabilities(mode))


def orchestration_only_bindings() -> dict[str, Any]:
    bindings: dict[str, Any] = {"open": blocked_open}
    for module_name, capabilities in _MODULE_CAPABILITIES.items():
        bindings[module_name] = BlockedLocalModule(module_name, capabilities)
    for alias, canonical in _PREDEFINED_ALIASES.items():
        module_name = canonical.partition(".")[0]
        bindings[alias] = BlockedLocalModule(
            canonical, _MODULE_CAPABILITIES[module_name]
        )
    return bindings


def validate_orchestration_code(
    code: str,
    available_tool_names: Sequence[str],
) -> None:
    tree = ast.parse(code, mode="exec")
    _OrchestrationOnlyValidator(available_tool_names).visit(tree)


def capabilities_for_missing_symbol(symbol: str) -> tuple[str, ...]:
    root = symbol.partition(".")[0]
    if root in _MODULE_CAPABILITIES:
        return _MODULE_CAPABILITIES[root]
    if symbol in _PREDEFINED_ALIASES:
        module_name = _PREDEFINED_ALIASES[symbol].partition(".")[0]
        return _MODULE_CAPABILITIES[module_name]
    return _BUILTIN_OPERATIONS.get(symbol, ())


def capabilities_for_operation(operation: str) -> tuple[str, ...]:
    exact = _BLOCKED_ATTRIBUTES.get(operation)
    if exact is not None:
        return exact
    operation_name = operation.rpartition(".")[2]
    if operation_name in _READ_OPERATIONS:
        return ("filesystem.read",)
    if operation_name in _WRITE_OPERATIONS:
        return ("filesystem.write",)
    if operation_name in _SYSTEM_OPERATIONS:
        return ("system.execute",)
    root = operation.partition(".")[0]
    return _MODULE_CAPABILITIES.get(root, ())


class _OrchestrationOnlyValidator(ast.NodeVisitor):
    def __init__(self, available_tool_names: Sequence[str]):
        self._available_tool_names = frozenset(available_tool_names)
        self._aliases = dict(_PREDEFINED_ALIASES)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module_name = alias.name.partition(".")[0]
            local_name = alias.asname or module_name
            self._aliases[local_name] = alias.name
            self._reject_module(module_name, f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self.generic_visit(node)
            return
        module_name = node.module.partition(".")[0]
        self._reject_module(module_name, f"import from {node.module}")
        for alias in node.names:
            local_name = alias.asname or alias.name
            canonical = f"{node.module}.{alias.name}"
            self._aliases[local_name] = canonical
            capabilities = _BLOCKED_ATTRIBUTES.get(canonical)
            if capabilities is not None:
                raise OrchestrationPolicyError(f"import {canonical}", capabilities)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts = _attribute_parts(node)
        if not parts or parts[0] in {"atools", "tools"}:
            self.generic_visit(node)
            return
        canonical = self._canonicalize(parts)
        operation = ".".join(canonical)
        root = canonical[0]
        if root in _MODULE_CAPABILITIES:
            raise OrchestrationPolicyError(
                operation, capabilities_for_operation(operation)
            )
        capabilities = _BLOCKED_ATTRIBUTES.get(operation)
        if capabilities is not None:
            raise OrchestrationPolicyError(operation, capabilities)
        if len(canonical) >= 2 and canonical[:2] == ("sys", "modules"):
            raise OrchestrationPolicyError(operation, ("python.execute",))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        getattr_parts = _getattr_parts(node)
        if getattr_parts:
            canonical_getattr = self._canonicalize(getattr_parts)
            getattr_operation = ".".join(canonical_getattr)
            getattr_capabilities = capabilities_for_operation(getattr_operation)
            if (
                canonical_getattr[0] in _MODULE_CAPABILITIES
                or getattr_operation in _BLOCKED_ATTRIBUTES
            ):
                raise OrchestrationPolicyError(getattr_operation, getattr_capabilities)

        parts = _attribute_parts(node.func)
        if not parts:
            self.generic_visit(node)
            return
        if len(parts) == 1 and parts[0] in self._available_tool_names:
            self.generic_visit(node)
            return
        if parts[0] in {"atools", "tools", "acall_tool", "call_tool"}:
            self.generic_visit(node)
            return

        canonical = self._canonicalize(parts)
        root = canonical[0]
        operation = ".".join(canonical)
        if operation in {"open", "builtins.open", "io.open"}:
            raise OrchestrationPolicyError(
                operation,
                _open_capabilities(_open_mode(node)),
            )
        if root in _MODULE_CAPABILITIES:
            raise OrchestrationPolicyError(
                operation,
                capabilities_for_operation(operation),
            )
        if len(canonical) == 1 and root in _BUILTIN_OPERATIONS:
            raise OrchestrationPolicyError(root, _BUILTIN_OPERATIONS[root])
        if root == "get_ipython" or canonical[-1] in _BLOCKED_METHODS:
            raise OrchestrationPolicyError(
                operation,
                _BLOCKED_METHODS.get(canonical[-1], ("system.execute",)),
            )
        self.generic_visit(node)

    def _canonicalize(self, parts: tuple[str, ...]) -> tuple[str, ...]:
        aliased = self._aliases.get(parts[0])
        if aliased is None:
            return parts
        return (*aliased.split("."), *parts[1:])

    @staticmethod
    def _reject_module(module_name: str, operation: str) -> None:
        capabilities = _MODULE_CAPABILITIES.get(module_name)
        if capabilities is not None:
            raise OrchestrationPolicyError(operation, capabilities)


def _attribute_parts(node: ast.expr) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        parent = _attribute_parts(node.value)
        if parent:
            return (*parent, node.attr)
    if isinstance(node, ast.Call):
        return _attribute_parts(node.func)
    return ()


def _getattr_parts(node: ast.Call) -> tuple[str, ...]:
    if _attribute_parts(node.func) != ("getattr",) or len(node.args) < 2:
        return ()
    owner = _attribute_parts(node.args[0])
    attribute = node.args[1]
    if not owner or not isinstance(attribute, ast.Constant):
        return ()
    if not isinstance(attribute.value, str):
        return ()
    return (*owner, attribute.value)


def _open_mode(node: ast.Call) -> Any:
    for keyword_argument in node.keywords:
        if keyword_argument.arg == "mode" and isinstance(
            keyword_argument.value, ast.Constant
        ):
            return keyword_argument.value.value
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        return node.args[1].value
    return None


def _open_capabilities(mode: Any) -> tuple[str, ...]:
    if not isinstance(mode, str):
        return ("filesystem.read",)
    writes = any(flag in mode for flag in "wax+")
    reads = "r" in mode or "+" in mode
    if reads and writes:
        return _FILESYSTEM_CAPABILITIES
    if writes:
        return ("filesystem.write",)
    return ("filesystem.read",)
