from __future__ import annotations

import keyword
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from openhands.sdk.tool import ToolDefinition


TOOL_ROUTING_META_KEY = "openhands.dev/tool-routing"
_COMPATIBLE_META_KEYS = (
    TOOL_ROUTING_META_KEY,
    "platoon.dev/tool-routing",
    "platoon.dev/ptc",
)
_TASK_EXECUTION_DOMAINS = frozenset({"task", "task_environment", "environment"})
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_CAPABILITY_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MAX_AVAILABLE_TOOLS = 64
_MAX_ROUTES = 64
_MAX_TARGETS_PER_DISPATCHER = 32


@dataclass(frozen=True, slots=True)
class ToolRoute:
    tool_name: str
    capabilities: frozenset[str]
    kind: Literal["direct", "dispatcher"]
    target_name: str | None = None
    name_argument: str = "name"
    arguments_argument: str = "arguments"
    discovery_tool: str | None = None

    def render(self) -> str:
        callable_name = _render_atools_callable(self.tool_name)
        if self.kind == "direct":
            return f"await {callable_name}(...)"

        assert self.target_name is not None
        target = repr(self.target_name)
        arguments = "{...}"
        if self.name_argument.isidentifier() and self.arguments_argument.isidentifier():
            return (
                f"await {callable_name}({self.name_argument}={target}, "
                f"{self.arguments_argument}={arguments})"
            )
        return (
            f"await {callable_name}("
            f"**{{{self.name_argument!r}: {target}, "
            f"{self.arguments_argument!r}: {arguments}}})"
        )

    def render_discovery(self) -> str | None:
        if self.discovery_tool is None or self.target_name is None:
            return None
        callable_name = _render_atools_callable(self.discovery_tool)
        return f"await {callable_name}(name={self.target_name!r})"


@dataclass(frozen=True, slots=True)
class ToolRoutingIndex:
    routes: tuple[ToolRoute, ...]
    available_tool_names: tuple[str, ...]

    @classmethod
    def from_tools(
        cls,
        tools: Mapping[str, ToolDefinition],
        excluded_tool_name: str,
    ) -> ToolRoutingIndex:
        available = tuple(
            sorted(
                name
                for name in tools
                if name != excluded_tool_name and _is_safe_tool_name(name)
            )[:_MAX_AVAILABLE_TOOLS]
        )
        routes: list[ToolRoute] = []
        for tool_name in available:
            routes.extend(_routes_from_tool(tool_name, tools[tool_name].meta))
        unique_routes = _deduplicate_routes(routes)
        sorted_routes = tuple(
            sorted(
                unique_routes,
                key=lambda route: (
                    route.tool_name,
                    route.target_name or "",
                    tuple(sorted(route.capabilities)),
                ),
            )[:_MAX_ROUTES]
        )
        return cls(routes=sorted_routes, available_tool_names=available)

    def for_capabilities(self, capabilities: Sequence[str]) -> tuple[ToolRoute, ...]:
        requested = frozenset(capabilities)
        if not requested:
            return ()
        matching = _deduplicate_routes(
            route for route in self.routes if requested.intersection(route.capabilities)
        )
        return tuple(
            sorted(
                matching,
                key=lambda route: _route_rank(route, requested),
            )
        )

    def for_symbol(self, symbol: str) -> tuple[ToolRoute, ...]:
        normalized = _normalize_name(symbol)
        declared = _deduplicate_routes(
            route
            for route in self.routes
            if _normalize_name(route.target_name or route.tool_name) == normalized
        )
        if declared:
            return declared

        for tool_name in self.available_tool_names:
            if _normalize_name(tool_name) == normalized:
                return (
                    ToolRoute(
                        tool_name=tool_name,
                        capabilities=frozenset(),
                        kind="direct",
                    ),
                )
        return ()

    def guidance_for_capabilities(self, capabilities: Sequence[str]) -> str:
        routes = self.for_capabilities(capabilities)
        if routes:
            return _render_route_guidance(routes)
        return self.general_guidance()

    def guidance_for_symbol(self, symbol: str) -> str:
        routes = self.for_symbol(symbol)
        if routes:
            return _render_route_guidance(routes)
        return self.general_guidance()

    def general_guidance(self) -> str:
        available = (
            ", ".join(repr(name) for name in self.available_tool_names) or "none"
        )
        message = (
            "Use an OpenHands environment tool through `atools` instead. "
            f"Currently available tools: {available}."
        )
        dispatcher = next(
            (
                name
                for name in self.available_tool_names
                if _normalize_name(name) in {"call_tool", "tool_call"}
            ),
            None,
        )
        if dispatcher is None:
            return message
        callable_name = _render_atools_callable(dispatcher)
        return (
            f"{message} This environment exposes a catalog dispatcher; inspect the "
            "task's tool catalog, then call it with "
            f'`await {callable_name}(name="<catalog tool>", arguments={{...}})`.'
        )


def _routes_from_tool(tool_name: str, meta: dict[str, Any] | None) -> list[ToolRoute]:
    if not _is_safe_tool_name(tool_name):
        return []
    routing = _routing_metadata(meta)
    if routing is None:
        return []
    execution_domain = routing.get("execution_domain")
    if (
        not isinstance(execution_domain, str)
        or execution_domain not in _TASK_EXECUTION_DOMAINS
    ):
        return []

    capabilities = _string_set(routing.get("capabilities"))
    invocation = routing.get("invocation")
    if not isinstance(invocation, Mapping):
        dispatch = routing.get("dispatch")
        if isinstance(dispatch, Mapping):
            return _dispatcher_routes(tool_name, dispatch)
        invocation = {}
    kind = invocation.get("kind", "direct")

    if kind == "direct":
        return [
            ToolRoute(
                tool_name=tool_name,
                capabilities=capabilities,
                kind="direct",
            )
        ]
    if kind == "dispatcher":
        return _dispatcher_routes(tool_name, invocation)
    if kind == "catalog":
        return _legacy_catalog_routes(tool_name, capabilities, invocation)

    return []


def _dispatcher_routes(
    tool_name: str,
    invocation: Mapping[str, Any],
) -> list[ToolRoute]:
    name_argument = _argument_name(invocation.get("name_argument"), default="name")
    arguments_argument = _argument_name(
        invocation.get("arguments_argument"), default="arguments"
    )
    if name_argument is None or arguments_argument is None:
        return []
    discovery_tool = _tool_name(invocation.get("discovery_tool"))
    targets = invocation.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return []

    routes: list[ToolRoute] = []
    for target in targets[:_MAX_TARGETS_PER_DISPATCHER]:
        if not isinstance(target, Mapping):
            continue
        target_name = _tool_name(target.get("name"))
        if target_name is None:
            continue
        routes.append(
            ToolRoute(
                tool_name=tool_name,
                capabilities=_string_set(target.get("capabilities")),
                kind="dispatcher",
                target_name=target_name,
                name_argument=name_argument,
                arguments_argument=arguments_argument,
                discovery_tool=discovery_tool,
            )
        )
    return routes


def _legacy_catalog_routes(
    tool_name: str,
    capabilities: frozenset[str],
    invocation: Mapping[str, Any],
) -> list[ToolRoute]:
    target_name = _tool_name(invocation.get("target"))
    if target_name is None:
        return []
    dispatcher = _tool_name(invocation.get("dispatcher")) or tool_name
    name_argument = _argument_name(invocation.get("name_argument"), default="name")
    arguments_argument = _argument_name(
        invocation.get("arguments_argument"), default="arguments"
    )
    if name_argument is None or arguments_argument is None:
        return []
    return [
        ToolRoute(
            tool_name=dispatcher,
            capabilities=capabilities,
            kind="dispatcher",
            target_name=target_name,
            name_argument=name_argument,
            arguments_argument=arguments_argument,
            discovery_tool=_tool_name(invocation.get("discovery_tool")),
        )
    ]


def _routing_metadata(meta: dict[str, Any] | None) -> Mapping[str, Any] | None:
    if meta is None:
        return None
    for key in _COMPATIBLE_META_KEYS:
        value = meta.get(key)
        if (
            isinstance(value, Mapping)
            and type(value.get("version")) is int
            and value["version"] == 1
        ):
            return value
    return None


def _render_route_guidance(routes: Sequence[ToolRoute]) -> str:
    rendered: list[str] = []
    for route in routes[:3]:
        discovery = route.render_discovery()
        if discovery is not None:
            rendered.append(f"inspect with `{discovery}`, then use `{route.render()}`")
        else:
            rendered.append(f"use `{route.render()}`")
    return "Use the task-environment route: " + "; or ".join(rendered) + "."


def _render_atools_callable(tool_name: str) -> str:
    if tool_name.isidentifier() and not keyword.iskeyword(tool_name):
        return f"atools.{tool_name}"
    return f"atools[{tool_name!r}]"


def _deduplicate_routes(routes: Iterable[ToolRoute]) -> tuple[ToolRoute, ...]:
    unique: list[ToolRoute] = []
    seen: set[tuple[str, str | None]] = set()
    for route in routes:
        key = (route.tool_name, route.target_name)
        if key in seen:
            continue
        seen.add(key)
        unique.append(route)
    return tuple(unique)


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def _route_rank(
    route: ToolRoute,
    requested: frozenset[str],
) -> tuple[int, int, int, int, str, str]:
    overlap = requested.intersection(route.capabilities)
    covers_all = requested.issubset(route.capabilities)
    exact = requested == route.capabilities
    extra_capabilities = route.capabilities.difference(requested)
    return (
        -int(covers_all),
        -int(exact),
        -len(overlap),
        len(extra_capabilities),
        route.tool_name,
        route.target_name or "",
    )


def _tool_name(value: Any) -> str | None:
    if not isinstance(value, str) or not _is_safe_tool_name(value):
        return None
    return value


def _argument_name(value: Any, default: str) -> str | None:
    if value is None:
        return default
    if (
        not isinstance(value, str)
        or len(value) > 64
        or not value.isidentifier()
        or keyword.iskeyword(value)
    ):
        return None
    return value


def _is_safe_tool_name(value: str) -> bool:
    return _TOOL_NAME_RE.fullmatch(value) is not None


def _string_set(value: Any) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return frozenset()
    capabilities = [
        item
        for item in value[:32]
        if isinstance(item, str) and _CAPABILITY_RE.fullmatch(item) is not None
    ]
    return frozenset(capabilities)
