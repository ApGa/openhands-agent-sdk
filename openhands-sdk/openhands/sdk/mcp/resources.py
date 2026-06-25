"""Declared resource policy helpers for MCP tools."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from string import Formatter
from typing import Any

from fastmcp.mcp_config import MCPConfig
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import DeclaredResources


logger = get_logger(__name__)

OPENHANDS_MCP_CONFIG_KEY = "openhands"
MCP_TOOL_RESOURCE_POLICIES_KEY = "mcpToolResourcePolicies"
MCP_TOOL_RESOURCE_POLICIES_SNAKE_KEY = "mcp_tool_resource_policies"
DECLARED_RESOURCES_META_KEY = "openhands.dev/declared_resources"
_FORMAT_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MCPToolResourcePolicy(BaseModel):
    """OpenHands resource declaration policy for an MCP tool."""

    model_config = ConfigDict(
        frozen=True,
        populate_by_name=True,
        extra="forbid",
    )

    match: str = Field(
        default="*",
        description=(
            "fnmatch pattern for the MCP tool. It is tested against the final "
            "OpenHands tool name, the original MCP tool name when known, and "
            "'{server_name}.{original_tool_name}' when a server is known."
        ),
    )
    declared_resources: tuple[str, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("declared_resources", "declaredResources"),
        description=(
            "Resource key templates this tool touches. An empty list explicitly "
            "declares that no shared resources need locking."
        ),
    )

    def matches(
        self,
        *,
        tool_name: str,
        original_tool_name: str,
        server_name: str | None,
    ) -> bool:
        candidates = {tool_name, original_tool_name}
        if server_name is not None:
            candidates.add(f"{server_name}.{original_tool_name}")
            candidates.add(f"{server_name}.{tool_name}")
        return any(fnmatchcase(candidate, self.match) for candidate in candidates)

    def to_declared_resources(
        self,
        *,
        arguments: Mapping[str, Any],
        tool_name: str,
        original_tool_name: str,
        server_name: str | None,
    ) -> DeclaredResources:
        template_values = {
            "tool_name": tool_name,
            "mcp_tool_name": tool_name,
            "original_tool_name": original_tool_name,
            "server_name": server_name or "",
            **arguments,
        }
        rendered_keys: list[str] = []
        for template in self.declared_resources:
            rendered = _render_resource_template(template, template_values)
            if rendered is None:
                return DeclaredResources(keys=(), declared=False)
            rendered_keys.append(rendered)
        return DeclaredResources(
            keys=tuple(dict.fromkeys(rendered_keys)), declared=True
        )


def resource_policies_from_config(
    config: dict[str, Any] | MCPConfig,
) -> tuple[MCPToolResourcePolicy, ...]:
    """Extract OpenHands MCP resource policies from an MCP config."""

    if isinstance(config, MCPConfig):
        extra = config.model_extra or {}
    else:
        extra = config

    openhands_config = extra.get(OPENHANDS_MCP_CONFIG_KEY)
    if openhands_config is None:
        return ()
    if not isinstance(openhands_config, Mapping):
        raise ValueError("mcp_config.openhands must be an object when provided")

    raw_policies = openhands_config.get(
        MCP_TOOL_RESOURCE_POLICIES_KEY,
        openhands_config.get(MCP_TOOL_RESOURCE_POLICIES_SNAKE_KEY, ()),
    )
    if raw_policies is None:
        return ()
    if not isinstance(raw_policies, Sequence) or isinstance(raw_policies, str):
        raise ValueError("mcp_config.openhands.mcpToolResourcePolicies must be a list")

    policies: list[MCPToolResourcePolicy] = []
    for raw_policy in raw_policies:
        policies.append(MCPToolResourcePolicy.model_validate(raw_policy))
    return tuple(policies)


def infer_mcp_server_and_original_tool_name(
    *, tool_name: str, server_names: Sequence[str]
) -> tuple[str | None, str]:
    """Infer source server information from FastMCP's visible tool name."""

    if len(server_names) == 1:
        return server_names[0], tool_name

    for server_name in sorted(server_names, key=len, reverse=True):
        prefix = f"{server_name}_"
        if tool_name.startswith(prefix):
            return server_name, tool_name.removeprefix(prefix)
    return None, tool_name


def select_mcp_tool_resource_policy(
    *,
    tool_name: str,
    original_tool_name: str,
    server_name: str | None,
    meta: Mapping[str, Any] | None,
    config_policies: Sequence[MCPToolResourcePolicy],
) -> MCPToolResourcePolicy | None:
    """Select the resource policy for an MCP tool.

    Client config has precedence over server-authored metadata.
    """

    for policy in config_policies:
        if policy.matches(
            tool_name=tool_name,
            original_tool_name=original_tool_name,
            server_name=server_name,
        ):
            return policy

    return resource_policy_from_meta(meta)


def resource_policy_from_meta(
    meta: Mapping[str, Any] | None,
) -> MCPToolResourcePolicy | None:
    if not meta or DECLARED_RESOURCES_META_KEY not in meta:
        return None

    raw_policy = meta[DECLARED_RESOURCES_META_KEY]
    try:
        if isinstance(raw_policy, Mapping):
            policy_data = dict(raw_policy)
            if "keys" in policy_data:
                policy_data["declared_resources"] = policy_data.pop("keys")
            return MCPToolResourcePolicy.model_validate(policy_data)
        if isinstance(raw_policy, Sequence) and not isinstance(raw_policy, str):
            return MCPToolResourcePolicy(declared_resources=tuple(raw_policy))
    except ValidationError:
        logger.warning(
            "Ignoring invalid MCP declared resource metadata",
            exc_info=True,
        )
        return None

    logger.warning("Ignoring invalid MCP declared resource metadata: %r", raw_policy)
    return None


def _render_resource_template(
    template: str,
    values: Mapping[str, Any],
) -> str | None:
    try:
        _validate_template_fields(template)
        rendered = template.format_map(values)
    except (KeyError, ValueError):
        logger.warning(
            "MCP declared resource template %r could not be rendered",
            template,
            exc_info=True,
        )
        return None

    if not rendered:
        logger.warning("MCP declared resource template %r rendered empty", template)
        return None
    return rendered


def _validate_template_fields(template: str) -> None:
    for _, field_name, _, _ in Formatter().parse(template):
        if field_name and not _FORMAT_FIELD_RE.fullmatch(field_name):
            raise ValueError(f"Unsupported MCP resource template field: {field_name!r}")
