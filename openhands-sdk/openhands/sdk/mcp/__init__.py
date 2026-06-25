"""MCP (Model Context Protocol) integration for agent-sdk."""

from openhands.sdk.mcp.client import MCPClient
from openhands.sdk.mcp.definition import MCPToolAction, MCPToolObservation
from openhands.sdk.mcp.exceptions import MCPError, MCPTimeoutError
from openhands.sdk.mcp.resources import (
    DECLARED_RESOURCES_META_KEY,
    MCP_TOOL_RESOURCE_POLICIES_KEY,
    OPENHANDS_MCP_CONFIG_KEY,
    MCPToolResourcePolicy,
)
from openhands.sdk.mcp.tool import (
    MCPToolDefinition,
    MCPToolExecutor,
)
from openhands.sdk.mcp.utils import (
    create_mcp_tools,
)


__all__ = [
    "MCPClient",
    "MCPToolDefinition",
    "MCPToolAction",
    "MCPToolObservation",
    "MCPToolExecutor",
    "MCPToolResourcePolicy",
    "OPENHANDS_MCP_CONFIG_KEY",
    "MCP_TOOL_RESOURCE_POLICIES_KEY",
    "DECLARED_RESOURCES_META_KEY",
    "create_mcp_tools",
    "MCPError",
    "MCPTimeoutError",
]
