"""Utility functions for MCP integration."""

import logging

import mcp.types
from fastmcp.client.logging import LogMessage
from fastmcp.mcp_config import MCPConfig

from openhands.sdk.logger import get_logger
from openhands.sdk.mcp.client import MCPClient
from openhands.sdk.mcp.exceptions import MCPTimeoutError
from openhands.sdk.mcp.resources import (
    MCPToolResourcePolicy,
    infer_mcp_server_and_original_tool_name,
    resource_policies_from_config,
    select_mcp_tool_resource_policy,
)
from openhands.sdk.mcp.tool import MCPToolDefinition


logger = get_logger(__name__)
LOGGING_LEVEL_MAP = logging.getLevelNamesMapping()


async def log_handler(message: LogMessage):
    """
    Handles incoming logs from the MCP server and forwards them
    to the standard Python logging system.
    """
    msg = message.data.get("msg")
    extra = message.data.get("extra")

    # Convert the MCP log level to a Python log level
    level = LOGGING_LEVEL_MAP.get(message.level.upper(), logging.INFO)

    # Log the message using the standard logging library
    logger.log(level, msg, extra=extra)


async def _connect_and_list_tools(
    client: MCPClient,
    server_names: tuple[str, ...],
    resource_policies: tuple[MCPToolResourcePolicy, ...] = (),
) -> None:
    """Connect to MCP server and populate client._tools."""
    await client.connect()
    mcp_type_tools: list[mcp.types.Tool] = await client.list_tools()
    for mcp_tool in mcp_type_tools:
        server_name, original_tool_name = infer_mcp_server_and_original_tool_name(
            tool_name=mcp_tool.name,
            server_names=server_names,
        )
        resource_policy = select_mcp_tool_resource_policy(
            tool_name=mcp_tool.name,
            original_tool_name=original_tool_name,
            server_name=server_name,
            meta=mcp_tool.meta,
            config_policies=resource_policies,
        )
        tool_sequence = MCPToolDefinition.create(
            mcp_tool=mcp_tool,
            mcp_client=client,
            resource_policy=resource_policy,
            server_name=server_name,
            original_tool_name=original_tool_name,
        )
        client._tools.extend(tool_sequence)


def create_mcp_tools(
    config: dict | MCPConfig,
    timeout: float = 30.0,
) -> MCPClient:
    """Create MCP tools from MCP configuration.

    Returns an MCPClient with tools populated. Use as a context manager:

        with create_mcp_tools(config) as client:
            for tool in client.tools:
                # use tool
        # Connection automatically closed
    """
    if isinstance(config, dict):
        config = MCPConfig.model_validate(config)
    resource_policies = resource_policies_from_config(config)
    server_names = tuple(config.mcpServers.keys())
    client = MCPClient(config, log_handler=log_handler)

    try:
        client.call_async_from_sync(
            _connect_and_list_tools,
            timeout=timeout,
            client=client,
            server_names=server_names,
            resource_policies=resource_policies,
        )
    except TimeoutError as e:
        client.sync_close()
        # Extract server names from config for better error message
        server_names = (
            list(config.mcpServers.keys()) if config.mcpServers else ["unknown"]
        )
        error_msg = (
            f"MCP tool listing timed out after {timeout} seconds.\n"
            f"MCP servers configured: {', '.join(server_names)}\n\n"
            "Possible solutions:\n"
            "  1. Increase the timeout value (default is 30 seconds)\n"
            "  2. Check if the MCP server is running and responding\n"
            "  3. Verify network connectivity to the MCP server\n"
        )
        raise MCPTimeoutError(
            error_msg, timeout=timeout, config=config.model_dump()
        ) from e
    except BaseException:
        try:
            client.sync_close()
        except Exception as close_exc:
            logger.warning(
                "Failed to close MCP client during error cleanup", exc_info=close_exc
            )
        raise

    logger.info("Created %d MCP tools", len(client.tools))
    return client
