from unittest.mock import Mock

import mcp.types
from fastmcp.mcp_config import MCPConfig

from openhands.sdk.mcp import DECLARED_RESOURCES_META_KEY
from openhands.sdk.mcp.client import MCPClient
from openhands.sdk.mcp.definition import MCPToolAction
from openhands.sdk.mcp.resources import (
    MCPToolResourcePolicy,
    infer_mcp_server_and_original_tool_name,
    resource_policies_from_config,
    resource_policy_from_meta,
    select_mcp_tool_resource_policy,
)
from openhands.sdk.mcp.tool import MCPToolDefinition


def _mcp_tool(name: str, meta: dict | None = None) -> mcp.types.Tool:
    return mcp.types.Tool(
        name=name,
        description=f"{name} tool",
        inputSchema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
            },
        },
        _meta=meta,
    )


def _wrapped_tool(
    name: str,
    *,
    meta: dict | None = None,
    resource_policy: MCPToolResourcePolicy | None = None,
    server_name: str | None = None,
    original_tool_name: str | None = None,
) -> MCPToolDefinition:
    tools = MCPToolDefinition.create(
        mcp_tool=_mcp_tool(name, meta=meta),
        mcp_client=Mock(spec=MCPClient),
        resource_policy=resource_policy,
        server_name=server_name,
        original_tool_name=original_tool_name,
    )
    return tools[0]


def test_mcp_tool_default_declared_resources_are_conservative() -> None:
    tool = _wrapped_tool("search_issues")

    resources = tool.declared_resources(MCPToolAction(data={"owner": "OpenHands"}))

    assert resources.declared is False
    assert resources.keys == ()


def test_server_metadata_declares_empty_resources() -> None:
    policy = resource_policy_from_meta({DECLARED_RESOURCES_META_KEY: []})
    tool = _wrapped_tool("search_issues", resource_policy=policy)

    resources = tool.declared_resources(MCPToolAction(data={"owner": "OpenHands"}))

    assert resources.declared is True
    assert resources.keys == ()


def test_server_metadata_renders_action_arguments_and_context() -> None:
    policy = resource_policy_from_meta(
        {
            DECLARED_RESOURCES_META_KEY: [
                "github:repo:{owner}/{repo}:issues",
                "mcp:{server_name}:{original_tool_name}",
            ]
        }
    )
    tool = _wrapped_tool(
        "github_create_issue",
        resource_policy=policy,
        server_name="github",
        original_tool_name="create_issue",
    )

    resources = tool.declared_resources(
        MCPToolAction(data={"owner": "OpenHands", "repo": "agent-sdk"})
    )

    assert resources.declared is True
    assert resources.keys == (
        "github:repo:OpenHands/agent-sdk:issues",
        "mcp:github:create_issue",
    )


def test_invalid_resource_template_falls_back_to_conservative() -> None:
    policy = resource_policy_from_meta(
        {DECLARED_RESOURCES_META_KEY: ["github:repo:{missing}"]}
    )
    tool = _wrapped_tool("create_issue", resource_policy=policy)

    resources = tool.declared_resources(MCPToolAction(data={"owner": "OpenHands"}))

    assert resources.declared is False
    assert resources.keys == ()


def test_config_policy_matches_qualified_name_and_overrides_metadata() -> None:
    config_policy = MCPToolResourcePolicy(
        match="github.create_issue",
        declared_resources=("config:{owner}/{repo}",),
    )
    selected = select_mcp_tool_resource_policy(
        tool_name="github_create_issue",
        original_tool_name="create_issue",
        server_name="github",
        meta={DECLARED_RESOURCES_META_KEY: ["metadata:{owner}/{repo}"]},
        config_policies=(config_policy,),
    )
    tool = _wrapped_tool(
        "github_create_issue",
        resource_policy=selected,
        server_name="github",
        original_tool_name="create_issue",
    )

    resources = tool.declared_resources(
        MCPToolAction(data={"owner": "OpenHands", "repo": "agent-sdk"})
    )

    assert resources.declared is True
    assert resources.keys == ("config:OpenHands/agent-sdk",)


def test_resource_policies_from_config_accepts_openhands_extension() -> None:
    config = MCPConfig.model_validate(
        {
            "mcpServers": {"github": {"url": "https://example.com/mcp"}},
            "openhands": {
                "mcpToolResourcePolicies": [
                    {"match": "github.search_*", "declaredResources": []},
                    {
                        "match": "github.create_issue",
                        "declared_resources": ["github:repo:{owner}/{repo}:issues"],
                    },
                ]
            },
        }
    )

    policies = resource_policies_from_config(config)

    assert policies == (
        MCPToolResourcePolicy(match="github.search_*", declared_resources=()),
        MCPToolResourcePolicy(
            match="github.create_issue",
            declared_resources=("github:repo:{owner}/{repo}:issues",),
        ),
    )


def test_resource_policies_from_compact_mcp_config_dump() -> None:
    config = MCPConfig.model_validate(
        {
            "mcpServers": {"github": {"url": "https://example.com/mcp"}},
            "openhands": {
                "mcpToolResourcePolicies": [
                    {"match": "github.search_*", "declaredResources": []}
                ]
            },
        }
    )
    dumped = config.model_dump(exclude_none=True, exclude_defaults=True)

    policies = resource_policies_from_config(dumped)

    assert policies == (
        MCPToolResourcePolicy(match="github.search_*", declared_resources=()),
    )


def test_infers_fastmcp_multiserver_prefixed_tool_names() -> None:
    server_name, original_tool_name = infer_mcp_server_and_original_tool_name(
        tool_name="github_create_issue",
        server_names=("browser", "github"),
    )

    assert server_name == "github"
    assert original_tool_name == "create_issue"
