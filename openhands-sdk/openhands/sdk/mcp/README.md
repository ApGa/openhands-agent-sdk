# MCP Integration

The SDK can load tools from Model Context Protocol (MCP) servers through
FastMCP. `create_mcp_tools()` accepts a standard `mcp_config`, connects to the
configured server or servers, lists their tools, and wraps each MCP tool as an
`MCPToolDefinition`.

Each wrapped tool:

- exposes the MCP tool schema through the SDK's normal tool interfaces;
- validates model-provided arguments against the MCP input schema before
  execution;
- calls the MCP server through the shared `MCPClient`;
- returns MCP content blocks as SDK observations;
- expands and masks configured conversation secrets around tool execution.

For a single MCP server, tool names are the server-provided MCP tool names. For
multi-server configs, FastMCP exposes tools with server-name prefixes such as
`github_create_issue`.

## Declared Resources

MCP tools are conservative by default for parallel execution. Clients or MCP
server authors can explicitly declare the resources an MCP tool touches, and the
SDK feeds those declarations into the existing `DeclaredResources` model. Native
parallel tool execution and programmatic tool calling therefore share the same
locking semantics.

### Semantics

- No declaration means `DeclaredResources(keys=(), declared=False)`.
- A declaration with an empty resource list means
  `DeclaredResources(keys=(), declared=True)`, so no lock is acquired.
- A declaration with one or more resource keys means
  `DeclaredResources(keys=(...), declared=True)`, so those keys are locked.
- Invalid declarations or templates fall back to `declared=False`.

### Client Config

Clients can add OpenHands-specific resource policies to `mcp_config`:

```json
{
  "mcpServers": {
    "github": {
      "url": "https://example.com/mcp"
    }
  },
  "openhands": {
    "mcpToolResourcePolicies": [
      {
        "match": "github.search_*",
        "declaredResources": []
      },
      {
        "match": "github.create_issue",
        "declaredResources": ["github:repo:{owner}/{repo}:issues"]
      },
      {
        "match": "browser.*",
        "declaredResources": ["mcp-server:{server_name}"]
      }
    ]
  }
}
```

`match` uses `fnmatch` patterns. It is evaluated against the final OpenHands
tool name, the original MCP tool name when known, and
`{server_name}.{original_tool_name}` when a server is known. FastMCP prefixes
tool names in multi-server configs, so `github.create_issue` can match the
visible tool `github_create_issue`.

The first matching client policy wins.

### Server Metadata

MCP server authors can declare resources through tool `_meta`:

```json
{
  "name": "search_issues",
  "_meta": {
    "openhands.dev/declared_resources": []
  }
}
```

Resource-scoped declarations use the same template syntax:

```json
{
  "name": "create_issue",
  "_meta": {
    "openhands.dev/declared_resources": [
      "github:repo:{owner}/{repo}:issues"
    ]
  }
}
```

The metadata also accepts an object form:

```json
{
  "openhands.dev/declared_resources": {
    "keys": ["mcp-server:{server_name}"]
  }
}
```

Client config overrides server metadata so operators can correct declarations
for their deployment.

### Templates

Resource templates can reference MCP action arguments plus these built-ins:

- `{tool_name}` or `{mcp_tool_name}`: final OpenHands tool name.
- `{original_tool_name}`: original MCP tool name when known.
- `{server_name}`: MCP server name when known, otherwise an empty string.

Only simple field names are accepted. Missing fields or invalid templates make
the tool declaration conservative for that call.
