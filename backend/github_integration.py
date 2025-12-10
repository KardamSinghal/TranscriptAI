"""
Issue Writer Agent - formats and publishes issues.

The agent formats structured data into GitHub issues and can publish them via MCP.
If GitHub is not configured, the agent still formats the issue but skips publishing.

Usage:
    from github_integration import write_issue, write_issues_batch

    result = write_issue("incident_report", tool_input_data)
    results = write_issues_batch("task", [task1, task2, task3])
"""

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

GITHUB_MCP_URL = "https://api.githubcopilot.com/mcp/"
PROMPTS_DIR = Path(__file__).parent / "prompts"
ISSUE_WRITER_PROMPT = (PROMPTS_DIR / "issue_writer.txt").read_text().strip()

# Limit concurrent requests to avoid rate limits
MAX_CONCURRENT_ISSUES = 5


def check_github_config() -> bool:
    """Check and log GitHub configuration status at startup."""
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_ISSUES_REPO")

    if not token:
        print("[github] GitHub integration: DISABLED (GITHUB_TOKEN not set)")
        return False

    if not repo:
        print("[github] GitHub integration: DISABLED (GITHUB_ISSUES_REPO not set)")
        return False

    if "/" not in repo:
        print(
            f"[github] GitHub integration: DISABLED "
            f"(invalid GITHUB_ISSUES_REPO format: '{repo}', expected 'owner/repo')"
        )
        return False

    print(f"[github] GitHub integration: ENABLED (target: {repo})")
    return True


# =============================================================================
# Structured Output Model
# =============================================================================


class GitHubIssueResult(BaseModel):
    """Structured output from Issue Writer Agent."""

    title: str = Field(description="Issue title")
    body: str = Field(description="Markdown-formatted issue body")
    labels: list[str] = Field(description="GitHub labels for the issue")
    github_issue_url: str | None = Field(
        default=None, description="URL if issue was created in GitHub"
    )
    github_issue_number: int | None = Field(
        default=None, description="Issue number if created"
    )


# =============================================================================
# Agent Dependencies
# =============================================================================


@dataclass
class IssueWriterDeps:
    """Dependencies injected into Issue Writer Agent via RunContext.

    The MCP session allows the agent's tool to call GitHub directly.
    If mcp_session is None, the agent can still format issues but won't create them.
    """

    mcp_session: ClientSession | None
    github_owner: str | None
    github_repo: str | None
    # Mutable field to capture tool results
    tool_called: bool = False
    issue_url: str | None = None
    issue_number: int | None = None


# =============================================================================
# Issue Writer Agent (Truly Agentic)
# =============================================================================

_issue_writer_agent: Agent[IssueWriterDeps, GitHubIssueResult] | None = None


def _create_issue_writer_agent() -> Agent[IssueWriterDeps, GitHubIssueResult]:
    """Create Issue Writer Agent with MCP tool access."""
    model = OpenAIChatModel(
        os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5"),
        provider=OpenAIProvider(
            base_url=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.getenv("LLM_API_KEY", ""),
        ),
    )

    agent: Agent[IssueWriterDeps, GitHubIssueResult] = Agent(
        model,
        deps_type=IssueWriterDeps,
        result_type=GitHubIssueResult,
        system_prompt=ISSUE_WRITER_PROMPT,
    )

    @agent.tool
    async def create_github_issue(
        ctx: RunContext[IssueWriterDeps],
        title: str,
        body: str,
        labels: list[str],
    ) -> str:
        """Create a GitHub issue via MCP.

        Call this tool AFTER you have formatted the issue title, body, and labels.
        The issue will be created in the configured GitHub repository.
        """
        session = ctx.deps.mcp_session

        if session is None:
            print("[issue-writer] Tool called but MCP session not available")
            return "GitHub integration not configured. Issue formatted but not created in GitHub."

        print(f"[issue-writer] Tool called: creating issue '{title}'")
        try:
            result = await session.call_tool(
                "create_issue",
                arguments={
                    "owner": ctx.deps.github_owner,
                    "repo": ctx.deps.github_repo,
                    "title": title,
                    "body": body,
                    "labels": labels,
                },
            )

            # Mark that tool was called and capture result
            ctx.deps.tool_called = True

            # Parse MCP result to extract issue URL/number
            if result.content:
                content = result.content
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].text if hasattr(content[0], "text") else str(content[0])
                    try:
                        issue_data = json.loads(text_content)
                        ctx.deps.issue_url = issue_data.get("html_url")
                        ctx.deps.issue_number = issue_data.get("number")
                    except (json.JSONDecodeError, TypeError):
                        pass

            print("[issue-writer] MCP call successful")
            return f"Successfully created GitHub issue: {title}"

        except Exception as e:
            print(f"[issue-writer] MCP call failed: {e}")
            return f"Failed to create GitHub issue: {e}"

    return agent


def _get_issue_writer_agent() -> Agent[IssueWriterDeps, GitHubIssueResult]:
    """Get or create the Issue Writer Agent (lazy initialization)."""
    global _issue_writer_agent
    if _issue_writer_agent is None:
        _issue_writer_agent = _create_issue_writer_agent()
    return _issue_writer_agent


# =============================================================================
# Public API
# =============================================================================


async def _invoke_issue_writer_agent(
    content_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Invoke Issue Writer Agent - it autonomously decides to create GitHub issue."""
    token = os.getenv("GITHUB_TOKEN")
    repo_env = os.getenv("GITHUB_ISSUES_REPO")

    prompt = f"""Content Type: {content_type}

Structured Data:
{json.dumps(data, indent=2, default=str)}

Format this as a GitHub issue with appropriate title, body (markdown), and labels.
Then create the issue in GitHub using the create_github_issue tool."""

    # Skip entirely if GitHub not configured
    if not token or not repo_env:
        print("[issue-writer] Skipping - GitHub not configured")
        return None

    agent = _get_issue_writer_agent()
    print("[issue-writer] Agent ready")

    try:
        owner, repo = repo_env.split("/")
    except ValueError:
        print(f"[issue-writer] Invalid repo format: '{repo_env}'")
        return {"status": "error", "message": f"Invalid repo format: {repo_env}"}

    headers = {"Authorization": f"Bearer {token}"}

    print(f"[issue-writer] Processing {content_type} for {owner}/{repo}")

    async with (
        httpx.AsyncClient(headers=headers) as http_client,
        streamablehttp_client(GITHUB_MCP_URL, http_client=http_client) as (
            read,
            write,
            _,
        ),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        print("[issue-writer] MCP session ready")

        deps = IssueWriterDeps(
            mcp_session=session,
            github_owner=owner,
            github_repo=repo,
        )

        print("[issue-writer] Running agent...")
        result = await agent.run(prompt, deps=deps)

        print(f"[issue-writer] Agent returned: {result.output.title}")
        if deps.tool_called:
            print(f"[issue-writer] Published to GitHub: {deps.issue_url or 'URL not captured'}")

        return {
            "status": "success",
            "title": result.output.title,
            "body": result.output.body,
            "labels": result.output.labels,
            "repo": f"{owner}/{repo}",
            "github_issue_created": deps.tool_called,
            "github_issue_url": deps.issue_url,
            "github_issue_number": deps.issue_number,
        }


def write_issue(
    content_type: str,
    data: dict[str, Any],
) -> dict[str, Any] | None:
    """Invoke Issue Writer Agent to format and publish an issue.

    The agent autonomously decides whether to publish (e.g., to GitHub).
    """
    try:
        return asyncio.run(_invoke_issue_writer_agent(content_type, data))
    except Exception as e:
        print(f"[issue-writer] Error: {e}")
        return {"status": "error", "message": str(e)}


# Backward compatibility alias
create_github_issue = write_issue


def write_issues_batch(
    content_type: str,
    data_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Invoke Issue Writer Agent for multiple items in parallel (max 5 concurrent)."""
    if not data_list:
        return []

    # Skip entirely if GitHub not configured
    token = os.getenv("GITHUB_TOKEN")
    repo_env = os.getenv("GITHUB_ISSUES_REPO")
    if not token or not repo_env:
        return []

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_ISSUES)

    async def _create_single(data: dict) -> dict[str, Any]:
        async with semaphore:
            result = await _invoke_issue_writer_agent(content_type, data)
            return result if result else {"status": "error", "message": "Not configured"}

    async def _batch_workflow():
        print(f"[issue-writer] Processing {len(data_list)} items (max {MAX_CONCURRENT_ISSUES} concurrent)...")
        tasks = [_create_single(d) for d in data_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append({"status": "error", "message": str(r)})
            else:
                processed.append(r)

        success_count = sum(1 for r in processed if r.get("status") == "success")
        print(f"[issue-writer] Batch complete: {success_count}/{len(data_list)} succeeded")
        return processed

    try:
        return asyncio.run(_batch_workflow())
    except Exception as e:
        print(f"[issue-writer] Batch error: {e}")
        return []


# Backward compatibility alias
create_github_issues_batch = write_issues_batch
