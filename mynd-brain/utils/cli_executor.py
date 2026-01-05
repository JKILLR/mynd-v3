"""
Claude CLI Executor
====================
Replaces direct Anthropic API calls with Claude Code CLI.
Uses Max subscription instead of per-token billing.

Usage:
    from utils.cli_executor import call_claude_cli

    response = await call_claude_cli(
        prompt="Hello, how are you?",
        system_prompt="You are a helpful assistant.",
        max_tokens=4096
    )
"""

import asyncio
import json
import os
import shutil
import tempfile
from typing import Optional


# Check if claude+ is available (handles root user restrictions)
CLAUDE_PLUS_AVAILABLE = shutil.which("claude+") is not None


async def call_claude_cli(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    timeout: float = 300.0,  # 5 minutes for tool-heavy tasks
    enable_tools: bool = True
) -> str:
    """
    Call Claude via CLI instead of API.

    Args:
        prompt: The user message/prompt to send
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens in response (not directly supported by CLI,
                    but kept for API compatibility)
        timeout: Timeout in seconds
        enable_tools: Whether to enable full tool access (web search, file ops, bash, etc.)

    Returns:
        The text response from Claude

    Raises:
        RuntimeError: If CLI execution fails
        asyncio.TimeoutError: If execution times out
    """
    # Build the claude command
    claude_args = [
        "claude",
        "-p",  # Print mode (non-interactive)
        "--output-format", "stream-json",
        "--verbose",  # Required when using stream-json with -p
    ]

    if enable_tools:
        # Enable all tools: WebSearch, WebFetch, Read, Write, Edit, Bash, Glob, Grep, Task
        claude_args.extend(["--tools", "default"])
        # Full permission bypass for complete tool access
        claude_args.append("--dangerously-skip-permissions")
        # Allow access to additional directories
        claude_args.extend(["--add-dir", "/workspace"])
        claude_args.extend(["--add-dir", "/tmp"])
    else:
        # No tools, single turn response
        claude_args.extend(["--max-turns", "1"])

    # Combine system prompt with user prompt
    if system_prompt:
        full_prompt = f"""<system_context>
{system_prompt}
</system_context>

{prompt}"""
    else:
        full_prompt = prompt

    # Force CLI auth by removing API key from environment
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    # Check if running as root - if so, use claude+ wrapper
    is_root = os.geteuid() == 0

    if is_root and CLAUDE_PLUS_AVAILABLE:
        # Use claude+ which handles root restrictions via temporary user
        # Replace 'claude' with 'claude+' in the command
        claude_args[0] = "claude+"

        # Write prompt to temp file for claude+
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(full_prompt)
            prompt_file = f.name
        os.chmod(prompt_file, 0o644)

        # claude+ needs the prompt via stdin redirection in a shell
        claude_cmd_str = " ".join(claude_args) + f" < {prompt_file}"
        cmd = ["bash", "-c", claude_cmd_str]
        stdin_input = None
    elif is_root:
        # Fallback: try runuser if claude+ not available
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(full_prompt)
            prompt_file = f.name
        os.chmod(prompt_file, 0o644)

        claude_cmd_str = " ".join(claude_args) + f" < {prompt_file}"
        cmd = ["runuser", "-u", "claude-temp", "--", "bash", "-c", claude_cmd_str]
        stdin_input = None
    else:
        # Running as non-root, can use stdin directly
        claude_args.append("-")  # Read from stdin
        cmd = claude_args
        stdin_input = full_prompt.encode('utf-8')
        prompt_file = None

    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin_input else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=stdin_input),
            timeout=timeout
        )

        if process.returncode != 0:
            stderr_text = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Claude CLI failed with code {process.returncode}: {stderr_text}")

        # Parse streaming JSON output
        stdout_text = ""
        try:
            stdout_text = stdout.decode()
            result_text = _parse_stream_json(stdout_text)
        except UnicodeDecodeError:
            # Handle invalid UTF-8 in output
            stdout_text = stdout.decode('utf-8', errors='replace')
            result_text = _parse_stream_json(stdout_text)

        if not result_text:
            # Debug: log what we actually received
            stderr_text = stderr.decode() if stderr else ""
            print(f"⚠️ CLI returned empty result. stdout length: {len(stdout_text)}, stderr: {stderr_text[:500]}")
            print(f"⚠️ Raw stdout (first 1000 chars): {stdout_text[:1000]}")
            raise RuntimeError(f"No response received from Claude CLI. stdout_len={len(stdout_text)}, stderr={stderr_text[:200]}")

        return result_text

    except asyncio.TimeoutError:
        # Kill the subprocess on timeout to prevent zombie processes
        if process is not None:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
        raise asyncio.TimeoutError(f"Claude CLI timed out after {timeout}s")

    except Exception:
        # Ensure subprocess is cleaned up on any exception
        if process is not None and process.returncode is None:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass  # Process already terminated
        raise

    finally:
        # Clean up temp file if we created one
        if prompt_file and os.path.exists(prompt_file):
            try:
                os.unlink(prompt_file)
            except OSError:
                pass


def _parse_stream_json(output: str) -> str:
    """
    Parse the streaming JSON output from Claude CLI.

    The CLI outputs newline-delimited JSON events like:
    - {"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}
    - {"type": "result", "result": "..."}
    - {"type": "tool_use", ...} - when Claude uses tools
    - {"type": "tool_result", ...} - tool execution results

    Returns the combined text content (final response after any tool use).
    """
    result_text = ""
    tool_uses = []
    last_assistant_text = ""

    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue

        try:
            event = json.loads(line)
            event_type = event.get("type")

            if event_type == "result":
                # Final result - use this preferentially
                result = event.get("result", "")
                if result:
                    return result

            elif event_type == "assistant":
                # Extract from message content blocks
                message = event.get("message", {})
                content_blocks = message.get("content", [])
                current_text = ""
                for block in content_blocks:
                    if block.get("type") == "text":
                        current_text += block.get("text", "")
                    elif block.get("type") == "tool_use":
                        # Track tool usage for context
                        tool_uses.append(block.get("name", "unknown"))
                if current_text:
                    last_assistant_text = current_text
                    result_text = current_text  # Keep updating with latest

            elif event_type == "content_block_delta":
                # Streaming delta
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    result_text += delta.get("text", "")

            elif event_type == "tool_use":
                # Tool being invoked
                tool_name = event.get("name", "unknown")
                tool_uses.append(tool_name)
                print(f"🔧 Axel using tool: {tool_name}")

            elif event_type == "tool_result":
                # Tool completed - log but don't add to response
                pass

        except json.JSONDecodeError:
            # If line isn't JSON, it might be direct text output
            # This handles edge cases with non-standard output
            continue

    # If we tracked tool uses, log summary
    if tool_uses:
        print(f"🔧 Tools used in this response: {', '.join(tool_uses)}")

    return result_text


async def call_claude_cli_with_conversation(
    messages: list,
    system_prompt: Optional[str] = None,
    timeout: float = 300.0,  # 5 minutes for tool-heavy tasks
    enable_tools: bool = True
) -> str:
    """
    Call Claude CLI with a conversation history.

    Since the CLI doesn't support multi-turn directly in one call,
    we format the conversation as a single prompt.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        system_prompt: Optional system prompt
        timeout: Timeout in seconds
        enable_tools: Whether to enable full tool access

    Returns:
        The text response from Claude
    """
    # Format conversation as a prompt
    # The last message should be from the user
    if not messages:
        raise ValueError("Messages list cannot be empty")

    # If there's only one user message, send it directly
    if len(messages) == 1 and messages[0].get("role") == "user":
        return await call_claude_cli(
            prompt=messages[0].get("content", ""),
            system_prompt=system_prompt,
            timeout=timeout,
            enable_tools=enable_tools
        )

    # Format multi-turn as a single prompt with clear delimiters
    formatted_parts = []
    for msg in messages[:-1]:  # All but last message
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            formatted_parts.append(f"User: {content}")
        else:
            formatted_parts.append(f"Assistant: {content}")

    # Add the final user message
    last_msg = messages[-1]
    if last_msg.get("role") != "user":
        raise ValueError("Last message must be from user")

    formatted_parts.append(f"User: {last_msg.get('content', '')}")

    # Add instruction for continuation
    conversation_prompt = "\n\n".join(formatted_parts)

    # Prepend conversation context to system prompt
    context_prefix = "Continue this conversation. Respond as the Assistant:\n\n"
    full_prompt = context_prefix + conversation_prompt

    return await call_claude_cli(
        prompt=full_prompt,
        system_prompt=system_prompt,
        timeout=timeout,
        enable_tools=enable_tools
    )
