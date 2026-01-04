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
from typing import Optional


async def call_claude_cli(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    timeout: float = 120.0
) -> str:
    """
    Call Claude via CLI instead of API.

    Args:
        prompt: The user message/prompt to send
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens in response (not directly supported by CLI,
                    but kept for API compatibility)
        timeout: Timeout in seconds

    Returns:
        The text response from Claude

    Raises:
        RuntimeError: If CLI execution fails
        asyncio.TimeoutError: If execution times out
    """
    cmd = [
        "claude",
        "-p",  # Print mode (non-interactive)
        "--output-format", "stream-json",
        "--verbose",  # Required when using stream-json with -p
        "--max-turns", "1",  # Single response
    ]

    if system_prompt:
        cmd.extend(["--append-system-prompt", system_prompt])

    cmd.append(prompt)

    # Force CLI auth by removing API key from environment
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout
        )

        if process.returncode != 0:
            stderr_text = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Claude CLI failed with code {process.returncode}: {stderr_text}")

        # Parse streaming JSON output
        try:
            result_text = _parse_stream_json(stdout.decode())
        except UnicodeDecodeError:
            # Handle invalid UTF-8 in output
            result_text = _parse_stream_json(stdout.decode('utf-8', errors='replace'))

        if not result_text:
            raise RuntimeError("No response received from Claude CLI")

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


def _parse_stream_json(output: str) -> str:
    """
    Parse the streaming JSON output from Claude CLI.

    The CLI outputs newline-delimited JSON events like:
    - {"type": "assistant", "message": {"content": [{"type": "text", "text": "..."}]}}
    - {"type": "result", "result": "..."}

    Returns the combined text content.
    """
    result_text = ""

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
                for block in content_blocks:
                    if block.get("type") == "text":
                        result_text += block.get("text", "")

            elif event_type == "content_block_delta":
                # Streaming delta
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    result_text += delta.get("text", "")

        except json.JSONDecodeError:
            # If line isn't JSON, it might be direct text output
            # This handles edge cases with non-standard output
            continue

    return result_text


async def call_claude_cli_with_conversation(
    messages: list,
    system_prompt: Optional[str] = None,
    timeout: float = 120.0
) -> str:
    """
    Call Claude CLI with a conversation history.

    Since the CLI doesn't support multi-turn directly in one call,
    we format the conversation as a single prompt.

    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        system_prompt: Optional system prompt
        timeout: Timeout in seconds

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
            timeout=timeout
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
        timeout=timeout
    )
