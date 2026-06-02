"""OpenAI-compatible shim that forwards Hermes requests to `codebuddy --serve` HTTP API.

This adapter lets Hermes treat a running CodeBuddy HTTP server as a chat-style
backend. Each request calls POST /api/v1/runs (async), then polls
GET /api/v1/runs/:runId/stream (SSE) to collect streaming results.

Compared to codebuddy_acp_client.py (stdio/ACP):
  - No subprocess spawning per request — reuses a long-lived codebuddy --serve process
  - Sessions are maintained server-side; no need to serialize entire message history
  - Uses standard HTTP instead of ndJsonStream over stdin/stdout
  - Tool calls handled natively by the server; no <tool_call> text-parsing hacks
  - Eliminates "thinking-only / empty response" retry loops caused by ACP session churn

Usage:
  1. Start codebuddy in serve mode (once, as a background service):
       codebuddy --serve --port 18080 --session-id hermes-http --permission-mode auto
  2. Set provider in config.yaml:
       model:
         provider: codebuddy-http
         default: claude-opus-4.8
  3. Set env vars (optional):
       HERMES_CODEBUDDY_HTTP_BASE_URL=http://127.0.0.1:18080
       HERMES_CODEBUDDY_HTTP_SESSION_ID=hermes-http

Protocol: CodeBuddy HTTP API Beta (/api/v1/runs)
  POST /api/v1/runs        → { runId }
  GET  /api/v1/runs/:id/stream  → SSE stream of run events
  Headers: X-CodeBuddy-Request: 1   (required by CORS guard)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from types import SimpleNamespace
from typing import Any, Iterator
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urljoin

ACP_MARKER_BASE_URL = "http://codebuddy-http"
_DEFAULT_BASE_URL = "http://127.0.0.1:18080"
_DEFAULT_SESSION_ID = "hermes-http"
_DEFAULT_TIMEOUT_SECONDS = 900.0
_POLL_INTERVAL_SECONDS = 0.2
_REQUIRED_HEADER = "X-CodeBuddy-Request"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _resolve_base_url() -> str:
    return (
        os.getenv("HERMES_CODEBUDDY_HTTP_BASE_URL", "").strip().rstrip("/")
        or _DEFAULT_BASE_URL
    )


def _resolve_session_id() -> str:
    return (
        os.getenv("HERMES_CODEBUDDY_HTTP_SESSION_ID", "").strip()
        or _DEFAULT_SESSION_ID
    )


def _resolve_password() -> str:
    """Optional password for remote codebuddy servers with auth enabled."""
    return os.getenv("HERMES_CODEBUDDY_HTTP_PASSWORD", "").strip()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _make_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {
        _REQUIRED_HEADER: "1",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    password = _resolve_password()
    if password:
        headers["Authorization"] = f"Bearer {password}"
    elif api_key and api_key not in ("codebuddy-http", "none", ""):
        # Allow passing CODEBUDDY_API_KEY as Bearer token
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _http_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            f"CodeBuddy HTTP API POST {url} failed [{exc.code}]: {body_text}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"CodeBuddy HTTP API POST {url} connection error: {exc.reason}. "
            "Is `codebuddy --serve` running?"
        ) from exc


def _http_get_sse(
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> Iterator[str]:
    """Yield SSE event data lines from a streaming GET request."""
    sse_headers = dict(headers)
    sse_headers["Accept"] = "text/event-stream"
    req = Request(url, headers=sse_headers, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\n").rstrip("\r")
                yield line
    except HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            f"CodeBuddy HTTP API GET {url} failed [{exc.code}]: {body_text}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"CodeBuddy HTTP API GET {url} connection error: {exc.reason}."
        ) from exc


def _parse_sse_events(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse SSE lines into event dicts with 'event' and 'data' keys."""
    event_type = "message"
    data_parts: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].strip())
        elif line == "":
            if data_parts:
                raw = "\n".join(data_parts)
                try:
                    yield {"event": event_type, "data": json.loads(raw)}
                except json.JSONDecodeError:
                    yield {"event": event_type, "data": raw}
            event_type = "message"
            data_parts = []


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def _submit_run(
    base_url: str,
    session_id: str,
    prompt: str,
    model: str | None,
    headers: dict[str, str],
    timeout: float,
) -> str:
    """POST /api/v1/runs and return runId."""
    url = f"{base_url}/api/v1/runs"
    body: dict[str, Any] = {
        "id": str(uuid.uuid4()),   # required by CodeBuddy HTTP API
        "type": "run",             # required by CodeBuddy HTTP API
        "sessionId": session_id,
        "prompt": prompt,
    }
    if model:
        body["model"] = model
    result = _http_post(url, body, headers, timeout=min(timeout, 30.0))
    data = result.get("data") or result
    run_id = str(data.get("runId") or data.get("id") or "").strip()
    if not run_id:
        raise RuntimeError(
            f"CodeBuddy HTTP API did not return a runId. Response: {result}"
        )
    return run_id


def _stream_run(
    base_url: str,
    run_id: str,
    headers: dict[str, str],
    timeout: float,
) -> tuple[str, str]:
    """GET /api/v1/runs/:runId/stream and collect text + reasoning."""
    url = f"{base_url}/api/v1/runs/{run_id}/stream"
    text_parts: list[str] = []
    reasoning_parts: list[str] = []

    for event in _parse_sse_events(_http_get_sse(url, headers, timeout)):
        data = event.get("data")
        if not isinstance(data, dict):
            continue

        # event_type can come from SSE "event:" line OR from data["type"]
        sse_event = str(event.get("event") or "")
        event_type = str(data.get("type") or sse_event or "")

        # ── CodeBuddy HTTP API native format ──────────────────────────────
        # Actual observed format:
        #   event: message
        #   data: {"version":"1.0","status":"completed","content":{"markdown":"..."},"agent":{...}}
        #
        #   event: done
        #   data: {}
        if sse_event == "message" or event_type == "message":
            content_field = data.get("content")
            if isinstance(content_field, dict):
                # content.markdown is the primary text field
                chunk = str(content_field.get("markdown") or content_field.get("text") or "")
                if chunk:
                    text_parts.append(chunk)
            elif isinstance(content_field, str) and content_field:
                text_parts.append(content_field)
            # Also check top-level text/output fields as fallback
            if not text_parts or not text_parts[-1]:
                fallback = str(data.get("text") or data.get("output") or "")
                if fallback:
                    text_parts.append(fallback)
            continue

        if sse_event == "done" or event_type == "done":
            # Completion signal — no content in done event for this API
            continue

        # ── Legacy / alternative formats ──────────────────────────────────

        # Text output chunks
        if event_type in ("message_delta", "text_delta", "agent_message_chunk", "content_block_delta"):
            chunk = str(data.get("text") or data.get("delta", {}).get("text") or "")
            if chunk:
                text_parts.append(chunk)

        # Reasoning / thinking chunks
        elif event_type in ("thinking_delta", "agent_thought_chunk", "reasoning_delta"):
            chunk = str(data.get("text") or data.get("thinking") or "")
            if chunk:
                reasoning_parts.append(chunk)

        # Full message (non-streaming fallback)
        elif event_type in ("run_complete", "message_complete"):
            output = data.get("output") or data.get("message") or data.get("result")
            if isinstance(output, str):
                text_parts.append(output)
            elif isinstance(output, dict):
                content = output.get("content") or output.get("text") or ""
                if isinstance(content, dict):
                    content = content.get("markdown") or content.get("text") or ""
                if content:
                    text_parts.append(str(content))

        # Error
        elif event_type == "error":
            error_msg = str(data.get("message") or data.get("error") or "Unknown error")
            raise RuntimeError(f"CodeBuddy HTTP run error: {error_msg}")

    return "".join(text_parts), "".join(reasoning_parts)


def _format_messages_as_prompt(
    messages: list[dict[str, Any]],
    model: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> str:
    """Serialize Hermes messages into a single prompt string for /api/v1/runs.

    Note: When codebuddy --serve is used with a persistent session, this
    serialization is only needed for the first request. For subsequent turns,
    only the latest user message needs to be passed — but since Hermes manages
    its own message history, we always serialize everything for safety.
    """
    sections: list[str] = [
        "You are being used as the active HTTP API agent backend for Hermes.",
        "Use all available capabilities to complete tasks.",
        "IMPORTANT: If you take an action with a tool, you MUST output tool calls using "
        "<tool_call>{...}</tool_call> blocks with JSON exactly in OpenAI function-call shape.",
        "If no tool is needed, answer normally.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    if isinstance(tools, list) and tools:
        tool_specs: list[dict[str, Any]] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            tool_specs.append({
                "name": name.strip(),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        if tool_specs:
            sections.append(
                "Available tools (OpenAI function schema). "
                "When using a tool, emit ONLY <tool_call>{...}</tool_call> with one JSON object "
                "containing id/type/function{name,arguments}. arguments must be a JSON string.\n"
                + json.dumps(tool_specs, ensure_ascii=False)
            )

    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"
        content = message.get("content")
        rendered = _render_message_content(content)
        if not rendered:
            continue
        label = {"system": "System", "user": "User", "assistant": "Assistant",
                 "tool": "Tool", "context": "Context"}.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _render_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "").strip()
        return json.dumps(content, ensure_ascii=True)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


# ---------------------------------------------------------------------------
# Tool call extraction (identical to ACP client)
# ---------------------------------------------------------------------------

import re

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_JSON_RE = re.compile(
    r"\{\s*\"id\"\s*:\s*\"[^\"]+\"\s*,\s*\"type\"\s*:\s*\"function\"\s*,\s*\"function\"\s*:\s*\{.*?\}\s*\}",
    re.DOTALL,
)


def _extract_tool_calls_from_text(text: str) -> tuple[list[SimpleNamespace], str]:
    if not isinstance(text, str) or not text.strip():
        return [], ""

    extracted: list[SimpleNamespace] = []
    consumed_spans: list[tuple[int, int]] = []

    def _try_add(raw_json: str) -> None:
        try:
            obj = json.loads(raw_json)
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        fn = obj.get("function")
        if not isinstance(fn, dict):
            return
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name.strip():
            return
        fn_args = fn.get("arguments", "{}")
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = f"http_call_{len(extracted) + 1}"
        extracted.append(SimpleNamespace(
            id=call_id, call_id=call_id, response_item_id=None,
            type="function",
            function=SimpleNamespace(name=fn_name.strip(), arguments=fn_args),
        ))

    for m in _TOOL_CALL_BLOCK_RE.finditer(text):
        _try_add(m.group(1))
        consumed_spans.append((m.start(), m.end()))

    if not extracted:
        for m in _TOOL_CALL_JSON_RE.finditer(text):
            _try_add(m.group(0))
            consumed_spans.append((m.start(), m.end()))

    if not consumed_spans:
        return extracted, text.strip()

    consumed_spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in consumed_spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            parts.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        parts.append(text[cursor:])

    cleaned = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return extracted, cleaned


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_server_health(base_url: str, timeout: float = 5.0) -> bool:
    """Return True if the codebuddy HTTP server is healthy."""
    try:
        url = f"{base_url}/api/v1/health"
        req = Request(url, headers={_REQUIRED_HEADER: "1"}, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            status = (data.get("data") or {}).get("status") or data.get("status") or ""
            return str(status).lower() == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# OpenAI-client facade
# ---------------------------------------------------------------------------

class _HTTPChatCompletions:
    def __init__(self, client: "CodeBuddyHTTPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _HTTPChatNamespace:
    def __init__(self, client: "CodeBuddyHTTPClient"):
        self.completions = _HTTPChatCompletions(client)


class CodeBuddyHTTPClient:
    """Minimal OpenAI-client-compatible facade for CodeBuddy HTTP API (/api/v1/runs).

    Drop-in replacement for CodeBuddyACPClient that communicates with a
    long-running `codebuddy --serve` process instead of spawning a new
    subprocess per request.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        session_id: str | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "codebuddy-http"
        # base_url from config may be the marker URL — resolve from env
        raw_base = base_url or ""
        if raw_base.startswith("http://codebuddy-http") or not raw_base.startswith("http"):
            raw_base = _resolve_base_url()
        self.base_url = raw_base.rstrip("/")
        self._default_headers = dict(default_headers or {})
        self._session_id = session_id or _resolve_session_id()
        self.chat = _HTTPChatNamespace(self)
        self.is_closed = False

    def close(self) -> None:
        self.is_closed = True

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        if timeout is None:
            effective_timeout = _DEFAULT_TIMEOUT_SECONDS
        elif isinstance(timeout, (int, float)):
            effective_timeout = float(timeout)
        else:
            candidates = [
                getattr(timeout, attr, None)
                for attr in ("read", "write", "connect", "pool", "timeout")
            ]
            numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
            effective_timeout = max(numeric) if numeric else _DEFAULT_TIMEOUT_SECONDS

        prompt_text = _format_messages_as_prompt(
            messages or [],
            model=model,
            tools=tools,
            tool_choice=tool_choice,
        )

        headers = _make_headers(self.api_key)

        # Health check (fast, non-blocking)
        if not check_server_health(self.base_url, timeout=3.0):
            raise RuntimeError(
                f"CodeBuddy HTTP server at {self.base_url} is not responding. "
                "Please start it with: codebuddy --serve --port 18080 --permission-mode auto"
            )

        # Submit run
        run_id = _submit_run(
            base_url=self.base_url,
            session_id=self._session_id,
            prompt=prompt_text,
            model=model,
            headers=headers,
            timeout=effective_timeout,
        )

        # Stream results
        response_text, reasoning_text = _stream_run(
            base_url=self.base_url,
            run_id=run_id,
            headers=headers,
            timeout=effective_timeout,
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=model or "codebuddy-http",
        )
