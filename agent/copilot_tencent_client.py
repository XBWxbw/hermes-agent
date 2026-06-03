"""OpenAI-compatible client that directly calls copilot.tencent.com/v2.

This provider bypasses `codebuddy --serve` and calls the underlying model
endpoint directly, using the same SSO accessToken that codebuddy stores locally.

Key advantages over codebuddy-http:
  - No intermediate codebuddy process required
  - Hermes agent loop runs natively (real function calling, not text parsing)
  - Standard OpenAI streaming SSE format
  - Supports all models: glm-5.1-ioa, deepseek-v3-2-volc-ioa, claude-*, etc.

Auth:
  Token is read from CodeBuddy's local credential store:
    %LOCALAPPDATA%/CodeBuddyExtension/Data/Public/auth/Tencent-Cloud.coding-copilot.info
  Required headers:
    Authorization: Bearer <accessToken>
    X-Domain: tencent.sso.copilot.tencent.com
    X-Enterprise-Id: <enterpriseId>
    X-Tenant-Id: <enterpriseId>

Endpoint: https://copilot.tencent.com/v2/chat/completions
  - Only streaming mode is supported (stream: true required)
  - Standard OpenAI SSE format
"""

from __future__ import annotations

import json
import os
import pathlib
import ssl
import time
from types import SimpleNamespace
from typing import Any, Iterator
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

_COPILOT_BASE_URL = "https://copilot.tencent.com/v2"
_IOA_DOMAIN = "tencent.sso.copilot.tencent.com"
_DEFAULT_TIMEOUT_SECONDS = 300.0

# Marker used in config.yaml base_url to identify this provider
COPILOT_TENCENT_MARKER_BASE_URL = "https://copilot.tencent.com"


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------

def _get_cred_path() -> pathlib.Path:
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        p = pathlib.Path(local_app_data) / "CodeBuddyExtension" / "Data" / "Public" / "auth" / "Tencent-Cloud.coding-copilot.info"
        if p.exists():
            return p
    # Fallback: search common locations
    candidates = [
        pathlib.Path.home() / ".codebuddy" / "auth" / "Tencent-Cloud.coding-copilot.info",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Cannot find CodeBuddy credentials file. "
        "Expected at: %LOCALAPPDATA%/CodeBuddyExtension/Data/Public/auth/Tencent-Cloud.coding-copilot.info"
    )


def _load_credentials() -> tuple[str, str]:
    """Load (accessToken, enterpriseId) from CodeBuddy credential store."""
    cred_path = _get_cred_path()
    with open(cred_path, encoding="utf-8") as f:
        data = json.load(f)
    auth = data.get("auth") or {}
    account = data.get("account") or {}
    access_token = str(auth.get("accessToken") or "").strip()
    enterprise_id = str(account.get("enterpriseId") or "").strip()
    if not access_token:
        raise RuntimeError("CodeBuddy credential file missing accessToken")
    if not enterprise_id:
        raise RuntimeError("CodeBuddy credential file missing enterpriseId")
    return access_token, enterprise_id


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _make_auth_headers(access_token: str, enterprise_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {access_token}",
        "X-Domain": _IOA_DOMAIN,
        "X-Enterprise-Id": enterprise_id,
        "X-Tenant-Id": enterprise_id,
    }


def _http_post_stream(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Iterator[str]:
    """POST to copilot endpoint and yield SSE lines."""
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        # Use unverified SSL context if needed (corporate proxy)
        ctx = ssl.create_default_context()
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                yield line
    except ssl.SSLError:
        # Fallback: unverified
        ctx = ssl._create_unverified_context()
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                yield line
    except HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8")
        except Exception:
            pass
        raise RuntimeError(
            f"copilot.tencent.com POST {url} failed [{exc.code}]: {body_text}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"copilot.tencent.com POST {url} connection error: {exc.reason}"
        ) from exc


def _parse_openai_sse_stream(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse standard OpenAI SSE stream and yield delta dicts."""
    for line in lines:
        if not line.startswith("data: "):
            continue
        raw = line[6:].strip()
        if raw == "[DONE]":
            break
        if not raw:
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------
# OpenAI-client facade
# ---------------------------------------------------------------------------

class _CopilotChatCompletions:
    def __init__(self, client: "CopilotTencentClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _CopilotChatNamespace:
    def __init__(self, client: "CopilotTencentClient"):
        self.completions = _CopilotChatCompletions(client)


class CopilotTencentClient:
    """OpenAI-compatible client that directly calls copilot.tencent.com/v2.

    Drop-in replacement for OpenAI client. Hermes treats this exactly like
    any other OpenAI-compatible provider — function calling, streaming,
    message history all work natively.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        **_: Any,
    ):
        self.api_key = api_key or "copilot-tencent"
        self.base_url = _COPILOT_BASE_URL
        self._base_url = _COPILOT_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._custom_headers: dict[str, str] = {}  # agent_init.py reads this for default_headers
        self.chat = _CopilotChatNamespace(self)
        self.is_closed = False
        # Cache credentials (reload if needed)
        self._cached_creds: tuple[str, str] | None = None

    def _get_creds(self) -> tuple[str, str]:
        if self._cached_creds is None:
            self._cached_creds = _load_credentials()
        return self._cached_creds

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
        stream: bool = False,
        **extra: Any,
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

        access_token, enterprise_id = self._get_creds()
        headers = _make_auth_headers(access_token, enterprise_id)

        # Build request body — standard OpenAI format
        request_body: dict[str, Any] = {
            "model": model or "glm-5.1-ioa",
            "messages": messages or [],
            "stream": True,  # copilot.tencent.com only supports streaming
        }

        # Pass extra params
        for key in ("max_tokens", "temperature", "top_p", "stop", "presence_penalty", "frequency_penalty"):
            if key in extra:
                request_body[key] = extra[key]

        # Tool / function calling
        if tools:
            request_body["tools"] = tools
        if tool_choice is not None:
            request_body["tool_choice"] = tool_choice

        url = f"{self._base_url}/chat/completions"

        # Collect streaming response
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}  # index -> partial tool call
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0

        line_iter = _http_post_stream(url, request_body, headers, effective_timeout)
        for chunk in _parse_openai_sse_stream(line_iter):
            # Extract usage if present
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)

            choices = chunk.get("choices") or []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue

                # Text content
                content = delta.get("content")
                if isinstance(content, str) and content:
                    text_parts.append(content)

                # Reasoning content (some models)
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)

                # Tool calls (incremental)
                delta_tool_calls = delta.get("tool_calls") or []
                for tc in delta_tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc.get("id", f"call_{idx}"),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    existing = tool_calls_map[idx]
                    if tc.get("id"):
                        existing["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        existing["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        existing["function"]["arguments"] += fn["arguments"]

        # Build tool_calls list
        assembled_tool_calls: list[SimpleNamespace] = []
        for idx in sorted(tool_calls_map.keys()):
            tc = tool_calls_map[idx]
            fn = tc["function"]
            assembled_tool_calls.append(SimpleNamespace(
                id=tc["id"],
                call_id=tc["id"],
                response_item_id=None,
                type="function",
                function=SimpleNamespace(
                    name=fn["name"],
                    arguments=fn["arguments"],
                ),
            ))

        if assembled_tool_calls:
            finish_reason = "tool_calls"

        usage_ns = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        response_text = "".join(text_parts)
        response_reasoning = "".join(reasoning_parts)
        assistant_message = SimpleNamespace(
            content=response_text,
            tool_calls=assembled_tool_calls,
            reasoning=response_reasoning or None,
            reasoning_content=response_reasoning or None,
            reasoning_details=None,
        )
        choice = SimpleNamespace(
            message=assistant_message,
            finish_reason=finish_reason,
        )
        return SimpleNamespace(
            choices=[choice],
            usage=usage_ns,
            model=model or "glm-5.1-ioa",
        )
