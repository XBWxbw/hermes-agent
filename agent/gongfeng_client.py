"""OpenAI-compatible client that directly calls the Gongfeng Copilot Gateway.

Calls https://copilot.code.woa.com/server/openclaw/copilot-gateway/v1/chat/completions
using standard OAUTH-TOKEN / DEVICE-ID / X-Username headers.

Key advantages:
  - No intermediate adapter process required
  - Hermes agent loop runs natively (real function calling, not text parsing)
  - Supports all Gongfeng models: claude-sonnet-4.6, hy3-preview-ioa, glm-5.1-ioa, etc.

Auth headers (read from env vars or passed directly):
  OAUTH-TOKEN: <GF_TOKEN>
  DEVICE-ID:   <GF_DEVICE_ID>
  X-Username:  <GF_USERNAME>

Gongfeng Gateway also requires:
  X-Model-Name: human-readable model name e.g. "Claude Sonnet 4.6"

Endpoint: https://copilot.code.woa.com/server/openclaw/copilot-gateway/v1/chat/completions
"""

from __future__ import annotations

import json
import os
import ssl
import time
from types import SimpleNamespace
from typing import Any, Iterator
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

_GONGFENG_BASE_URL = "https://copilot.code.woa.com/server/openclaw/copilot-gateway/v1"
_DEFAULT_TIMEOUT_SECONDS = 300.0

# Marker used in config.yaml base_url to identify this provider
GONGFENG_MARKER_BASE_URL = "https://copilot.code.woa.com"

# Map model id -> human-readable X-Model-Name (required by Gongfeng Gateway)
_GATEWAY_MODEL_NAMES: dict[str, str] = {
    # Claude
    "claude-opus-4.6": "Claude Opus 4.6",
    "claude-opus-4.7": "Claude Opus 4.7",
    "claude-4.5": "Claude Sonnet 4.5",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-opus-4-6": "Claude Opus 4.6",
    # Gemini
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    # GPT
    "gpt-5.4": "GPT 5.4",
    "gpt-5.3-codex": "GPT 5.3 Codex",
    # 国内模型
    "glm-5.1-ioa": "GLM 5.1",
    "deepseek-v4-flash-ioa": "DeepSeek V4 Flash",
    "hy3-preview-ioa": "Hy3 Preview",
    "hy-3-dev": "Hy3 Preview",
    "deepseek-v3-2-volc-ioa": "DeepSeek V3",
    "deepseek-v3-2": "DeepSeek V3",
    "kimi-k2.6-ioa": "Kimi K2",
    "kimi-k2.5-ioa": "Kimi K2",
    "minimax-m2.7-ioa": "MiniMax M2",
    "minimax-m2.5-ioa": "MiniMax M2",
}


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _load_credentials() -> tuple[str, str, str]:
    """Return (oauth_token, device_id, username) from environment variables.

    Primary source: environment variables GF_TOKEN / GF_DEVICE_ID / GF_USERNAME.
    These should be set in ~/.hermes/.env or as system env vars.
    """
    token = (
        os.environ.get("GF_TOKEN")
        or os.environ.get("GONGFENG_OAUTH_TOKEN")
        or ""
    ).strip()
    device_id = (
        os.environ.get("GF_DEVICE_ID")
        or os.environ.get("GONGFENG_DEVICE_ID")
        or ""
    ).strip()
    username = (
        os.environ.get("GF_USERNAME")
        or os.environ.get("GONGFENG_USERNAME")
        or ""
    ).strip()

    if not token:
        raise RuntimeError(
            "Gongfeng credentials missing. Set GF_TOKEN (or GONGFENG_OAUTH_TOKEN) "
            "in ~/.hermes/.env or as system environment variable."
        )
    if not device_id:
        raise RuntimeError(
            "Gongfeng credentials missing. Set GF_DEVICE_ID (or GONGFENG_DEVICE_ID) "
            "in ~/.hermes/.env or as system environment variable."
        )
    if not username:
        raise RuntimeError(
            "Gongfeng credentials missing. Set GF_USERNAME (or GONGFENG_USERNAME) "
            "in ~/.hermes/.env or as system environment variable."
        )
    return token, device_id, username


def _resolve_model_id(model: str) -> str:
    """Strip gongfeng/ prefix and normalize aliases."""
    if model.startswith("gongfeng/"):
        model = model[9:]
    if model.startswith("codebuddy/"):
        model = model[10:]
    # Common dash-dot alias normalization
    aliases = {
        "claude-sonnet-4-5": "claude-4.5",
        "claude-sonnet-4-6": "claude-sonnet-4.6",
        "claude-opus-4-6": "claude-opus-4.6",
        "gpt-5-3-codex": "gpt-5.3-codex",
        "gpt-5-4": "gpt-5.4",
        "deepseek-v3-2": "deepseek-v3-2-volc-ioa",
        "hy-3-dev": "hy3-preview-ioa",
    }
    return aliases.get(model, model)


def _make_auth_headers(
    oauth_token: str,
    device_id: str,
    username: str,
    model_id: str,
    stream: bool,
) -> dict[str, str]:
    x_model_name = _GATEWAY_MODEL_NAMES.get(model_id, model_id)
    return {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "OAUTH-TOKEN": oauth_token,
        "DEVICE-ID": device_id,
        "X-Username": username,
        "X-Model-Name": x_model_name,
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_post_stream(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> Iterator[str]:
    """POST to Gongfeng Gateway and yield raw lines."""
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        ctx = ssl.create_default_context()
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                yield line
    except ssl.SSLError:
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
            f"Gongfeng Gateway POST {url} failed [{exc.code}]: {body_text}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Gongfeng Gateway POST {url} connection error: {exc.reason}"
        ) from exc


def _parse_openai_sse_stream(lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse standard OpenAI SSE stream and yield delta dicts.

    Handles both 'data: {...}' (with space) and 'data:{...}' (without space)
    formats, as Gongfeng Gateway omits the space after 'data:'.
    """
    for line in lines:
        if line.startswith("data: "):
            raw = line[6:].strip()
        elif line.startswith("data:"):
            raw = line[5:].strip()
        else:
            continue
        if raw == "[DONE]":
            break
        if not raw:
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------
# Streaming adapter — lets Hermes iterate SSE chunks in real-time
# ---------------------------------------------------------------------------

class _GongfengStreamAdapter:
    """Wraps a live SSE stream so Hermes can do ``for chunk in stream``.

    Hermes's ``_call_chat_completions`` expects:
        for chunk in stream:
            chunk.choices[0].delta.content  (text delta or "")
            chunk.choices[0].delta.tool_calls  (list or None)
            chunk.choices[0].finish_reason  (None or "stop"/"tool_calls")
            chunk.usage  (None or usage object on final chunk)
            chunk.model  (str)

    We yield one SimpleNamespace chunk per SSE event, following the standard
    OpenAI streaming format so Hermes's existing accumulator logic works.
    """

    def __init__(
        self,
        url: str,
        request_body: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
        raw_model: str,
    ):
        self._url = url
        self._request_body = request_body
        self._headers = headers
        self._timeout = timeout
        self._raw_model = raw_model
        # Stub attributes that Hermes reads before iterating
        self.response = None  # no httpx Response object

    def __iter__(self) -> Iterator[Any]:
        line_iter = _http_post_stream(
            self._url, self._request_body, self._headers, self._timeout
        )
        for raw_chunk in _parse_openai_sse_stream(line_iter):
            choices = raw_chunk.get("choices") or []
            usage_dict = raw_chunk.get("usage")
            usage_ns = None
            if isinstance(usage_dict, dict):
                usage_ns = SimpleNamespace(
                    prompt_tokens=usage_dict.get("prompt_tokens", 0),
                    completion_tokens=usage_dict.get("completion_tokens", 0),
                    total_tokens=usage_dict.get("total_tokens", 0),
                    prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                )

            if not choices:
                # Usage-only chunk (some providers send this at the end)
                if usage_ns is not None:
                    yield SimpleNamespace(
                        choices=[],
                        usage=usage_ns,
                        model=raw_chunk.get("model", self._raw_model),
                    )
                continue

            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta_dict = choice.get("delta") or {}
                # Build tool_calls delta list
                tc_list = None
                raw_tc = delta_dict.get("tool_calls")
                if isinstance(raw_tc, list) and raw_tc:
                    tc_list = []
                    for tc in raw_tc:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function") or {}
                        tc_list.append(SimpleNamespace(
                            index=tc.get("index", 0),
                            id=tc.get("id") or None,
                            type=tc.get("type", "function"),
                            function=SimpleNamespace(
                                name=fn.get("name") or "",
                                arguments=fn.get("arguments") or "",
                            ),
                        ))

                delta_ns = SimpleNamespace(
                    role=delta_dict.get("role") or "assistant",
                    content=delta_dict.get("content") or "",
                    tool_calls=tc_list,
                    reasoning=delta_dict.get("reasoning") or None,
                    reasoning_content=delta_dict.get("reasoning_content") or None,
                )
                chunk_ns = SimpleNamespace(
                    choices=[SimpleNamespace(
                        index=choice.get("index", 0),
                        delta=delta_ns,
                        finish_reason=choice.get("finish_reason"),
                    )],
                    usage=usage_ns,
                    model=raw_chunk.get("model", self._raw_model),
                    id=raw_chunk.get("id", ""),
                )
                yield chunk_ns


# ---------------------------------------------------------------------------
# OpenAI-client facade
# ---------------------------------------------------------------------------

class _GongfengChatCompletions:
    def __init__(self, client: "GongfengClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _GongfengChatNamespace:
    def __init__(self, client: "GongfengClient"):
        self.completions = _GongfengChatCompletions(client)


class GongfengClient:
    """OpenAI-compatible client that directly calls the Gongfeng Copilot Gateway.

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
        self.api_key = api_key or "gongfeng"
        self.base_url = _GONGFENG_BASE_URL
        self._base_url = _GONGFENG_BASE_URL
        self._default_headers = dict(default_headers or {})
        self._custom_headers: dict[str, str] = {}  # agent_init.py reads this for default_headers
        self.chat = _GongfengChatNamespace(self)
        self.is_closed = False
        # Cache credentials (reload once per instance)
        self._cached_creds: tuple[str, str, str] | None = None

    def _get_creds(self) -> tuple[str, str, str]:
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

        raw_model = model or "hy3-preview-ioa"
        model_id = _resolve_model_id(raw_model)

        oauth_token, device_id, username = self._get_creds()
        # Gongfeng Gateway supports streaming; use streaming internally and aggregate
        headers = _make_auth_headers(oauth_token, device_id, username, model_id, stream=True)

        # Build request body — standard OpenAI format
        # Gongfeng Gateway uses the human-readable model name via X-Model-Name header,
        # but also needs model field in body (use the normalized id or gateway name)
        gw_model_name = _GATEWAY_MODEL_NAMES.get(model_id, model_id)
        request_body: dict[str, Any] = {
            "model": gw_model_name,
            "messages": messages or [],
            "stream": True,
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

        # --- Streaming path: return adapter so caller can iterate chunks ---
        if stream:
            # Re-build headers with stream=True (already set above, but be explicit)
            stream_headers = _make_auth_headers(oauth_token, device_id, username, model_id, stream=True)
            return _GongfengStreamAdapter(
                url=url,
                request_body=request_body,
                headers=stream_headers,
                timeout=effective_timeout,
                raw_model=raw_model,
            )

        # --- Non-streaming path: switch to non-stream Accept header ---
        headers = _make_auth_headers(oauth_token, device_id, username, model_id, stream=False)
        request_body["stream"] = False

        # Collect streaming response
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        prompt_tokens = 0
        completion_tokens = 0

        line_iter = _http_post_stream(url, request_body, headers, effective_timeout)
        for chunk in _parse_openai_sse_stream(line_iter):
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

                # Reasoning content
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
        choice_ns = SimpleNamespace(
            message=assistant_message,
            finish_reason=finish_reason,
        )
        return SimpleNamespace(
            choices=[choice_ns],
            usage=usage_ns,
            model=raw_model,
        )
