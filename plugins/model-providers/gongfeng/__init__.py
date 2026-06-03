"""Gongfeng Copilot Gateway provider profile.

Calls https://copilot.code.woa.com/server/openclaw/copilot-gateway/v1/chat/completions
using OAUTH-TOKEN / DEVICE-ID / X-Username headers.
Auth credentials are read from env vars: GF_TOKEN, GF_DEVICE_ID, GF_USERNAME.
The actual HTTP transport is handled by GongfengClient (agent/gongfeng_client.py).
"""

from providers import register_provider
from providers.base import ProviderProfile


class GongfengProfile(ProviderProfile):
    """Gongfeng Copilot Gateway - direct API, no intermediate process."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing not supported via REST; return static list."""
        return [
            "hy3-preview-ioa",
            "hy-3-dev",
            "glm-5.1-ioa",
            "deepseek-v3-2-volc-ioa",
            "deepseek-v4-flash-ioa",
            "claude-sonnet-4.6",
            "claude-opus-4.6",
            "claude-opus-4.7",
            "claude-4.5",
            "gpt-5.4",
            "gemini-3.1-pro",
        ]


gongfeng = GongfengProfile(
    name="gongfeng",
    aliases=("gongfeng-copilot", "gf-copilot", "woa-copilot"),
    api_mode="chat_completions",
    env_vars=("GF_TOKEN", "GF_DEVICE_ID", "GF_USERNAME"),
    base_url="https://copilot.code.woa.com/server/openclaw/copilot-gateway/v1",
    auth_type="external_process",  # auth handled internally by GongfengClient
    supports_health_check=False,   # skip /models probe — not a standard endpoint
)

register_provider(gongfeng)
