"""Settings, all from the environment."""

import ipaddress
import json
import os
from dataclasses import dataclass
from pathlib import Path


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _token() -> str | None:
    if token := os.environ.get("SCRIBE_API_TOKEN"):
        return token
    # A JSON file of secrets, read at start and never copied anywhere (docs/spec.md, 3.6).
    if path := os.environ.get("SCRIBE_SECRETS_FILE"):
        return json.loads(Path(path).read_text()).get("SCRIBE_API_TOKEN")
    return None


@dataclass(frozen=True)
class Settings:
    hosts: list[str]
    port: int
    data_dir: Path
    models_dir: Path
    api_token: str | None
    device: str
    keep_audio: bool
    default_model: str
    queue_depth: int
    trust_localhost: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("SCRIBE_DATA_DIR", "~/.local/share/scribe")).expanduser()
        return cls(
            hosts=[h.strip() for h in os.environ.get("SCRIBE_HOST", "127.0.0.1").split(",") if h.strip()],
            port=int(os.environ.get("SCRIBE_PORT", "8787")),
            data_dir=data_dir,
            models_dir=Path(os.environ.get("SCRIBE_MODELS_DIR", str(data_dir / "models"))).expanduser(),
            api_token=_token(),
            device=os.environ.get("SCRIBE_DEVICE", "cuda"),
            keep_audio=_flag("SCRIBE_KEEP_AUDIO"),
            default_model=os.environ.get("SCRIBE_DEFAULT_MODEL", "parakeet-tdt-0.6b-v2"),
            queue_depth=int(os.environ.get("SCRIBE_QUEUE_DEPTH", "32")),
            trust_localhost=_flag("SCRIBE_TRUST_LOCALHOST"),
        )

    def check(self) -> None:
        """Refuse to listen beyond loopback without a token (docs/spec.md, 3.7)."""
        exposed = [h for h in self.hosts if not _is_loopback(h)]
        if exposed and not self.api_token:
            raise SystemExit(
                f"scribe: refusing to listen on {', '.join(exposed)} without SCRIBE_API_TOKEN "
                "(or SCRIBE_SECRETS_FILE holding it)"
            )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
