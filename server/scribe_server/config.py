"""Settings, all from the environment."""

import os
from dataclasses import dataclass
from pathlib import Path


# The Chrome extension in extension/: its manifest pins this ID with a public `key`, so an unpacked
# install gets the same origin on every machine. SCRIBE_EXTENSION_ORIGINS="" turns it off.
EXTENSION_ORIGIN = "chrome-extension://mcnpmckibkddnnenfondlblnddldonfm"


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    hosts: list[str]
    port: int
    data_dir: Path
    models_dir: Path
    device: str
    keep_audio: bool
    default_model: str
    queue_depth: int
    allowed_hosts: tuple[str, ...] = ()
    extension_origins: tuple[str, ...] = (EXTENSION_ORIGIN,)
    browser_idle_s: float = 300
    diarization: bool = True
    diarization_threads: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("SCRIBE_DATA_DIR", "~/.local/share/scribe")).expanduser()
        return cls(
            hosts=[h.strip() for h in os.environ.get("SCRIBE_HOST", "127.0.0.1").split(",") if h.strip()],
            port=int(os.environ.get("SCRIBE_PORT", "8787")),
            data_dir=data_dir,
            models_dir=Path(os.environ.get("SCRIBE_MODELS_DIR", str(data_dir / "models"))).expanduser(),
            device=os.environ.get("SCRIBE_DEVICE", "cuda"),
            keep_audio=_flag("SCRIBE_KEEP_AUDIO"),
            default_model=os.environ.get("SCRIBE_DEFAULT_MODEL", "parakeet-tdt-0.6b-v2"),
            queue_depth=int(os.environ.get("SCRIBE_QUEUE_DEPTH", "32")),
            allowed_hosts=tuple(h.strip() for h in os.environ.get("SCRIBE_ALLOWED_HOSTS", "").split(",") if h.strip()),
            diarization=_flag("SCRIBE_DIARIZATION", True),
            diarization_threads=int(os.environ.get("SCRIBE_DIARIZATION_THREADS", "8")),
            browser_idle_s=float(os.environ.get("SCRIBE_BROWSER_IDLE_SECONDS", "300")),
            extension_origins=tuple(
                o.strip() for o in os.environ.get("SCRIBE_EXTENSION_ORIGINS", EXTENSION_ORIGIN).split(",") if o.strip()
            ),
        )

    def host_allowlist(self) -> list[str]:
        """Host headers to answer: the listen addresses, this machine's names, tailnet names, and
        SCRIBE_ALLOWED_HOSTS (e.g. a public domain behind a proxy). Anything else is refused,
        which stops a DNS-rebinding page from reaching the server."""
        import socket

        name = socket.gethostname()
        return sorted({*self.hosts, "localhost", "127.0.0.1", "::1", name, f"{name}.local", "*.ts.net", *self.allowed_hosts})

    def check(self) -> None:
        """Refuse to listen on every interface: with no token, who can reach the server is decided
        by the addresses it listens on, so they must be named (docs/spec.md, 3.7)."""
        wide = [h for h in self.hosts if h in {"0.0.0.0", "::", "", "*"}]
        if wide:
            raise SystemExit(
                f"scribe: refusing to listen on {', '.join(repr(h) for h in wide)}: there is no API token, "
                "so name the addresses to listen on (SCRIBE_HOST), e.g. 127.0.0.1 and the tailnet address"
            )
