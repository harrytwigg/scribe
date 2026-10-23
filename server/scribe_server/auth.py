"""Who may call the server (docs/spec.md, 3.7).

- With a token configured, every request needs it, from this machine too: harry-box also runs
  agent sessions, so loopback is not a trust boundary. It comes as a bearer token (the bot, API
  clients), or as the session cookie the GUI gets by signing in with it.
- `SCRIBE_TRUST_LOCALHOST=1` exempts requests from this machine that did not come through a proxy
  (no forwarding header: `tailscale serve` and Caddy both connect from loopback): from loopback, or
  from one of the addresses the server itself listens on (opening `http://<this machine's tailnet
  name>:8787` on this machine arrives from its own tailnet address). The exemption covers the GUI
  and every API call, so any process on the machine can use the server without the token. Off by
  default.
- Without a token, nothing is checked; the server then refuses to listen beyond loopback
  (`Settings.check`). That is for development only.
- Any request carrying an `Origin` that is not this server's own is refused, which stops other web
  pages in the browser from driving the API (CSRF), local or not. The one exception is the scribe
  Chrome extension's own origin (`extension_origins`, docs/spec.md 4): a web page cannot send a
  `chrome-extension://` origin, and its requests are authorised like any other (token or cookie).
"""

import hmac
import ipaddress
import secrets
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

COOKIE = "scribe_session"
PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "tailscale-user-login")
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def _normal(address: str) -> str:
    try:
        return str(ipaddress.ip_address(address))
    except ValueError:
        return address


class Auth:
    def __init__(
        self, token: str | None, trust_localhost: bool = False, extension_origins: tuple[str, ...] = (),
        own_addresses: tuple[str, ...] = (),
    ):
        self._token = token
        self._trust_localhost = trust_localhost
        self._own_addresses = frozenset(_normal(a) for a in own_addresses)
        self._extension_origins = frozenset(extension_origins)
        self._sessions: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._token is not None

    def check_token(self, candidate: str) -> bool:
        return self._token is not None and hmac.compare_digest(candidate.encode(), self._token.encode())

    def new_session(self) -> str:
        session = secrets.token_urlsafe(32)
        self._sessions.add(session)
        return session

    def end_session(self, session: str | None) -> None:
        if session:
            self._sessions.discard(session)

    def is_local(self, request: Request) -> bool:
        """From this machine: loopback, or one of the addresses this server listens on (the
        source address of a connection a machine makes to its own tailnet address). A client
        cannot claim either from elsewhere: it is the connection's own source address."""
        if any(header in request.headers for header in PROXY_HEADERS):
            return False
        host = request.client.host if request.client else ""
        try:
            return ipaddress.ip_address(host).is_loopback or _normal(host) in self._own_addresses
        except ValueError:
            return host == "testclient"  # Starlette's TestClient; local in tests

    def authorised(self, request: Request) -> bool:
        if self._token is None:
            return True
        if self._trust_localhost and self.is_local(request):
            return True
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer ") and self.check_token(header[7:].strip()):
            return True
        return request.cookies.get(COOKIE) in self._sessions

    @staticmethod
    def same_origin(request: Request) -> bool:
        origin = request.headers.get("origin")
        if origin is None:
            return True
        return urlsplit(origin).netloc == request.headers.get("host", "")

    def allowed_origin(self, request: Request) -> bool:
        return self.same_origin(request) or request.headers.get("origin") in self._extension_origins

    def guard(self, request: Request) -> None:
        if request.method not in SAFE_METHODS and not self.allowed_origin(request):
            raise HTTPException(403, "cross-origin request refused")
        if not self.authorised(request):
            raise HTTPException(401, "sign in, or send the API token as a bearer token")
