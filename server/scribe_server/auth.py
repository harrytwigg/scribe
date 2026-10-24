"""What the server checks on a request (docs/spec.md, 3.7).

There is no API token or sign-in: the server is meant to be reachable only from this machine and
its tailnet (it listens on the addresses it is told to, never on every interface: `Settings.check`),
and anyone who can reach it can use it.

What it does check is where a request comes from *in a browser*:

- Any request that changes something and carries an `Origin` that is not this server's own is
  refused, which stops other web pages from driving the API (CSRF), local or not. The one exception
  is the scribe Chrome extension's own origin (`extension_origins`, docs/spec.md 4): a web page
  cannot send a `chrome-extension://` origin.
- The `Host` header is checked against an allowlist by the app (`Settings.host_allowlist`), which
  stops DNS-rebinding pages from reaching the server.
"""

from urllib.parse import urlsplit

from fastapi import HTTPException, Request

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class Auth:
    def __init__(self, extension_origins: tuple[str, ...] = ()):
        self._extension_origins = frozenset(extension_origins)

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
