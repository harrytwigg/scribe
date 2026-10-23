"""A Fetcher that serves canned pages, feeds and audio, so link resolution runs without a network."""

import json
from email.message import Message

from scribe_server import podcast
from scribe_server.podcast import FetchError, Fetcher


class FakeReply(podcast.Reply):
    def __init__(self, url: str, body: bytes, content_type: str, final_url: str | None = None, length: int | None = None):
        self.url = final_url or url
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Content-Length"] = str(len(body) if length is None else length)
        self._body = body
        self._at = 0

    def read(self, n: int = -1) -> bytes:
        end = len(self._body) if n < 0 else self._at + n
        chunk = self._body[self._at : end]
        self._at += len(chunk)
        return chunk

    def __exit__(self, *exc):
        pass


class FakeFetcher(Fetcher):
    """`pages` maps a URL to (body, content type) or (body, content type, final URL after a redirect)."""

    def __init__(self, pages: dict):
        super().__init__()
        self.pages = pages
        self.requested: list[str] = []

    def request(self, url, headers=None):
        self.requested.append(url)
        if url not in self.pages:
            raise FetchError(f"{url} answered HTTP 404 Not Found")
        body, content_type, *final = self.pages[url]
        return FakeReply(url, body, content_type, final[0] if final else None)


def spotify_embed(kind: str, spotify_id: str, title: str, show: str, released: str, duration_ms: int) -> tuple[bytes, str]:
    data = {"props": {"pageProps": {"state": {"data": {"entity": {
        "type": "episode", "title": title, "name": title, "subtitle": show, "id": spotify_id,
        "releaseDate": {"isoString": released}, "duration": duration_ms,
    }}}}}}
    page = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script></html>'
    return page.encode(), "text/html"


def itunes(results: list[dict]) -> tuple[bytes, str]:
    return json.dumps({"resultCount": len(results), "results": results}).encode(), "application/json"


def feed(show: str, items: list[tuple[str, str, str, str]], author: str = "Some Network") -> tuple[bytes, str]:
    """items: (title, pubDate, duration, audio URL)."""
    body = "".join(
        f"<item><title>{t}</title><pubDate>{d}</pubDate><itunes:duration>{dur}</itunes:duration>"
        f'<enclosure url="{u}" length="1" type="audio/mpeg"/></item>'
        for t, d, dur, u in items
    )
    xml = (
        '<?xml version="1.0"?><rss xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" version="2.0"><channel>'
        f"<title>{show}</title><itunes:author>{author}</itunes:author>{body}</channel></rss>"
    )
    return xml.encode(), "application/rss+xml"
