"""Find the public audio of a podcast episode from a link (docs/cli.md).

Spotify's own audio is DRM-protected and not downloadable, so, like podcastmp3.com, a Spotify link
is turned into the episode's public copy: read the episode's metadata off Spotify's pages, find
the show's public RSS feed (Apple's iTunes search), match the episode in the feed (title, date and
duration) and take the `<enclosure>` audio. Apple Podcasts links, RSS feeds and audio URLs resolve
the same way. A show that publishes no public feed (a Spotify exclusive) cannot be fetched: that is
a `NotFound` with a message saying so.

Standard library only, so the CLI starts instantly and runs without the server's GPU stack.
"""

import difflib
import email.utils
import html
import ipaddress
import json
import re
import socket
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

USER_AGENT = "Mozilla/5.0 (compatible; scribe/1.0; +https://github.com/harrytwigg/scribe)"
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".m4b", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".mp4", ".webm", ".wma"}
PAGE_LIMIT = 8 * 1024 * 1024
FEED_LIMIT = 64 * 1024 * 1024
AUDIO_LIMIT = 2 * 1024 * 1024 * 1024
ITUNES = "https://itunes.apple.com"


class PodcastError(Exception):
    """Base of every error here; the message is meant to be shown to a person or an agent."""


class Unsupported(PodcastError):
    """The link is not one this understands."""


class NotFound(PodcastError):
    """No public audio for the episode: no feed, or the feed has no matching episode."""


class FetchError(PodcastError):
    """The network or a remote server failed. `status` is the HTTP status when a server answered."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


# -- fetching -----------------------------------------------------------------------------------


class Reply:
    """An open HTTP response: `url` after redirects, `headers`, and `read`."""

    def __init__(self, response, url: str):
        self._response = response
        self.url = url
        self.headers = response.headers

    @property
    def content_type(self) -> str:
        return (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()

    @property
    def length(self) -> int | None:
        value = self.headers.get("Content-Length")
        return int(value) if value and value.isdigit() else None

    def read(self, n: int = -1) -> bytes:
        return self._response.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._response.close()


def check_public(url: str) -> None:
    """Refuse a URL that is not http(s) to a public address (a server fetching for a client)."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise Unsupported(f"not an http(s) link: {url}")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except OSError as error:
        raise FetchError(f"cannot resolve {parts.hostname}: {error}") from error
    for info in infos:
        if not ipaddress.ip_address(info[4][0].split("%")[0]).is_global:
            raise Unsupported(f"{parts.hostname} is not a public address; refusing to fetch it")


class _Redirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, public_only: bool):
        self.public_only = public_only

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urljoin(req.full_url, newurl)
        if self.public_only:
            check_public(target)
        elif urllib.parse.urlsplit(target).scheme not in ("http", "https"):
            raise Unsupported(f"redirect to a non-http link: {target}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Fetcher:
    """HTTP GET with a browser-ish User-Agent. `public_only` is for a server fetching on a
    client's behalf: it refuses loopback, private and link-local addresses, redirects included
    (it resolves the name before connecting, so a DNS-rebinding host could still slip through;
    the server is reachable only by a holder of the API token)."""

    def __init__(self, public_only: bool = False, timeout: float = 30):
        self.public_only = public_only
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_Redirects(public_only))

    def request(self, url: str, headers: dict[str, str] | None = None) -> Reply:
        if self.public_only:
            check_public(url)
        elif urllib.parse.urlsplit(url).scheme not in ("http", "https"):
            raise Unsupported(f"not an http(s) link: {url}")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*", **(headers or {})})
        try:
            response = self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            raise FetchError(f"{url} answered HTTP {error.code} {error.reason}", error.code) from error
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise FetchError(f"cannot reach {url}: {getattr(error, 'reason', error)}") from error
        return Reply(response, response.geturl())

    def get(self, url: str, limit: int = PAGE_LIMIT, headers: dict[str, str] | None = None) -> tuple[bytes, Reply]:
        with self.request(url, headers) as reply:
            try:
                body = reply.read(limit + 1)
            except OSError as error:
                raise FetchError(f"reading {url} failed: {error}") from error
        if len(body) > limit:
            raise FetchError(f"{url} is larger than {limit // (1024 * 1024)} MB")
        return body, reply

    def get_json(self, url: str):
        body, _ = self.get(url, headers={"Accept": "application/json"})
        try:
            return json.loads(body)
        except ValueError as error:
            raise FetchError(f"{url} did not return JSON") from error


# -- what an episode is --------------------------------------------------------------------------


@dataclass
class Episode:
    title: str
    audio_url: str
    show: str | None = None
    publisher: str | None = None
    published: datetime | None = None
    duration_s: float | None = None
    page_url: str | None = None
    feed_url: str | None = None
    source: str = "url"  # spotify, apple, rss, url
    source_url: str | None = None
    # How sure the match is, for a Spotify link: `title` (0 to 1), `overall`, and whether the
    # `date` and `duration` agreed (None: the feed did not say). A weak match is worth a look.
    match: dict | None = None

    def to_json(self) -> dict:
        data = asdict(self)
        data["published"] = self.published.isoformat() if self.published else None
        return data


@dataclass
class Wanted:
    """What the link said the episode is (Spotify's metadata), to match in a feed."""

    title: str
    show: str
    published: datetime | None = None
    duration_s: float | None = None


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", clean(text).lower().replace("&", " and "))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"^(the|a|an) ", "", text)


def similar(a: str, b: str) -> float:
    a, b = normalise(a), normalise(b)
    if not a or not b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    # "Ep 12: The thing" in a feed against "The thing" on Spotify, or the reverse.
    if min(len(a), len(b)) >= 8 and (a in b or b in a):
        ratio = max(ratio, 0.92)
    return ratio


def parse_duration(text: str | None) -> float | None:
    """iTunes durations: `3600`, `59:30` or `1:02:03`."""
    if not text:
        return None
    parts = text.strip().split(":")
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60 + float(part)
    except ValueError:
        return None
    return seconds if len(parts) <= 3 and seconds > 0 else None


def parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    text = text.strip()
    try:
        date = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            date = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    return date if date.tzinfo else date.replace(tzinfo=timezone.utc)


def numbers(text: str) -> list[str]:
    return re.findall(r"\d+", normalise(text))


@dataclass
class Match:
    title: float
    overall: float
    date: bool | None  # agreed within a day or two; None: one side has no date
    duration: bool | None  # agreed within a few percent; None: one side has no length
    numerals_differ: bool

    @property
    def agreed(self) -> bool:
        return self.date is True or self.duration is True

    @property
    def contradicted(self) -> bool:
        return self.date is False and self.duration is not True or self.duration is False and self.date is not True

    def accepted(self) -> bool:
        """Is this the wanted episode? An identical title is not enough alone: sibling episodes
        share titles ("Weekly Roundup"), differ only in a number ("Part 1", "Episode 12"), or
        a repeat of the title may be another year's. So a date or length that both sides give must
        not disagree, a title that differs in a number needs the date or length to agree, and
        a title that is not identical needs something besides the title to agree."""
        if self.contradicted or (self.numerals_differ and not self.agreed):
            return False
        if self.title >= 0.97:
            return True
        return self.agreed and self.overall >= 0.8 and self.title >= 0.6

    def to_json(self) -> dict:
        return {"title": round(self.title, 3), "overall": round(self.overall, 3), "date": self.date, "duration": self.duration}


def score(want: Wanted, have: Episode) -> Match:
    """How well a feed's episode matches what the link said. The title counts most; the date (within
    a day or two: time zones, late uploads) and the duration (within a few percent: Spotify and the
    feed insert different ads) confirm or contradict it."""
    title = similar(want.title, have.title)
    signals = [(0.6, title)]
    date = duration = None
    if want.published and have.published:
        days = abs((want.published - have.published).total_seconds()) / 86400
        signals.append((0.2, 1.0 if days <= 1.6 else 0.5 if days <= 3.5 else 0.0))
        date = days <= 1.6 if days <= 1.6 or days > 3.5 else None  # in between: neither
    if want.duration_s and have.duration_s:
        gap = abs(want.duration_s - have.duration_s)
        relative = gap / max(want.duration_s, have.duration_s)
        signals.append((0.2, 1.0 if gap <= 60 or relative <= 0.03 else 0.5 if relative <= 0.1 else 0.0))
        duration = True if gap <= 60 or relative <= 0.03 else None if relative <= 0.1 else False
    # Both titles carry numbers and they are not the same numbers ("Part 1" / "Part 2"). A number
    # on only one side is a feed's "Ep 12:" prefix.
    a, b = numbers(want.title), numbers(have.title)
    return Match(
        title, sum(w * v for w, v in signals) / sum(w for w, _ in signals), date, duration, bool(a and b and a != b)
    )


def best_match(want: Wanted, episodes: list[Episode]) -> tuple[Episode | None, list[tuple[float, Episode]]]:
    """The feed's episode that is the wanted one, or None; with the top candidates by score."""
    ranked = sorted(((score(want, e), e) for e in episodes), key=lambda pair: pair[0].overall, reverse=True)
    top = [(m.overall, e) for m, e in ranked[:3]]
    for match, episode in sorted(ranked, key=lambda pair: (pair[0].accepted(), pair[0].overall), reverse=True)[:1]:
        if match.accepted():
            episode.match = match.to_json()
            return episode, top
    return None, top


# -- feeds -------------------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if _local(child.tag) == name and (child.text or "").strip():
            return clean(child.text)
    return None


def looks_like_feed(body: bytes) -> bool:
    head = body[:2048].lstrip().lower()
    return head.startswith(b"<?xml") or head.startswith(b"<rss") or b"<rss" in head or b"<feed" in head


def parse_feed(body: bytes, feed_url: str | None = None) -> tuple[str, list[Episode]]:
    """The show's title and its episodes with audio, newest first."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as error:
        raise FetchError(f"not a podcast feed: {error}") from error
    channel = next((c for c in root if _local(c.tag) == "channel"), None)
    if channel is None:
        raise FetchError("not a podcast feed: no <channel>")
    show = _text(channel, "title") or ""
    publisher = _text(channel, "author") or _text(channel, "owner")
    episodes = []
    for item in channel:
        if _local(item.tag) != "item":
            continue
        enclosure = next((c for c in item if _local(c.tag) == "enclosure" and c.get("url")), None)
        if enclosure is None:
            continue
        episodes.append(
            Episode(
                title=_text(item, "title") or "(untitled)",
                audio_url=enclosure.get("url").strip(),
                show=show,
                publisher=publisher,
                published=parse_date(_text(item, "pubDate")),
                duration_s=parse_duration(_text(item, "duration")),
                page_url=_text(item, "link"),
                feed_url=feed_url,
                source="rss",
            )
        )
    episodes.sort(key=lambda e: e.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return show, episodes


def load_feed(fetcher: Fetcher, url: str) -> tuple[str, list[Episode]]:
    body, reply = fetcher.get(url, limit=FEED_LIMIT, headers={"Accept": "application/rss+xml, application/xml, text/xml, */*"})
    return parse_feed(body, reply.url)


# -- Apple's directory -------------------------------------------------------------------------


def itunes_feeds(fetcher: Fetcher, show: str) -> list[dict]:
    """Directory entries (name, artist, feed URL) for shows called roughly `show`, best name first."""
    query = urllib.parse.urlencode({"media": "podcast", "entity": "podcast", "limit": 15, "term": show})
    results = fetcher.get_json(f"{ITUNES}/search?{query}").get("results", [])
    found = [
        {"name": r.get("collectionName", ""), "artist": r.get("artistName"), "feed": r["feedUrl"], "name_score": similar(show, r.get("collectionName", ""))}
        for r in results
        if r.get("feedUrl")
    ]
    return sorted(found, key=lambda r: r["name_score"], reverse=True)


def itunes_episode_search(fetcher: Fetcher, want: Wanted) -> Episode | None:
    """A last resort for a show the name search misses: search for the episode itself."""
    query = urllib.parse.urlencode({"media": "podcast", "entity": "podcastEpisode", "limit": 25, "term": f"{want.title} {want.show}"})
    episodes = []
    for r in fetcher.get_json(f"{ITUNES}/search?{query}").get("results", []):
        if r.get("episodeUrl") and similar(want.show, r.get("collectionName", "")) >= 0.8:
            episodes.append(
                Episode(
                    title=r.get("trackName", ""), audio_url=r["episodeUrl"], show=r.get("collectionName"),
                    publisher=r.get("artistName"), published=parse_date(r.get("releaseDate")),
                    duration_s=(r.get("trackTimeMillis") or 0) / 1000 or None, feed_url=r.get("feedUrl"), source="spotify",
                )
            )
    return best_match(want, episodes)[0]


# -- Spotify -----------------------------------------------------------------------------------

SPOTIFY = re.compile(
    r"^(?:https?://)?(?:open|play)\.spotify\.com/(?:intl-[a-z-]+/)?(?:embed/)?(?P<kind>episode|show)/(?P<id>[A-Za-z0-9]{22})"
)
SPOTIFY_URI = re.compile(r"^spotify:(?P<kind>episode|show):(?P<id>[A-Za-z0-9]{22})$")
SPOTIFY_SHORT = re.compile(r"^(?:https?://)?spotify\.link/", re.I)


def parse_spotify(link: str) -> tuple[str, str] | None:
    """('episode' | 'show', id) for a Spotify link or URI, else None."""
    match = SPOTIFY.match(link.strip()) or SPOTIFY_URI.match(link.strip())
    return (match["kind"], match["id"]) if match else None


def _meta(page: str, key: str) -> str | None:
    for tag in re.findall(r"<meta\b[^>]*>", page):
        if re.search(rf'(?:property|name)="{re.escape(key)}"', tag):
            content = re.search(r'content="([^"]*)"', tag)
            if content:
                return clean(content.group(1))
    return None


def spotify_meta(fetcher: Fetcher, kind: str, spotify_id: str) -> Wanted:
    """An episode's title, show, release date and length from Spotify's public pages. For a show,
    its latest episode (Spotify's embed shows that), which also names the show."""
    try:
        embed = fetcher.get(f"https://open.spotify.com/embed/{kind}/{spotify_id}")[0].decode("utf-8", "replace")
    except FetchError as error:
        if error.status in (400, 404):
            raise NotFound(f"Spotify has no public {kind} with id {spotify_id}") from error
        raise
    data = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', embed, re.S)
    if data:
        try:
            entity = json.loads(data.group(1))["props"]["pageProps"]["state"]["data"]["entity"]
            released = (entity.get("releaseDate") or {}).get("isoString")
            return Wanted(
                title=clean(entity.get("title") or entity.get("name") or ""),
                show=clean(entity.get("subtitle") or ""),
                published=parse_date(released),
                duration_s=(entity.get("duration") or 0) / 1000 or None,
            )
        except (KeyError, ValueError, TypeError):
            pass
    # The embed's layout changed: the page's own tags say less but still enough.
    page = fetcher.get(f"https://open.spotify.com/{kind}/{spotify_id}")[0].decode("utf-8", "replace")
    title = _meta(page, "og:title")
    if not title:
        raise FetchError(f"Spotify did not describe {kind} {spotify_id} (is the link right, and public?)")
    if kind == "show":
        return Wanted(title="", show=title)
    about = _meta(page, "description") or ""
    show = re.search(r"episode from (.+?) on Spotify", about)
    duration = _meta(page, "music:duration")
    return Wanted(
        title=title, show=clean(show.group(1)) if show else (_meta(page, "og:description") or "").split(" · ")[0],
        published=parse_date(_meta(page, "music:release_date")), duration_s=float(duration) if duration else None,
    )


EXCLUSIVE = (
    "Spotify's own audio is DRM-protected and cannot be downloaded; scribe fetches the public copy from the "
    "show's RSS feed, and {what}. A Spotify exclusive has no public feed (and a feed may have dropped an "
    "old episode), so it cannot be transcribed from the link. Pass the RSS or mp3 URL, or an audio file, if "
    "you have one."
)


def resolve_spotify(fetcher: Fetcher, kind: str, spotify_id: str, nth: int = 1) -> Episode:
    want = spotify_meta(fetcher, kind, spotify_id)
    if not want.show:
        raise FetchError(f"Spotify did not name the show of {kind} {spotify_id}")
    found: list[tuple[float, Episode, list[Episode]]] = []
    tried, closest = [], []
    for entry in itunes_feeds(fetcher, want.show)[:6]:
        if entry["name_score"] < 0.8:
            continue
        tried.append(f"{entry['name']} ({entry['feed']})")
        try:
            _, episodes = load_feed(fetcher, entry["feed"])
        except PodcastError:
            continue
        match, top = best_match(want, episodes)
        if not match and top:
            closest.append(f"{top[0][1].title!r} ({top[0][1].published.date() if top[0][1].published else 'undated'})")
        if match:
            found.append((match.match["overall"] + entry["name_score"] / 10, match, episodes))
        elif kind == "show" and entry["name_score"] >= 0.95 and episodes:
            # A show link names the show, not an episode we must find: a feed with its name will do.
            found.append((entry["name_score"] / 10, episodes[0], episodes))
    if not found:
        if kind == "episode":
            fallback = itunes_episode_search(fetcher, want)
            if fallback:
                fallback.source_url = f"https://open.spotify.com/episode/{spotify_id}"
                return fallback
        what = (
            f"no public feed for a show called {want.show!r} has the episode {want.title!r}"
            if tried else f"no public feed was found for a show called {want.show!r}"
        )
        if tried:
            what += f" (looked in: {'; '.join(tried)}"
            what += f"; closest titles there: {'; '.join(closest)})" if closest else ")"
        raise NotFound(EXCLUSIVE.format(what=what))
    _, match, episodes = max(found, key=lambda f: f[0])
    chosen = match if kind == "episode" else pick_nth(episodes, nth)
    chosen.source = "spotify"
    chosen.source_url = f"https://open.spotify.com/{kind}/{spotify_id}"
    return chosen


def pick_nth(episodes: list[Episode], nth: int) -> Episode:
    if not episodes:
        raise NotFound("the feed has no episodes with audio")
    if not 1 <= nth <= len(episodes):
        raise NotFound(f"episode {nth} requested but the feed has {len(episodes)} (1 is the newest)")
    return episodes[nth - 1]


# -- Apple Podcasts ----------------------------------------------------------------------------

APPLE = re.compile(r"^(?:https?://)?podcasts\.apple\.com/.*?/id(?P<show>\d+)", re.I)


def resolve_apple(fetcher: Fetcher, link: str, nth: int = 1) -> Episode:
    match = APPLE.match(link.strip())
    if not match:
        raise Unsupported(f"not an Apple Podcasts show or episode link: {link}")
    episode_id = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query).get("i", [None])[0]
    query = urllib.parse.urlencode({"id": match["show"], "entity": "podcastEpisode", "limit": 300})
    results = fetcher.get_json(f"{ITUNES}/lookup?{query}").get("results", [])
    show = next((r for r in results if r.get("wrapperType") == "track" and r.get("kind") == "podcast"), None)
    if not show:
        raise NotFound(f"Apple Podcasts has no show with id {match['show']}")
    entries = {
        str(r.get("trackId")): Episode(
            title=r.get("trackName", ""), audio_url=r["episodeUrl"], show=show.get("collectionName"),
            publisher=show.get("artistName"), published=parse_date(r.get("releaseDate")),
            duration_s=(r.get("trackTimeMillis") or 0) / 1000 or None, page_url=r.get("trackViewUrl"),
            feed_url=show.get("feedUrl"), source="apple", source_url=link,
        )
        for r in results
        if r.get("kind") == "podcast-episode" and r.get("episodeUrl")
    }
    if episode_id:
        if episode_id in entries:
            return entries[episode_id]
        # An older episode than the directory lookup lists: the feed has all of them, but not
        # Apple's episode id, so say what is possible.
        raise NotFound(
            f"episode {episode_id} is not among the {len(entries)} latest episodes Apple lists for "
            f"{show.get('collectionName')!r}; use its RSS feed ({show.get('feedUrl')}) with --episode N"
        )
    if show.get("feedUrl"):
        try:
            return _as(pick_nth(load_feed(fetcher, show["feedUrl"])[1], nth), "apple", link)
        except PodcastError:
            pass
    return pick_nth(list(entries.values()), nth)


def _as(episode: Episode, source: str, source_url: str) -> Episode:
    episode.source, episode.source_url = source, source_url
    return episode


# -- any link ----------------------------------------------------------------------------------


def is_audio_path(url: str) -> bool:
    return Path(urllib.parse.urlsplit(url).path).suffix.lower() in AUDIO_EXTENSIONS


def title_from_url(url: str) -> str:
    return urllib.parse.unquote(Path(urllib.parse.urlsplit(url).path).stem) or urllib.parse.urlsplit(url).hostname or "audio"


def feed_links(page: str, base: str) -> list[str]:
    links = []
    for tag in re.findall(r"<link\b[^>]*>", page):
        if re.search(r'type="application/(?:rss|atom)\+xml"', tag, re.I):
            href = re.search(r'href="([^"]+)"', tag)
            if href:
                links.append(urllib.parse.urljoin(base, html.unescape(href.group(1))))
    return links


def resolve(fetcher: Fetcher, link: str, nth: int = 1) -> Episode:
    """The episode a link names. `nth` picks from a show's or feed's episodes, 1 being the newest.
    Does not download the audio."""
    link = link.strip()
    spotify = parse_spotify(link)
    if not spotify and SPOTIFY_SHORT.match(link):
        # spotify.link redirects to open.spotify.com/<kind>/<id>?...
        with fetcher.request(link if "://" in link else f"https://{link}") as reply:
            spotify = parse_spotify(reply.url)
    if spotify:
        return resolve_spotify(fetcher, spotify[0], spotify[1], nth)
    if APPLE.match(link):
        return resolve_apple(fetcher, link, nth)
    if is_audio_path(link):
        return Episode(title=title_from_url(link), audio_url=link, source="url", source_url=link)
    with fetcher.request(link) as reply:
        kind = reply.content_type
        if kind.startswith(("audio/", "video/")) or kind == "application/ogg":
            return Episode(title=title_from_url(reply.url), audio_url=reply.url, source="url", source_url=link)
        body = reply.read(FEED_LIMIT + 1)
    if len(body) > FEED_LIMIT:
        raise FetchError(f"{link} is larger than {FEED_LIMIT // (1024 * 1024)} MB and is not audio")
    if looks_like_feed(body):
        return _as(pick_nth(parse_feed(body, reply.url)[1], nth), "rss", link)
    # A show's web page: it usually points at its feed.
    for feed in feed_links(body[:PAGE_LIMIT].decode("utf-8", "replace"), reply.url):
        try:
            return _as(pick_nth(load_feed(fetcher, feed)[1], nth), "rss", link)
        except PodcastError:
            continue
    raise Unsupported(
        f"{link} is not a Spotify or Apple Podcasts link, an RSS feed, an audio URL or a page that links a feed"
    )


def list_episodes(fetcher: Fetcher, link: str, limit: int = 20) -> list[Episode]:
    """The newest episodes of the show or feed a link names (for choosing `--episode N`)."""
    first = resolve(fetcher, link, 1)
    if first.feed_url:
        return load_feed(fetcher, first.feed_url)[1][:limit]
    return [first]


# -- downloading -------------------------------------------------------------------------------


def safe_name(text: str, fallback: str = "episode") -> str:
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", plain).strip("-._")
    return name[:80] or fallback


CONTENT_EXTENSIONS = {
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a", "audio/x-m4a": ".m4a", "audio/aac": ".aac",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/flac": ".flac", "audio/ogg": ".ogg", "audio/opus": ".opus",
    "video/mp4": ".mp4", "video/webm": ".webm",
}


def download(
    fetcher: Fetcher, url: str, directory: Path, name: str,
    progress: Callable[[int, int | None], None] | None = None, limit: int = AUDIO_LIMIT,
) -> Path:
    """Stream the audio at `url` into `directory` as `<name>.<ext>` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    with fetcher.request(url) as reply:
        if reply.content_type.startswith(("text/", "application/json")):
            raise FetchError(f"{url} is a web page, not audio ({reply.content_type})")
        extension = Path(urllib.parse.urlsplit(reply.url).path).suffix.lower()
        if extension not in AUDIO_EXTENSIONS:
            extension = CONTENT_EXTENSIONS.get(reply.content_type, ".mp3")
        total = reply.length
        if total and total > limit:
            raise FetchError(f"the audio is {total // (1024 * 1024)} MB, over the {limit // (1024 * 1024)} MB limit")
        target = directory / f"{safe_name(name)}{extension}"
        partial = target.with_suffix(target.suffix + ".part")
        done = 0
        try:
            with partial.open("wb") as out:
                while chunk := reply.read(1024 * 256):
                    done += len(chunk)
                    if done > limit:
                        raise FetchError(f"the audio is over the {limit // (1024 * 1024)} MB limit")
                    out.write(chunk)
                    if progress:
                        progress(done, total)
        except OSError as error:
            partial.unlink(missing_ok=True)
            raise FetchError(f"downloading {url} failed: {error}") from error
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    if total and done < total:
        partial.unlink(missing_ok=True)
        raise FetchError(f"{url} ended after {done} of {total} bytes")
    partial.replace(target)
    return target
