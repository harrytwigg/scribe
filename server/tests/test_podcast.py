import urllib.parse
from datetime import datetime, timezone

import pytest

from scribe_server import podcast
from scribe_server.podcast import FetchError, NotFound, Unsupported

from .podcast_fakes import FakeFetcher, feed, itunes, spotify_embed

EP = "7makk4oTQel546B0PZlDM5"
SHOW = "2bzjLBEWRldARaf1IytFFS"
FEED_A = "https://feeds.example.com/the-show.xml"
FEED_B = "https://feeds.example.com/the-show-live.xml"


def search(term: str, **extra) -> str:
    query = urllib.parse.urlencode({"media": "podcast", "entity": "podcast", "limit": 15, "term": term, **extra})
    return f"https://itunes.apple.com/search?{query}"


def spotify_pages(**over) -> dict:
    """An episode 'Ep 12: The Big Idea' of 'The Show', released 2026-03-04, 40 minutes long."""
    return {
        f"https://open.spotify.com/embed/episode/{EP}": spotify_embed(
            "episode", EP, over.get("title", "The Big Idea"), "The Show", "2026-03-04T09:00:00Z", 2400_000
        ),
        search("The Show"): itunes([
            {"collectionName": "The Show Live", "artistName": "Other", "feedUrl": FEED_B},
            {"collectionName": "The Show", "artistName": "Some Network", "feedUrl": FEED_A},
        ]),
        FEED_A: feed("The Show", [
            ("Ep 13: Something Else", "Fri, 11 Mar 2026 09:00:00 GMT", "00:30:00", "https://cdn.example.com/13.mp3"),
            ("Ep 12: The Big Idea", "Wed, 04 Mar 2026 08:30:00 GMT", "40:10", "https://cdn.example.com/12.mp3"),
            ("Ep 11: Older", "Wed, 25 Feb 2026 08:30:00 GMT", "1:00:00", "https://cdn.example.com/11.mp3"),
        ]),
        FEED_B: feed("The Show Live", [
            ("The Big Idea", "Wed, 04 Mar 2026 08:30:00 GMT", "10:00", "https://cdn.example.com/live-12.mp3"),
        ]),
    }


def test_parse_spotify_links():
    cases = {
        f"https://open.spotify.com/episode/{EP}?si=abc": ("episode", EP),
        f"https://open.spotify.com/intl-fr/episode/{EP}": ("episode", EP),
        f"open.spotify.com/embed/episode/{EP}": ("episode", EP),
        f"https://open.spotify.com/show/{SHOW}": ("show", SHOW),
        f"spotify:episode:{EP}": ("episode", EP),
    }
    for link, want in cases.items():
        assert podcast.parse_spotify(link) == want
    assert podcast.parse_spotify("https://open.spotify.com/track/" + EP) is None
    assert podcast.parse_spotify("https://example.com/episode/" + EP) is None


def test_spotify_episode_is_found_in_its_shows_feed():
    episode = podcast.resolve(FakeFetcher(spotify_pages()), f"https://open.spotify.com/episode/{EP}")
    assert episode.audio_url == "https://cdn.example.com/12.mp3"
    assert (episode.title, episode.show, episode.source) == ("Ep 12: The Big Idea", "The Show", "spotify")
    assert episode.duration_s == pytest.approx(2410)
    assert episode.feed_url == FEED_A
    assert episode.source_url.endswith(EP)


def test_the_look_alike_show_with_the_same_title_loses_on_duration():
    # FEED_B's "The Big Idea" has the same title and date but is 10 minutes long, and the show's name is
    # the weaker match: the real show's episode must win.
    pages = spotify_pages()
    pages[search("The Show")] = itunes([
        {"collectionName": "The Show", "artistName": "Some Network", "feedUrl": FEED_B},
        {"collectionName": "The Show", "artistName": "Some Network", "feedUrl": FEED_A},
    ])
    assert podcast.resolve(FakeFetcher(pages), f"https://open.spotify.com/episode/{EP}").audio_url.endswith("/12.mp3")


def test_spotify_show_gives_the_newest_or_nth_episode():
    pages = spotify_pages()
    pages[f"https://open.spotify.com/embed/show/{SHOW}"] = pages[f"https://open.spotify.com/embed/episode/{EP}"]
    fetcher = FakeFetcher(pages)
    assert podcast.resolve(fetcher, f"https://open.spotify.com/show/{SHOW}").audio_url.endswith("/13.mp3")
    assert podcast.resolve(fetcher, f"https://open.spotify.com/show/{SHOW}", nth=3).audio_url.endswith("/11.mp3")
    with pytest.raises(NotFound, match="feed has 3"):
        podcast.resolve(fetcher, f"https://open.spotify.com/show/{SHOW}", nth=4)


def test_spotify_exclusive_fails_clearly():
    pages = spotify_pages()
    pages[search("The Show")] = itunes([{"collectionName": "The Show", "artistName": "Spotify Studios"}])  # no feedUrl
    pages["https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"media": "podcast", "entity": "podcastEpisode", "limit": 25, "term": "The Big Idea The Show"})] = itunes([])
    with pytest.raises(NotFound) as error:
        podcast.resolve(FakeFetcher(pages), f"https://open.spotify.com/episode/{EP}")
    message = str(error.value)
    assert "DRM" in message and "exclusive" in message and "'The Show'" in message


def test_feed_without_the_episode_says_where_it_looked():
    pages = spotify_pages(title="A Different Episode Entirely")
    pages[f"https://open.spotify.com/embed/episode/{EP}"] = spotify_embed(
        "episode", EP, "A Different Episode Entirely", "The Show", "2025-01-01T00:00:00Z", 100_000)
    pages["https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"media": "podcast", "entity": "podcastEpisode", "limit": 25, "term": "A Different Episode Entirely The Show"})] = itunes([])
    with pytest.raises(NotFound, match=FEED_A.replace(".", r"\.")):
        podcast.resolve(FakeFetcher(pages), f"https://open.spotify.com/episode/{EP}")


def test_episode_search_finds_a_show_the_name_search_misses():
    pages = spotify_pages()
    pages[search("The Show")] = itunes([])
    pages["https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"media": "podcast", "entity": "podcastEpisode", "limit": 25, "term": "The Big Idea The Show"})] = itunes([
        {"collectionName": "The Show", "trackName": "The Big Idea", "episodeUrl": "https://cdn.example.com/found.mp3",
         "releaseDate": "2026-03-04T08:30:00Z", "trackTimeMillis": 2405000},
    ])
    episode = podcast.resolve(FakeFetcher(pages), f"https://open.spotify.com/episode/{EP}")
    assert episode.audio_url == "https://cdn.example.com/found.mp3" and episode.source == "spotify"


def test_spotify_page_tags_are_the_fallback_when_the_embed_changes():
    pages = {
        f"https://open.spotify.com/embed/episode/{EP}": (b"<html>new layout</html>", "text/html"),
        f"https://open.spotify.com/episode/{EP}": (
            b'<meta name="description" content="Listen to this episode from The Show on Spotify. Blah."/>'
            b'<meta name="music:duration" content="2400"/><meta name="music:release_date" content="2026-03-04T09:00:00Z"/>'
            b'<meta property="og:title" content="The Big Idea"/>', "text/html"),
        **{k: v for k, v in spotify_pages().items() if "open.spotify" not in k},
    }
    assert podcast.resolve(FakeFetcher(pages), f"https://open.spotify.com/episode/{EP}").audio_url.endswith("/12.mp3")


def test_matching_scores():
    ep = lambda title, date, dur: podcast.Episode(title, "u", published=podcast.parse_date(date), duration_s=dur)
    want = podcast.Wanted("The Big Idea", "S", podcast.parse_date("2026-03-04T09:00:00Z"), 2400)
    assert podcast.best_match(want, [ep("Ep 12: The Big Idea", "Wed, 04 Mar 2026 08:00:00 GMT", 2410)])[0]
    # Same title in a different year and at a different length is another episode (a repeat is not it)
    assert podcast.best_match(want, [ep("Big Ideas", "Wed, 04 Mar 2020 08:00:00 GMT", 900)])[0] is None
    # An exact title is enough when the feed lacks a date and duration
    assert podcast.best_match(want, [ep("The Big Idea", None, None)])[0]
    # A near title with a matching date and length (a rewording) passes; with neither, it does not
    assert podcast.best_match(want, [ep("The Big Idea (rerun)!", "Wed, 04 Mar 2026 08:00:00 GMT", 2395)])[0]
    assert podcast.best_match(podcast.Wanted("Crime Scene Mars", "S"), [ep("Cooking pasta", None, None)])[0] is None


def test_durations_and_dates():
    assert podcast.parse_duration("1:02:03") == 3723
    assert podcast.parse_duration("59:30") == 3570
    assert podcast.parse_duration("3600") == 3600
    assert podcast.parse_duration("soon") is None and podcast.parse_duration("") is None
    assert podcast.parse_date("Wed, 04 Mar 2026 08:30:00 +0100") == datetime(2026, 3, 4, 7, 30, tzinfo=timezone.utc)
    assert podcast.parse_date("2026-03-04T09:00:00Z") == datetime(2026, 3, 4, 9, tzinfo=timezone.utc)
    assert podcast.parse_date("not a date") is None


def test_apple_episode_and_show_links():
    lookup = "https://itunes.apple.com/lookup?" + urllib.parse.urlencode({"id": "555", "entity": "podcastEpisode", "limit": 300})
    pages = {lookup: itunes([
        {"wrapperType": "track", "kind": "podcast", "collectionName": "The Show", "artistName": "Net", "feedUrl": FEED_A},
        {"kind": "podcast-episode", "trackId": 9002, "trackName": "Ep 12", "episodeUrl": "https://cdn.example.com/a12.mp3",
         "releaseDate": "2026-03-04T08:30:00Z", "trackTimeMillis": 2410000},
        {"kind": "podcast-episode", "trackId": 9001, "trackName": "Ep 11", "episodeUrl": "https://cdn.example.com/a11.mp3"},
    ]), FEED_A: feed("The Show", [("Ep 13", "Fri, 11 Mar 2026 09:00:00 GMT", "10:00", "https://cdn.example.com/13.mp3")])}
    fetcher = FakeFetcher(pages)
    one = podcast.resolve(fetcher, "https://podcasts.apple.com/us/podcast/the-show/id555?i=9001")
    assert one.audio_url.endswith("a11.mp3") and one.source == "apple" and one.show == "The Show"
    assert podcast.resolve(fetcher, "https://podcasts.apple.com/us/podcast/the-show/id555").audio_url.endswith("/13.mp3")
    with pytest.raises(NotFound, match="RSS feed"):
        podcast.resolve(fetcher, "https://podcasts.apple.com/us/podcast/the-show/id555?i=1")


def test_rss_audio_and_page_links():
    pages = {
        FEED_A: feed("The Show", [
            ("New", "Fri, 11 Mar 2026 09:00:00 GMT", "10:00", "https://cdn.example.com/new.mp3"),
            ("Old", "Fri, 01 Jan 2021 09:00:00 GMT", "10:00", "https://cdn.example.com/old.mp3"),
        ]),
        "https://example.com/show": (b'<html><link rel="alternate" type="application/rss+xml" href="/rss.xml"></html>', "text/html"),
        "https://example.com/rss.xml": feed("The Show", [("Via page", "Fri, 11 Mar 2026 09:00:00 GMT", "10:00", "https://cdn.example.com/p.mp3")]),
        "https://example.com/plain": (b"<html>nothing here</html>", "text/html"),
        "https://example.com/stream/42": (b"ID3", "audio/mpeg"),
    }
    fetcher = FakeFetcher(pages)
    assert podcast.resolve(fetcher, FEED_A).title == "New"
    assert podcast.resolve(fetcher, FEED_A, nth=2).title == "Old"
    assert podcast.resolve(fetcher, "https://example.com/show").audio_url.endswith("p.mp3")
    direct = podcast.resolve(fetcher, "https://cdn.example.com/a%20talk.mp3?x=1")
    assert direct.audio_url.endswith("?x=1") and direct.title == "a talk" and direct.source == "url"
    assert podcast.resolve(fetcher, "https://example.com/stream/42").source == "url"
    with pytest.raises(Unsupported, match="not a Spotify"):
        podcast.resolve(fetcher, "https://example.com/plain")
    assert [e.title for e in podcast.list_episodes(fetcher, FEED_A)] == ["New", "Old"]


def test_download_streams_to_a_named_file(tmp_path):
    fetcher = FakeFetcher({"https://cdn.example.com/stream/1": (b"abc" * 1000, "audio/mp4")})
    seen = []
    path = podcast.download(fetcher, "https://cdn.example.com/stream/1", tmp_path, "Ep 1: Héllo/World", progress=lambda d, t: seen.append((d, t)))
    assert path.name == "Ep-1-Hello-World.m4a" and path.read_bytes() == b"abc" * 1000
    assert seen[-1] == (3000, 3000)
    assert not list(tmp_path.glob("*.part"))


def test_download_refuses_pages_truncation_and_size(tmp_path):
    from .podcast_fakes import FakeReply

    fetcher = FakeFetcher({"https://x.test/page": (b"<html>", "text/html")})
    with pytest.raises(FetchError, match="web page"):
        podcast.download(fetcher, "https://x.test/page", tmp_path, "e")

    class Short(FakeFetcher):
        def request(self, url, headers=None):
            return FakeReply(url, b"abc", "audio/mpeg", length=10)

    with pytest.raises(FetchError, match="ended after 3 of 10"):
        podcast.download(Short({}), "https://x.test/a.mp3", tmp_path, "e")
    assert not list(tmp_path.iterdir())
    with pytest.raises(FetchError, match="limit"):
        podcast.download(FakeFetcher({"https://x.test/a.mp3": (b"x" * 100, "audio/mpeg")}), "https://x.test/a.mp3", tmp_path, "e", limit=50)
    assert not list(tmp_path.iterdir())


def test_public_only_fetcher_refuses_private_addresses():
    fetcher = podcast.Fetcher(public_only=True)
    for url in ("http://127.0.0.1:8787/api/status", "http://localhost/x", "http://169.254.169.254/latest", "http://10.0.0.5/a.mp3"):
        with pytest.raises(podcast.PodcastError, match="public address"):
            fetcher.request(url)
    with pytest.raises(Unsupported):
        fetcher.request("file:///etc/passwd")
    with pytest.raises(Unsupported):
        podcast.Fetcher().request("file:///etc/passwd")
