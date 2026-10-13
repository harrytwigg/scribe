"""Sessions and transcript lines, in SQLite (docs/spec.md, 3.5)."""

import sqlite3
import threading
import time
import uuid
from pathlib import Path

SCHEMA = """
create table if not exists sessions (
  id text primary key,
  source text not null,
  title text not null,
  guild text,
  channel text,
  started_at real not null,
  ended_at real
);
create table if not exists utterances (
  id integer primary key autoincrement,
  session_id text not null references sessions(id) on delete cascade,
  speaker_id text not null,
  speaker_name text not null,
  started_at real not null,
  duration_ms integer not null,
  text text not null,
  model_id text,
  asr_ms integer
);
create index if not exists utterances_session on utterances(session_id, started_at);
-- Per-session speakers: a name given in the GUI, and for a diarized channel the speaker's voice
-- (a unit-length embedding), so the same person keeps their label all call (docs/spec.md, 3.8).
create table if not exists speakers (
  session_id text not null references sessions(id) on delete cascade,
  speaker_id text not null,
  name text,
  voice blob,
  weight real,
  primary key (session_id, speaker_id)
);
create table if not exists optouts (
  source text not null,
  user_id text not null,
  primary key (source, user_id)
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("pragma foreign_keys = on")
        self._db.execute("pragma journal_mode = wal")
        self._db.executescript(SCHEMA)
        self._migrate()
        self._lock = threading.Lock()

    def _migrate(self) -> None:
        columns = {row["name"] for row in self._db.execute("pragma table_info(utterances)")}
        if "client_key" not in columns:
            # A client's own id for an utterance, so a retried post is stored once.
            self._db.execute("alter table utterances add column client_key text")
        self._db.execute(
            "create unique index if not exists utterances_client_key "
            "on utterances(session_id, client_key) where client_key is not null"
        )

    def _all(self, sql: str, *args) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._db.execute(sql, args).fetchall()]

    def _one(self, sql: str, *args) -> dict | None:
        rows = self._all(sql, *args)
        return rows[0] if rows else None

    def _exec(self, sql: str, *args) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, args)

    # -- sessions -------------------------------------------------------------------------------

    def create_session(self, source: str, title: str, guild: str | None = None, channel: str | None = None) -> dict:
        session_id = uuid.uuid4().hex[:12]
        self._exec(
            "insert into sessions (id, source, title, guild, channel, started_at) values (?, ?, ?, ?, ?, ?)",
            session_id, source, title, guild, channel, time.time(),
        )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict | None:
        return self._one(
            "select s.*, (select count(*) from utterances u where u.session_id = s.id) as lines "
            "from sessions s where id = ?",
            session_id,
        )

    def list_sessions(self, limit: int = 100) -> list[dict]:
        return self._all(
            "select s.*, (select count(*) from utterances u where u.session_id = s.id) as lines "
            "from sessions s order by started_at desc limit ?",
            limit,
        )

    def count_open_sessions(self) -> int:
        return self._one("select count(*) as n from sessions where ended_at is null")["n"]

    def open_sessions(self, source: str) -> list[dict]:
        return self._all("select * from sessions where ended_at is null and source = ?", source)

    def rename_session(self, session_id: str, title: str) -> None:
        self._exec("update sessions set title = ? where id = ?", title, session_id)

    def end_session(self, session_id: str) -> None:
        self._exec("update sessions set ended_at = ? where id = ? and ended_at is null", time.time(), session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._exec("delete from sessions where id = ?", session_id).rowcount > 0

    # -- utterances -----------------------------------------------------------------------------

    def add_utterance(
        self,
        session_id: str,
        speaker_id: str,
        speaker_name: str,
        started_at: float,
        duration_ms: int,
        text: str,
        model_id: str | None,
        asr_ms: int | None,
        client_key: str | None = None,
    ) -> tuple[dict, bool]:
        """Store a line; returns it and whether it is new (False: this client_key was stored before)."""
        cursor = self._exec(
            "insert into utterances (session_id, speaker_id, speaker_name, started_at, duration_ms, text, model_id, asr_ms, client_key) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?) on conflict do nothing",
            session_id, speaker_id, speaker_name, started_at, duration_ms, text, model_id, asr_ms, client_key,
        )
        if cursor.rowcount == 0:
            return self.utterance_by_key(session_id, client_key), False
        return self._one("select * from utterances where id = ?", cursor.lastrowid), True

    def utterance_by_key(self, session_id: str, client_key: str | None) -> dict | None:
        if client_key is None:
            return None
        return self._one("select * from utterances where session_id = ? and client_key = ?", session_id, client_key)

    def utterances_by_key_prefix(self, session_id: str, client_key: str) -> list[dict]:
        """The lines one diarized utterance became: keys `<client_key>#0`, `#1`, ..."""
        return self._all(
            "select * from utterances where session_id = ? and client_key like ? escape '\\' order by started_at, id",
            session_id, client_key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "#%",
        )

    def utterances(self, session_id: str) -> list[dict]:
        return self._all("select * from utterances where session_id = ? order by started_at, id", session_id)

    # -- speakers -------------------------------------------------------------------------------

    def speaker_names(self, session_id: str) -> dict[str, str]:
        rows = self._all("select speaker_id, name from speakers where session_id = ? and name is not null", session_id)
        return {row["speaker_id"]: row["name"] for row in rows}

    def voices(self, session_id: str, prefix: str) -> list[tuple[str, bytes, float]]:
        rows = self._all(
            "select speaker_id, voice, weight from speakers where session_id = ? and voice is not null", session_id
        )
        return [(r["speaker_id"], r["voice"], r["weight"]) for r in rows if r["speaker_id"].startswith(prefix)]

    def save_voice(self, session_id: str, speaker_id: str, voice: bytes, weight: float) -> None:
        self._exec(
            "insert into speakers (session_id, speaker_id, voice, weight) values (?, ?, ?, ?) "
            "on conflict (session_id, speaker_id) do update set voice = excluded.voice, weight = excluded.weight",
            session_id, speaker_id, voice, weight,
        )

    def rename_speaker(self, session_id: str, speaker_id: str, name: str) -> int:
        """Name a speaker for the session: their lines so far, and every line after. Returns lines changed."""
        with self._lock:
            self._db.execute("begin")
            try:
                self._db.execute(
                    "insert into speakers (session_id, speaker_id, name) values (?, ?, ?) "
                    "on conflict (session_id, speaker_id) do update set name = excluded.name",
                    (session_id, speaker_id, name),
                )
                changed = self._db.execute(
                    "update utterances set speaker_name = ? where session_id = ? and speaker_id = ?",
                    (name, session_id, speaker_id),
                ).rowcount
                self._db.execute("commit")
            except BaseException:
                self._db.execute("rollback")
                raise
        return changed

    # -- opt-outs -------------------------------------------------------------------------------

    def set_optout(self, source: str, user_id: str, opted_out: bool) -> None:
        if opted_out:
            self._exec("insert or ignore into optouts (source, user_id) values (?, ?)", source, user_id)
        else:
            self._exec("delete from optouts where source = ? and user_id = ?", source, user_id)

    def optouts(self, source: str) -> list[str]:
        return [row["user_id"] for row in self._all("select user_id from optouts where source = ?", source)]
