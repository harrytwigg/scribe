"""Transcript exports: Markdown, SRT, JSON."""

import json
import time


def _clock(seconds: float) -> str:
    seconds = max(seconds, 0)
    return f"{int(seconds // 3600):02d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def _srt_time(seconds: float) -> str:
    seconds = max(seconds, 0)
    ms = int(round(seconds * 1000))
    return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def render(fmt: str, session: dict, lines: list[dict]) -> tuple[str, str, str]:
    start = session["started_at"]
    if fmt == "md":
        header = [f"# {session['title']}", "", f"Started {time.strftime('%Y-%m-%d %H:%M', time.localtime(start))}", ""]
        body = [f"**[{_clock(line['started_at'] - start)}] {line['speaker_name']}:** {line['text']}" for line in lines]
        return "\n\n".join(["\n".join(header).rstrip(), *body]) + "\n", "text/markdown", "md"
    if fmt == "srt":
        blocks = []
        for index, line in enumerate(lines, 1):
            begin = line["started_at"] - start
            end = begin + line["duration_ms"] / 1000
            blocks.append(f"{index}\n{_srt_time(begin)} --> {_srt_time(end)}\n{line['speaker_name']}: {line['text']}\n")
        return "\n".join(blocks), "application/x-subrip", "srt"
    if fmt == "json":
        return json.dumps({**session, "utterances": lines}, indent=2), "application/json", "json"
    raise ValueError(f"unknown format {fmt!r}; use md, srt or json")
