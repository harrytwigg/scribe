"""Download one model repository into a directory. Run as a subprocess so it can be cancelled.

    python -m scribe_server.download <repo> <dest> <pattern>...

Prints one line, `total <bytes>`, before downloading, so the parent can show progress by
measuring the destination directory.
"""

import fnmatch
import sys


def split_patterns(patterns: list[str]) -> tuple[list[str], list[str]]:
    allow = [p for p in patterns if not p.startswith("!")]
    ignore = [p[1:] for p in patterns if p.startswith("!")]
    return allow or ["*"], ignore


def selected(name: str, allow: list[str], ignore: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in allow) and not any(fnmatch.fnmatch(name, p) for p in ignore)


def main(argv: list[str]) -> int:
    from huggingface_hub import HfApi, snapshot_download

    repo, dest, *patterns = argv
    allow, ignore = split_patterns(patterns)
    info = HfApi().model_info(repo, files_metadata=True)
    total = sum(s.size or 0 for s in info.siblings or [] if selected(s.rfilename, allow, ignore))
    print(f"total {total}", flush=True)
    snapshot_download(repo, local_dir=dest, allow_patterns=allow, ignore_patterns=ignore)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
