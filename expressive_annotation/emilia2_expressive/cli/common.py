"""Argument helpers shared by the annotation CLIs."""
from __future__ import annotations

import argparse


def parse_shard(parser: argparse.ArgumentParser, spec: str) -> tuple[int, int] | None:
    """``"i/n"`` -> ``(i, n)``; an empty spec means no sharding."""
    if not spec:
        return None
    try:
        index, count = (int(x) for x in spec.split("/", 1))
    except ValueError:
        parser.error(f"--shard {spec!r} is not `i/n`")
    if not 0 <= index < count:
        parser.error(f"--shard {spec!r}: need 0 <= i < n")
    return index, count
