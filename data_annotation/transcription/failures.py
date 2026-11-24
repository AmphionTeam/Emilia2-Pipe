"""Track files that keep failing so one bad recording cannot stall a run."""

from __future__ import annotations

import json
import os
import time

from utils.logger import Logger


def _touch(path: str) -> None:
    """Best-effort mtime bump (lease heartbeat). Silent on a vanished lock."""
    try:
        os.utime(path, None)
    except OSError:
        pass


_P2_MAX_FAILS = 3
_P2_FAILS: dict[str, int] = {}


def _poisoned(src: str) -> bool:
    return _P2_FAILS.get(src, 0) >= _P2_MAX_FAILS


def _poisoned_all() -> list[str]:
    return sorted(p for p, n in _P2_FAILS.items() if n >= _P2_MAX_FAILS)


def _root_poison(root: str) -> list[str]:
    pref = os.path.abspath(root).rstrip(os.sep) + os.sep
    return [p for p in _poisoned_all() if os.path.abspath(p).startswith(pref)]


def _record_phase2_fail(
    failed_log_path: str, src: str, e: Exception, key: str = "partial",
) -> None:
    """Append one failure record. ``key`` is the field name for ``src``
    (``partial`` for the short track, ``long_chunks`` for the long track)."""
    with open(failed_log_path, "a") as f:
        f.write(json.dumps({
            "ts": time.time(), key: src,
            "error_type": type(e).__name__, "error_msg": str(e),
        }, ensure_ascii=False) + "\n")
    n = _P2_FAILS.get(src, 0) + 1
    _P2_FAILS[src] = n
    if n == _P2_MAX_FAILS:
        Logger.get_logger().warning(
            f"{src}: failed {n} times; no more retries in this process "
            f"(not finalized; queue entry remains for restart). Details: "
            f"{os.path.basename(failed_log_path)}"
        )
