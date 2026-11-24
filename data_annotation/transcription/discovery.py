"""Find Phase 2 work: shard, scan or dequeue prefixes, and coordinate instances with lease locks."""

from __future__ import annotations

import glob
import hashlib
import os
import time

from pipeline.io import work_queue


def _shard_dirs(root: str, shard: int, num_shards: int) -> list[str]:
    """Top-level data dirs in this shard, for the start-up prescan.

    The big run uses a 2-hex fan-out (``<root>/<sid[:2]>/<sid>/``). A dir
    belongs to shard ``int(name, 16) % num_shards``, or to the md5 of its name
    when the name is not hex, so the flat single-folder layout
    (``<root>/<sid>/``) shards the same way. Only the prescan uses this: with N
    instances, each lists about 1/N of the tree instead of all of it. Which
    instance transcribes a prefix is decided by the claim locks, not by the
    shard. ``num_shards <= 1`` returns every data dir; ``_``-prefixed dirs
    (_claims/_manifests/_state/markers) are skipped.
    """
    out: list[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        if name.startswith("_"):
            continue
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        try:
            key = int(name, 16)
        except ValueError:
            key = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16)
        if num_shards <= 1 or key % num_shards == shard:
            out.append(full)
    return out


def _glob_processed(
    root: str, suffix: str, shard: int = 0, num_shards: int = 1
) -> list[str]:
    """Match both layouts, scanning only this shard's top-level dirs:"""
    out: list[str] = []
    for d in _shard_dirs(root, shard, num_shards):
        out.extend(glob.glob(os.path.join(d, f"*{suffix}")))        # flat
        out.extend(glob.glob(os.path.join(d, "*", f"*{suffix}")))   # fan-out
    return out


def _scan_partials(
    processed_root: str, shard: int = 0, num_shards: int = 1
) -> list[tuple[str, str]]:
    """Find ``(partial_path, final_path)`` pairs that still need Phase 2."""
    out: list[tuple[str, str]] = []
    for partial in _glob_processed(processed_root, ".partial.json", shard, num_shards):
        sid_dir = os.path.dirname(partial)
        sid = os.path.basename(partial)[: -len(".partial.json")]
        final = os.path.join(sid_dir, sid + ".json")
        if os.path.exists(final):
            continue  # already transcribed
        out.append((partial, final))
    return out


def _scan_long_chunks(
    processed_root: str, shard: int = 0, num_shards: int = 1
) -> list[tuple[str, str]]:
    """Find ``(long_chunks_path, long_final_path)`` pairs needing Phase 2.

    Mirrors ``_scan_partials`` but for the long track. Resume is
    independent — a file may have its shorts transcribed but not yet
    its longs, or vice versa.
    """
    out: list[tuple[str, str]] = []
    for src in _glob_processed(processed_root, ".long_chunks.json", shard, num_shards):
        sid_dir = os.path.dirname(src)
        sid = os.path.basename(src)[: -len(".long_chunks.json")]
        dst = os.path.join(sid_dir, sid + ".long.json")
        if os.path.exists(dst):
            continue
        out.append((src, dst))
    return out


def _prefix_dirs(root: str) -> list[str]:
    """Existing top-level data dirs (fan-out prefixes or flat SIDs)."""
    out: list[str] = []
    try:
        names = os.listdir(root)
    except OSError:
        return out
    for name in names:
        if name.startswith("_"):
            continue
        if os.path.isdir(os.path.join(root, name)):
            out.append(name)
    return out


def _claim_lock(lock_path: str, lease_s: float) -> bool:
    """Atomically claim ``lock_path`` (O_EXCL); steal it if its mtime is older
    than ``lease_s`` (previous owner died mid-prefix). World-writable for
    cross-uid multi-node. Returns True iff this process now owns it."""
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        os.write(fd, str(time.time()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock_path) > lease_s:
                os.utime(lock_path, None)  # steal a stale claim
                return True
        except OSError:
            pass
        return False
    except OSError:
        return False


# ---- drained-prefix markers -------------------------------------------------
# The claim lock only says "someone is working this prefix RIGHT NOW"; it is
# removed the moment they finish. So every instance still walks the full 256-
# prefix list and pays a real ``_scan_prefix`` (a ~3900-sid-dir glob) on prefixes
# a peer already drained, just to learn there is nothing left. With N instances
# that multiplies redundant scans across the 256 prefixes.
# A drained marker makes that "nothing left" answer cost 1 read + 1 stat instead
# of the glob. Two deliberate constraints keep it honest:
#   * Written ONLY when a scan found NOTHING — never after draining work. A
#     prefix whose drain hit a per-file failure (logged to _failed_phase2.jsonl,
#     final not written) therefore stays discoverable and is still retried by the
#     next instance exactly as before. This costs one extra scan per prefix (the
#     one that observes it empty and marks it) but changes no resume/retry
#     semantics: scans drop ~16x -> ~2x per prefix, not to 1x.
#   * Invalidated by a fingerprint of the prefix dir, captured BEFORE the scan
#     so anything phase 1 adds mid-scan invalidates the marker instead of being
#     silently skipped. This is what makes the marker safe under ``--loop``
#     (phase 1 still streaming), not just for a one-shot drain.


def _prefix_fingerprint(pdir: str) -> str | None:
    """Cheap fingerprint of a prefix dir: (entry count, mtime). None if unstattable.

    Phase 1 lands each recording in a NEW ``<prefix>/<sid>/`` dir and nothing
    ever removes one (``--prune-segments`` deletes segment FILES inside a sid
    dir, never the dir), so the entry count grows strictly monotonically as
    fresh work arrives.

    The count — not the mtime — is what makes this correct. mtime alone is
    unusable here for three independent reasons, all of which silently skip real
    work: (1) a sid created in the same filesystem timestamp tick as the marker
    leaves mtime byte-identical; (2) NFS
    attribute caching can serve a stale dir mtime for up to acdirmax (~60s);
    (3) mtime comes from the server's clock while any freshness comparison would
    use the client's, so clock skew corrupts it. An entry count has none of those
    failure modes. mtime rides along only as a cheap second signal.

    Costs 1 readdir + 1 stat, versus _scan_prefix's ~3900 per-sid readdirs.
    """
    try:
        n = len(os.listdir(pdir))
        mt = os.stat(pdir).st_mtime
    except OSError:
        return None
    return f"{n}:{mt!r}"


def _prefix_drained(marker: str, pdir: str) -> bool:
    """True iff a peer proved this prefix empty for our track AND the prefix dir
    is unchanged since (no new phase-1 output landed)."""
    try:
        with open(marker) as f:
            recorded = f.read().strip()
    except OSError:
        return False  # never marked (or unreadable) -> scan it
    fp = _prefix_fingerprint(pdir)
    return fp is not None and recorded == fp


def _mark_drained(marker: str, fp_before: str | None) -> None:
    """Record that this prefix scanned empty at fingerprint ``fp_before``
    (captured BEFORE the scan). Best-effort and atomic; world-writable for
    cross-uid multi-node. A lost marker only costs a redundant scan, never
    correctness."""
    if fp_before is None:
        return
    tmp = f"{marker}.{os.getpid()}.tmp"
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o666)
        try:
            os.write(fd, fp_before.encode())
        finally:
            os.close(fd)
        os.replace(tmp, marker)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _scan_prefix(
    root: str, prefix: str, secondary: bool, tracks: str = "both",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(short_work, long_work)`` for a single fan-out prefix dir.

    Primary mode skips sids whose ``<sid>.json`` already exists; secondary
    mode returns every partial (the fill-in logic skips covered indices).
    Long track is primary-only.

    ``tracks`` ("both"/"short"/"long") gates which half is *scanned*. The
    caller could scan both and discard the unwanted half, causing unnecessary
    metadata requests. Scanning only the requested track preserves the same
    (tracks x secondary) behavior."""
    pdir = os.path.join(root, prefix)

    def _file_names(d: str) -> list[str]:
        try:
            with os.scandir(d) as it:
                return [e.name for e in it]
        except OSError:
            return []

    def _entries(d: str) -> list[tuple[str, bool]]:
        try:
            with os.scandir(d) as it:
                return [(e.name, e.is_dir()) for e in it]
        except OSError:
            return []

    want_short = tracks in ("both", "short")
    want_long = (not secondary) and tracks in ("both", "long")

    def _collect(dirpath: str, names: list[str], work: list, long_work: list) -> None:
        fset = set(names)
        if want_short:
            for n in names:
                if not n.endswith(".partial.json"):
                    continue
                sid = n[: -len(".partial.json")]
                if secondary or (sid + ".json") not in fset:
                    work.append((os.path.join(dirpath, n),
                                 os.path.join(dirpath, sid + ".json")))
        if want_long:
            for n in names:
                if not n.endswith(".long_chunks.json"):
                    continue
                sid = n[: -len(".long_chunks.json")]
                if (sid + ".long.json") not in fset:
                    long_work.append((os.path.join(dirpath, n),
                                      os.path.join(dirpath, sid + ".long.json")))

    work: list[tuple[str, str]] = []
    long_work: list[tuple[str, str]] = []
    top = _entries(pdir)

    _collect(pdir, [n for n, _ in top], work, long_work)

    for name, is_dir in top:
        if not is_dir:
            continue
        sub = os.path.join(pdir, name)
        _collect(sub, _file_names(sub), work, long_work)
    return work, long_work


def _queue_prefix(
    root: str, prefix: str, tracks: str,
    pend_short: list[str], pend_long: list[str],
) -> tuple[list, list, list, list, int]:
    want_short = tracks in ("both", "short")
    want_long = tracks in ("both", "long")

    _names: dict[str, list[str]] = {}

    def names(sid: str) -> list[str]:
        if sid not in _names:
            try:
                with os.scandir(os.path.join(root, prefix, sid)) as it:
                    _names[sid] = [e.name for e in it]
            except OSError:
                _names[sid] = []
        return _names[sid]

    work: list[tuple[str, str]] = []
    long_work: list[tuple[str, str]] = []
    ack_short: list[tuple[str, str]] = []
    ack_long: list[tuple[str, str]] = []
    stale = 0

    def collect(sids, src_suffix, dst_suffix, out, acks, track):
        nonlocal stale
        for sid in sids:
            fset = set(names(sid))
            hit = False
            for n in sorted(fset):
                if not n.endswith(src_suffix):
                    continue
                base = n[: -len(src_suffix)]
                dst = base + dst_suffix
                sid_dir = os.path.join(root, prefix, sid)
                if dst in fset:
                    continue
                hit = True
                out.append((os.path.join(sid_dir, n), os.path.join(sid_dir, dst)))
                acks.append((sid, os.path.join(sid_dir, dst)))
            if not hit:
                work_queue.ack(root, sid, track)
                stale += 1

    if want_short:
        collect(pend_short, ".partial.json", ".json", work, ack_short, "short")
    if want_long:
        collect(pend_long, ".long_chunks.json", ".long.json",
                long_work, ack_long, "long")
    return work, long_work, ack_short, ack_long, stale
