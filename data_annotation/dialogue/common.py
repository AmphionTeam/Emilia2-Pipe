"""Shared helpers for the Phase-3 dialogue track (pure stdlib, no deps).

Centralised so the phase-1 window stage, the phase-2 dialogue finalizer
(``dialogue.phase2``) and the manifest exporter build the
transcript / language set the same way.
"""
from __future__ import annotations

import collections
import glob
import os
import re
import statistics

_WORKDIR = "_phase3_work"        # 3a writes <out_root>/_phase3_work/<bucket>.sids
_DIALOGUE = ".dialogue.json"
_HEX = frozenset("0123456789abcdef")

MAX_UNIT_S = 20.0   # cap a merge unit while retaining short speaker turns
MAX_GAP_S = 3.0     # keep pauses shorter than this inside a unit; break on longer


TRUNC_TOL = 0.05


DEGEN_MIN_CHARS = 80
DEGEN_CPS = 25.0
DEGEN_REPEAT = 3


def max_ngram_repeat(text: str) -> int:
    w = text.split()
    if len(w) >= 12:
        g = collections.Counter(tuple(w[i:i + 6]) for i in range(len(w) - 5))
        return g.most_common(1)[0][1]
    s = re.sub(r"\s+", "", text)
    if len(s) >= 30:
        g = collections.Counter(s[i:i + 10] for i in range(len(s) - 9))
        return g.most_common(1)[0][1]
    return 1


def is_degenerate(text: str, span_s: float) -> bool:
    from utils.tool import get_char_count

    c = get_char_count(text or "")
    return (c > DEGEN_MIN_CHARS and span_s > 0 and c / span_s > DEGEN_CPS
            and max_ngram_repeat(text) >= DEGEN_REPEAT)


def merge_units(segments: list[dict]) -> list[dict]:
    """Group a window's segments into same-speaker units (<= MAX_UNIT_S, internal
    gaps < MAX_GAP_S kept). Each unit records its span, the speaker tag, the
    majority LID hint (for routing), aggregate dnsmos, and source seg indices."""
    segs = sorted(segments, key=lambda s: s.get("start", 0))
    groups: list[list[dict]] = []
    cur: list[dict] = []
    for s in segs:
        if (cur and s.get("speaker") == cur[-1].get("speaker")
                and (s["start"] - cur[-1]["end"]) < MAX_GAP_S
                and (s["end"] - cur[0]["start"]) <= MAX_UNIT_S):
            cur.append(s)
        else:
            if cur:
                groups.append(cur)
            cur = [s]
    if cur:
        groups.append(cur)

    units = []
    for g in groups:
        lids = [(s.get("language") or "").strip().lower() for s in g]
        lids = [x for x in lids if x and x != "unknown"]
        hint = collections.Counter(lids).most_common(1)[0][0] if lids else None
        mos = [s["dnsmos"] for s in g if s.get("dnsmos") is not None]
        units.append({
            "tag": g[0].get("tag"), "speaker": g[0].get("speaker"),
            "start": g[0]["start"], "end": g[-1]["end"],
            "seg_indices": [s.get("index") for s in g],

            "speech_s": round(sum(x["end"] - x["start"] for x in g), 3),
            "lang_hint": hint, "dnsmos": round(statistics.mean(mos), 3) if mos else None,
            "text": None, "language": None, "text_status": "pending",
        })
    return units


def is_bucket(name: str) -> bool:
    """True for a 2-hex fan-out bucket dir name (``<sid[:2]>``); excludes the
    ``_``-prefixed state/work dirs."""
    return len(name) == 2 and name[0] in _HEX and name[1] in _HEX


def worklist_sids(root: str) -> list[str] | None:
    """Windowed sids from ``<root>/_phase3_work/*.sids`` (written by 3a), or
    None if the worklist dir is absent (caller falls back to a full scan)."""
    wd = os.path.join(root, _WORKDIR)
    if not os.path.isdir(wd):
        return None
    sids: list[str] = []
    for f in glob.glob(os.path.join(wd, "*.sids")):
        try:
            with open(f) as fh:
                sids.extend(line.strip() for line in fh if line.strip())
        except OSError:
            continue
    return sids


def seg_source(s: dict):
    v = s.get("seg_source")
    return v if v is not None else s.get("text_status")


def unit_text(s: dict):
    v = s.get("text")
    return v if v is not None else s.get("phase2_text")


def assemble_transcript(segments: list[dict]) -> str:
    """Interleaved ``[S1] … [S2] …`` transcript from a window's segments.

    Consecutive same-speaker (same ``tag``) turns are merged; segments with no
    text (skip / fill_empty / fill_unroutable / not-yet-filled) contribute
    audio but no words and are simply omitted from the text."""
    runs: list[list] = []
    for s in sorted(segments, key=lambda s: s.get("start", 0)):
        tag = s.get("tag")
        if not runs or runs[-1][0] != tag:
            runs.append([tag, []])
        t = unit_text(s)
        if t:
            runs[-1][1].append(t.strip())
    return " ".join(f"[{tag}] {' '.join(txts)}" for tag, txts in runs if txts)


def dominant_languages(segments: list[dict], cover: float = 0.9) -> list[str]:
    """The minimal set of languages whose **duration-weighted** share covers
    ``cover`` (default 90%) of the window's spoken time, dropping the long tail
    of rare / spurious per-segment labels.

    Duration-weighted (not segment-count) so a handful of short mis-detected
    segments can't inflate the set: a monolingual clip → ``[zh]``; a genuine
    bilingual clip (e.g. EN↔DE interpretation) → both; a Russian clip with a
    few short pt/cs mislabels → ``[ru]``. ``unknown`` and text-less segments
    are ignored.
    """
    dur: dict[str, float] = collections.defaultdict(float)
    for s in segments:
        lang = s.get("language")
        if unit_text(s) and lang and lang != "unknown":
            dur[lang] += max(0.0, s.get("end", 0) - s.get("start", 0))
    total = sum(dur.values())
    if total <= 0:
        return []
    out: list[str] = []
    acc = 0.0
    for lang, d in sorted(dur.items(), key=lambda kv: kv[1], reverse=True):
        out.append(lang)
        acc += d
        if acc >= cover * total:
            break
    return sorted(out)


