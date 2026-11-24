"""Helpers shared by the short, dialogue and long tracks: LID gating, ASR calls, text dumps."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import asr_texts
from pipeline.io import atomic_dump
from pipeline.stages.base import ASRModel
from pipeline.types import AudioBundle, Segment


# Sinitic dialect subtags the Phase-1 LID model (FireRedLID) emits as
# ``"zh <subtag>"``. Standard Mandarin (``"zh mandarin"`` / bare ``"zh"``) is
# NOT a dialect and does not qualify for the looser dialect DNSMOS floor.
_LID_DIALECT_SUBTAGS = frozenset(
    {"yue", "wu", "min", "xinan", "xiang", "hakka", "gan", "jin",
     "kejia", "hui", "pinghua", "cantonese"}
)


def _lid_is_dialect(lid_raw: str | None) -> bool:
    """True when Phase-1 LID's raw label names a Sinitic *dialect* (e.g.
    ``"zh xinan"``, ``"zh wu"``), not standard Mandarin. Used together with the
    ASR-side ``is_dialect`` flag: the looser dialect DNSMOS floor applies only
    when BOTH the ASR and the LID independently point at a dialect."""
    parts = (lid_raw or "").strip().lower().split()
    return len(parts) >= 2 and parts[0] == "zh" and parts[1] in _LID_DIALECT_SUBTAGS


# ====================================================================
# Short-track load / infer / finalize seam
# ====================================================================
# The short-track driver (``_run_short_track``) processes each file in three
# steps, so I/O can be prefetched off the inference thread (Stage 1) and
# several files' segments can share one ASR call (Stage 2):
#   _load_file_for_asr  — CPU/IO only, prefetchable, no GPU. Reads the
#                         partial, applies LID routing, loads the per-segment
#                         WAVs, builds the concat bundle + relative segments.
#                         Returns None to skip, or a _LoadedFile (which may be
#                         a ``write_empty`` marker needing no inference).
#   _infer_one          — the GPU step for ONE file (bucket by language hint,
#                         one asr.transcribe per bucket). Stage 2 replaces this
#                         with a cross-file batched variant.
#   _finalize_file      — stitch + the 4 post-ASR filters + secondary merge +
#                         atomic write + optional prune. CPU/IO only.


@dataclass
class _LoadedFile:
    """One file's CPU/IO-prepared state, ready for the GPU inference step.

    ``write_empty`` marks the terminal "write [] resume-marker" case (empty
    partial, or all rows routed away) — the driver writes the marker and skips
    inference. Otherwise ``bundle`` + ``rel_segments`` feed the ASR and
    ``transcribed`` (position i -> Segment) is filled in before finalize.
    """

    partial: str
    final: str
    sid_dir: str
    secondary: bool
    prune: bool = False  # driver policy, set on the returned object by the caller
    write_empty: bool = False

    dialogue_only: bool = False
    segments: list = field(default_factory=list)       # orig Segments (orig bounds)
    rows: list = field(default_factory=list)           # orig partial rows, aligned
    rel_segments: list = field(default_factory=list)   # relative to bundle, aligned
    bundle: object = None                              # AudioBundle | None
    sample_rate: int = 0
    existing_rows: list = field(default_factory=list)  # secondary pre-existing final
    transcribed: dict = field(default_factory=dict)    # position i -> Segment


def _checked_transcribe(
    asr: ASRModel, bundle: AudioBundle, segments: list[Segment], *,
    language: str | None, batch_size: int, context: str,
) -> list[Segment]:
    """Run ASR and prove one ordered result exists for every input segment.

    Positional ``zip`` silently truncates when a model drops a result, which
    can attach a short transcript to a longer audio interval. Cardinality and
    stable segment identity are therefore resume-critical invariants.
    """
    out = list(asr.transcribe(
        bundle, segments, language=language, batch_size=batch_size,
    ))
    if len(out) != len(segments):
        raise RuntimeError(
            f"ASR result count mismatch in {context}: "
            f"expected {len(segments)}, got {len(out)}"
        )
    for pos, (expected, actual) in enumerate(zip(segments, out)):
        if actual is None:
            raise RuntimeError(f"ASR returned None at {context}[{pos}]")
        if expected.index is not None and actual.index != expected.index:
            raise RuntimeError(
                f"ASR result order/id mismatch in {context}[{pos}]: "
                f"expected {expected.index!r}, got {actual.index!r}"
            )
    return out


_SUPPORTED_LIDS: frozenset[str] | None = None


def supported_lids() -> frozenset[str]:
    global _SUPPORTED_LIDS
    if _SUPPORTED_LIDS is not None:
        return _SUPPORTED_LIDS
    langs: set[str] = set()
    for mod_name in ("pipeline.adapters.qwen3_asr",
                     "pipeline.adapters.indic_conformer_asr"):
        try:
            mod = __import__(mod_name, fromlist=["x"])
        except Exception:
            continue
        for n in dir(mod):
            obj = getattr(mod, n)
            tbl = getattr(obj, "SUPPORTED_LANGUAGES", None)
            if isinstance(tbl, (set, frozenset)):
                langs |= set(tbl)
    _SUPPORTED_LIDS = frozenset(langs)
    return _SUPPORTED_LIDS


def lid_unsupported(lid: str | None) -> bool:
    lid = (lid or "").strip().lower()
    if lid in ("", "unknown"):
        return False
    tbl = supported_lids()
    if not tbl:
        return False
    return lid not in tbl


def _asr_model_name(asr: ASRModel) -> str:
    return getattr(asr, "NAME", None) or type(asr).__name__


def _dump_asr_texts(asr: ASRModel, lf: _LoadedFile, segments: list, rows: list) -> None:
    model = _asr_model_name(asr)
    table = {}
    for i, row in enumerate(rows):
        idx = row.get("index")
        if idx is None:
            continue
        table[idx] = asr_texts.build_row(lf.transcribed.get(i), asr, model)
    if not table:
        return
    sid = os.path.basename(lf.partial)[: -len(".partial.json")]
    path = asr_texts.path_for(lf.sid_dir, sid)
    merged = asr_texts.merge(asr_texts.load(path), table)
    atomic_dump(asr_texts.dump(merged), path)
