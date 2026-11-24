"""Long track: transcribe ``<sid>.long_chunks.json`` into ``<sid>.long.json``."""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf

import asr_texts
from pipeline.io import atomic_dump
from pipeline.stages.base import ASRModel
from pipeline.types import AudioBundle, Segment
from transcription.carrier_io import _read_carrier_windows, _resolve_path
from transcription.common import _asr_model_name, _checked_transcribe, lid_unsupported
from transcription.dialogue_windows import DIALOGUE_MIN_CHAR
from utils.tool import get_char_count


# Max duration (s) for the long-track WHOLE-CHUNK fallback one-shot ASR call.
# Long chunks can be hours; a multi-minute single clip hangs Qwen3-ASR/vLLM,
# so only attempt the whole-chunk fallback on short longs (per-member handles
# the rest). Tunable via env if needed.
_WHOLE_CHUNK_MAX_S = float(os.environ.get("EMILIA2_DATA_WHOLE_CHUNK_MAX_S", "60"))
# Min duration (s) for a SPLIT run to be kept as its own long. main() overrides
# from ``cfg.long_chunk.min_duration_s`` (env wins if explicitly set); the
# literal here only covers direct callers that never go through main() (tests).
_LONG_MIN_DUR_S = float(os.environ.get("EMILIA2_DATA_LONG_MIN_DUR_S", "30"))
# Max gap (s) between members that still belong to one run; main() overrides it
# from ``cfg.long_chunk.max_gap_s`` the same way.
_LONG_MAX_GAP_S = float(os.environ.get("EMILIA2_DATA_LONG_MAX_GAP_S", "2"))


# ====================================================================
# Long-track transcription
# ====================================================================
# Reads ``<sid>.long_chunks.json`` (Phase 1 kept-longs, with member
# detail) and produces ``<sid>.long.json`` containing only longs whose
# text was successfully recovered. Per :func:`_process_one_long` the
# strategy is, in order:
#   1. per-member  — slice each member from the long's WAV, ASR each
#   2. whole-chunk — one ASR call on the whole long (short chunks only).
#      Needs vLLM's ``max_num_batched_tokens`` bumped to at least 32 768
#      so a ~20-min single call fits in one scheduling round
#   3. split — cut at the failed member(s), keep >=30s runs as _p<k>
#   4. drop — if everything failed, the long isn't emitted
# We no longer split longs into smaller longs on irrecoverable failures.
# The configured Phase-1 long-chunk duration and quality gates define what a
# long is; a partial reconstruction
# wouldn't re-pass them cleanly.


def _long_audio_bundle(
    long_meta: dict, sid_dir: str, target_sr: int | None = None,
) -> AudioBundle:
    """Load one legacy long clip or one window from a shared carrier."""
    if long_meta.get("carrier_path"):
        wav, sr = _read_carrier_windows(
            [long_meta], sid_dir, target_sr=target_sr,
        )[0]
        return AudioBundle(waveform=wav, sample_rate=sr, name="long")
    path = long_meta.get("audio_path")
    if not path:
        raise RuntimeError(
            f"long-chunk {long_meta.get('index')} has neither audio_path "
            f"nor carrier_path; Phase 1 export is inconsistent"
        )
    wav, sr = sf.read(
        _resolve_path(path, sid_dir),
        dtype="float32", always_2d=False,
    )
    if wav.ndim == 2:
        wav = wav.mean(axis=1).astype(np.float32)
    return AudioBundle(waveform=wav, sample_rate=int(sr), name="long")


def _rel_member_segments(
    long_start: float, members: list[dict],
) -> list[Segment]:
    """Build Segments relative to the long-chunk WAV (start=0 = long.start)."""
    out = []
    for m in members:
        out.append(Segment(
            start=m["start"] - long_start,
            end=m["end"] - long_start,
            speaker=m.get("speaker"),
            index=m.get("index"),
            quality=m.get("dnsmos"),
            language=m.get("language"),
            extra={"is_fake": m.get("is_fake")},
        ))
    return out


def _safe_transcribe(
    asr: ASRModel, bundle: AudioBundle,
    rel_segs: list[Segment], logger, ratio_filter=None,
) -> list[tuple[str, str] | None]:
    """Per-segment ``(text, detected_language)``; None for any segment whose
    ASR call returned empty / no language / wasn't accepted by the adapter's
    language gate.

    ``detected_language`` is the ASR's OWN call (base ISO code), not the
    phase1 LID. The long track keys language off this so a chunk reflects
    what was actually transcribed — LID may say 'unknown' while the ASR
    knows it's en/zh. Batch-level exception → all None.
    """
    if not rel_segs:
        return []
    results = _checked_transcribe(
        asr, bundle, rel_segs, language=None, batch_size=len(rel_segs),
        context="long-track",
    )
    out: list[tuple[str, str] | None] = []
    for seg, r in zip(rel_segs, results):
        text = (r.text or "").strip() if r is not None else ""
        st = asr_texts.SegText(
            text=text, language=(r.language if r is not None else None),
            accepted=bool(r is not None and asr.accepts(r.language)),
            asr_model=_asr_model_name(asr),
            extra=dict((r.extra if r is not None else None) or {}),
        )
        reason = asr_texts.chunk_verdict(
            st, float(seg.extra.get("speech_s") or (seg.end - seg.start)),
            min_char=DIALOGUE_MIN_CHAR, ratio_filter=ratio_filter,
            char_count=get_char_count,
        )
        if reason is not None:
            out.append(None)
            continue
        out.append((text, r.language))
    return out


def _emit_long_row(
    long_meta: dict, sub_start: float, sub_end: float,
    members_with_tl: list,  # list[(member_meta, (text, detected_lang) | None)]
    audio_path: str | None,
    source: str,
    top_text: str | None,
    top_lang: str | None = None,
) -> dict:
    """Shape a long-track JSON row.

    ``members_with_tl`` is the per-member breakdown: one
    ``(member_meta, (text, detected_lang) | None)`` per member (None =
    failed / covered by an adjacent concat-patch). ``top_text`` is the
    canonical chunk text; ``top_lang`` is the detected language for the
    whole-chunk fallback (which has no per-member breakdown).

    Language uses the ASR's OWN detection, not the phase1 LID (LID may be
    'unknown' while the ASR knows the language). And the chunk can carry
    MULTIPLE languages — long audio often code-switches (zh+en). So:
      * ``language``  = comma-joined unique detected langs ("zh" or "zh,en")
      * ``languages`` = the list form, for clean downstream filtering
    Member ``language`` is the ASR detection for that member (falls back to
    the phase1 LID only when the ASR produced nothing for it).
    """
    row: dict = {
        "start": sub_start,
        "end": sub_end,
        "speaker": long_meta.get("speaker"),
        "index": long_meta.get("index"),
        "mean_dnsmos": long_meta.get("mean_dnsmos"),
        "text": top_text,
        "source": source,
    }
    detected: list[str] = []
    if members_with_tl:
        row["members"] = []
        for m, tl in members_with_tl:
            text = tl[0] if tl else None
            # ASR-detected lang for this member; fall back to phase1 LID
            # only when the ASR gave nothing (failed / concat-covered).
            lang = tl[1] if tl else (m.get("language") or None)
            if lang and lang != "unknown":
                detected.append(lang)
            row["members"].append({
                "index": m.get("index"),
                "start": m["start"],
                "end": m["end"],
                "speaker": m.get("speaker"),
                "language": lang,
                "dnsmos": m.get("dnsmos"),
                "is_fake": m.get("is_fake"),
                "deepfake_score": m.get("deepfake_score"),
                "text": text,
            })
    elif top_lang and top_lang != "unknown":
        detected.append(top_lang)
    # Chunk language(s): unique ASR-detected langs, order preserved. Single
    # language → "zh"; code-switch → "zh,en". `languages` is the list form.
    uniq = list(dict.fromkeys(detected))
    if uniq:
        row["language"] = ",".join(uniq)
        row["languages"] = uniq
    if audio_path is not None:
        row["audio_path"] = audio_path
        row["sample_rate"] = long_meta.get("sample_rate")
    elif long_meta.get("carrier_path"):
        sr = int(long_meta["sample_rate"])
        parent_a = int(long_meta["carrier_start_samples"])
        parent_b = int(long_meta["carrier_end_samples"])
        if sub_start == long_meta.get("start") and sub_end == long_meta.get("end"):
            a, b = parent_a, parent_b
        else:
            a = max(parent_a, min(parent_b, int(float(sub_start) * sr)))
            b = max(a, min(parent_b, int(float(sub_end) * sr)))
        row.update({
            "sample_rate": sr,
            "carrier_path": long_meta["carrier_path"],
            "carrier_start_samples": a,
            "carrier_end_samples": b,
        })
    return row


def _split_long_into_runs(
    long_meta: dict, sid_dir: str, members: list, rel_segs: list,
    patched: list, bundle: AudioBundle, logger,
    roles: list[str] | None = None, max_gap_s: float = 2.0,
) -> list[dict]:
    if roles is None:
        roles = [asr_texts.BREAK if t is None else asr_texts.OK for t in patched]

    runs = asr_texts.assemble_runs(
        members, roles, max_gap_s=max_gap_s,
        min_duration_s=_LONG_MIN_DUR_S, min_speakers=1)

    wav = bundle.waveform if bundle is not None else None
    sr = bundle.sample_rate if bundle is not None else (
        long_meta.get("sample_rate") or 44100)
    base, ext = os.path.splitext(
        os.path.basename(long_meta.get("audio_path") or "long.flac"))
    rows: list[dict] = []
    for k, run in enumerate(runs):
        i, j = run[0], run[-1]
        run_start, run_end = members[i]["start"], members[j]["end"]
        sub = None
        if not long_meta.get("carrier_path"):
            if wav is None:
                continue
            a = max(0, int(rel_segs[i].start * sr))
            b = int(rel_segs[j].end * sr)
            clip = wav[a:b]
            if not clip.size:
                continue
            sub = f"{base}_p{k}{ext}"
            sf.write(os.path.join(sid_dir, sub), clip, sr, subtype="PCM_16")

        text = " ".join(patched[x][0] for x in run
                        if roles[x] == asr_texts.OK and patched[x] and patched[x][0])
        rows.append(_emit_long_row(
            long_meta, run_start, run_end,
            members_with_tl=[(members[x],
                              patched[x] if roles[x] == asr_texts.OK else None)
                             for x in run],
            audio_path=sub, source="per_member_split", top_text=text,
        ))
    n_dropped = sum(1 for r in roles if r != asr_texts.OK)
    if rows or n_dropped:
        logger.debug(f"long-track split: {long_meta.get('index')} → {len(rows)} "
                     f"run(s), {n_dropped} non-ok member(s)")
    return rows


def _member_speech_s(m: dict) -> float:
    return float(m.get("speech_s") or (m.get("end", 0) - m.get("start", 0)))


def _member_roles_from_table(members: list[dict], table: dict,
                             ratio_filter=None) -> list[str] | None:

    out = []
    for m in members:
        if lid_unsupported(m.get("language")):
            out.append(asr_texts.BREAK)
            continue
        st = table.get(m.get("index"))
        if st is None:
            return None
        out.append(asr_texts.role_of(asr_texts.chunk_verdict(
            st, _member_speech_s(m), min_char=DIALOGUE_MIN_CHAR,
            ratio_filter=ratio_filter)))
    return out


def _member_texts_from_table(members: list[dict], table: dict,
                             ratio_filter=None) -> list | None:

    out = []
    for m in members:
        if lid_unsupported(m.get("language")):
            out.append(None)
            continue
        st = table.get(m.get("index"))
        if st is None:
            return None
        bad = asr_texts.chunk_verdict(
            st, _member_speech_s(m), min_char=DIALOGUE_MIN_CHAR,
            ratio_filter=ratio_filter)
        out.append(None if bad is not None else (st.text, st.language))
    return out


def _process_one_long(
    long_meta: dict, sid_dir: str, asr: ASRModel, logger,
    bundle: AudioBundle | None = None,
    table: dict | None = None,
    ratio_filter=None,
) -> list[dict]:
    """Process one Phase-1 long chunk → 0 or 1 kept long-track row.

    Strategy (per the latest spec):

      1. Per-member ASR walk. All members succeed → emit one row with
         ``source=per_member`` and per-member text.

      2. Some members fail (empty text / rejected language e.g. ms/id) →
         whole-chunk ASR as a fallback, short chunks only
         (``_WHOLE_CHUNK_MAX_S``; vLLM hangs on multi-minute one-shots).
         Success → ``source=whole_chunk_fallback``, single top-level
         text, no member breakdown.

      3. Otherwise split at the failed member(s) into runs of
         consecutive transcribed members (``source=per_member_split``,
         physical ``_p<k>`` sub-clips; failed members' audio excluded).

    """
    members = long_meta.get("members") or []
    if not members:
        return []

    long_start = long_meta["start"]
    long_end = long_meta["end"]

    rel_segs = _rel_member_segments(long_start, members)

    def _need_bundle():
        nonlocal bundle
        if bundle is None:
            bundle = _long_audio_bundle(
                long_meta, sid_dir, target_sr=int(getattr(asr, "SR", 0) or 0) or None,
            )
        return bundle

    roles = None
    results = _member_texts_from_table(members, table, ratio_filter) if table else None
    if results is not None:
        roles = _member_roles_from_table(members, table, ratio_filter)
    else:
        results = _safe_transcribe(
            asr, _need_bundle(), rel_segs, logger, ratio_filter=ratio_filter,
        )

    # ---- 1. per-member happy path ----
    if all(r is not None for r in results):
        top = " ".join(r[0] for r in results)  # type: ignore[index]
        return [_emit_long_row(
            long_meta, long_start, long_end,
            members_with_tl=list(zip(members, results)),
            audio_path=long_meta.get("audio_path"),
            source="per_member",
            top_text=top,
        )]

    patched: list[tuple[str, str] | None] = list(results)

    # ---- 3. whole-chunk fallback (SHORT longs only) ----
    # Qwen3-ASR/vLLM HANGS on a multi-minute single clip (long chunks can be
    # up to hours — a 90-min whole-chunk call wedged the engine for >1h). The
    # per-member path above is safe (it sends short member clips); only this
    # whole-chunk one-shot is dangerous, so cap it. Longer chunks whose
    # per-member already failed are dropped rather than risk hanging phase-2.
    whole_dur = long_end - long_start
    if whole_dur <= _WHOLE_CHUNK_MAX_S:
        whole_seg = Segment(start=0.0, end=whole_dur)
        whole = _safe_transcribe(
            asr, _need_bundle(), [whole_seg], logger, ratio_filter=ratio_filter,
        )
        if whole and whole[0]:
            wt, wl = whole[0]
            return [_emit_long_row(
                long_meta, long_start, long_end,
                members_with_tl=[],
                audio_path=long_meta.get("audio_path"),
                source="whole_chunk_fallback",
                top_text=wt, top_lang=wl,
            )]

    # ---- 4. split at the failed member(s), keep transcribed runs ----
    # (per-member couldn't be fully covered and whole-chunk wasn't applicable;
    # salvage the good stretches instead of dropping the whole long.)

    _b = None if long_meta.get("carrier_path") else _need_bundle()
    return _split_long_into_runs(
        long_meta, sid_dir, members, rel_segs, patched, _b, logger,
        roles=roles, max_gap_s=_LONG_MAX_GAP_S)


def _transcribe_long_chunks(
    src_path: str, dst_path: str, sid_dir: str, asr: ASRModel, logger,
    ratio_filter=None,
) -> None:
    """Phase-2 long-track work for a single ``<sid>.long_chunks.json``."""
    with open(src_path) as f:
        longs = json.load(f)
    if not longs:
        atomic_dump([], dst_path)
        return

    out: list[dict] = []

    _sid = os.path.basename(src_path)[: -len(".long_chunks.json")]
    table = asr_texts.load(asr_texts.path_for(sid_dir, _sid))
    need_audio = not table or not all(
        all(table.get(m.get("index")) is not None for m in (lm.get("members") or []))
        for lm in longs)
    if table and not need_audio:
        logger.info(f"long track: {len(longs)} chunks from asr_texts table "
                    f"(no ASR, no decode)")

    carrier_bundles = None
    if need_audio and all(lm.get("carrier_path") for lm in longs):
        loaded = _read_carrier_windows(
            longs, sid_dir,
            target_sr=int(getattr(asr, "SR", 0) or 0) or None,
        )
        carrier_bundles = [
            AudioBundle(waveform=wav, sample_rate=sr, name="long")
            for wav, sr in loaded
        ]
    for i, lm in enumerate(longs):
        bundle = carrier_bundles[i] if carrier_bundles is not None else None
        rows = _process_one_long(
            lm, sid_dir, asr, logger=logger, bundle=bundle, table=table,
            ratio_filter=ratio_filter,
        )
        out.extend(rows)

    atomic_dump(out, dst_path)

    n_pm = sum(1 for r in out if r.get("source") == "per_member")
    n_split = sum(1 for r in out if r.get("source") == "per_member_split")
    n_whole = sum(1 for r in out if r.get("source") == "whole_chunk_fallback")
    logger.info(
        f"{os.path.basename(dst_path)}: "
        f"kept={len(out)} (per_member={n_pm} "
        f"split={n_split} whole={n_whole}) in={len(longs)}"
    )


def configure(*, max_gap_s: float | None = None, min_dur_s: float | None = None) -> None:
    """Override the long-track limits; ``None`` keeps the env/default value."""
    global _LONG_MAX_GAP_S, _LONG_MIN_DUR_S
    if max_gap_s is not None:
        _LONG_MAX_GAP_S = float(max_gap_s)
    if min_dur_s is not None:
        _LONG_MIN_DUR_S = float(min_dur_s)
