"""Short track: load, transcribe and finalize ``<sid>.partial.json`` into ``<sid>.json``."""

from __future__ import annotations

import gc
import json
import os
import socket
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import numpy as np
import torch
import tqdm

import asr_texts
from pipeline.io import atomic_dump
from pipeline.stages.base import ASRModel
from pipeline.types import AudioBundle, Segment
from transcription.carrier_io import (
    _own_audio, _read_segments_parallel, _resolve_path, _segment_from_meta)
from transcription.common import (
    _LoadedFile, _asr_model_name, _checked_transcribe, _dump_asr_texts, _lid_is_dialect,
    lid_unsupported)
from transcription.dialogue_windows import (
    _finalize_dialogue_only, _finalize_dialogue_windows, _has_phase1_windows, rescue_enabled)
from transcription.failures import _record_phase2_fail
from utils.tool import get_char_count


_LEDGER_DIRNAME = "_shorts_ledger"


def _ledger_dir(sid_dir: str) -> str:
    coord = os.environ.get("EMILIA2_DATA_COORD_DIR")
    if coord:
        return os.path.join(coord, _LEDGER_DIRNAME)
    root = os.path.dirname(os.path.dirname(os.path.abspath(sid_dir)))
    return os.path.join(root, _LEDGER_DIRNAME)


def _append_shorts_ledger(sid_dir: str, sid: str, rows: list, secondary: bool) -> None:
    if not rows:
        return
    secs: dict[str, float] = {}
    for r in rows:
        try:
            d = float(r.get("end", 0)) - float(r.get("start", 0))
        except (TypeError, ValueError):
            continue
        if d <= 0:
            continue
        lang = (r.get("language") or "unknown")
        secs[lang] = round(secs.get(lang, 0.0) + d, 3)
    if not secs:
        return
    try:
        led = _ledger_dir(sid_dir)
        os.makedirs(led, exist_ok=True)
        path = os.path.join(led, f"{socket.gethostname()}_{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"sid": sid, "sec": secs, "ts": time.time(),
                 "pass": "secondary" if secondary else "primary"},
                ensure_ascii=False) + "\n")
    except OSError:
        pass


def _load_all_shorts_rows(partial: str) -> list[dict]:
    p = partial.replace(".partial.json", ".all_shorts.json")
    if p != partial:
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    with open(partial) as f:
        return json.load(f)


def _load_file_for_asr(
    partial: str, final: str, asr: ASRModel, secondary: bool,
) -> _LoadedFile | None:
    """CPU/IO half of Phase-2 for one file (prefetchable; touches no GPU).

    Returns ``None`` to skip the file entirely, or a :class:`_LoadedFile`
    (``write_empty=True`` for the empty-final marker case, else carrying the
    bundle + relative segments to infer). ``prune`` is the driver's policy, set
    on the returned object by the caller.
    """
    sid_dir = os.path.dirname(partial)

    rows = _load_all_shorts_rows(partial)

    rows = [r for r in rows if not lid_unsupported(r.get("language"))]

    def _mk(**kw) -> _LoadedFile:
        return _LoadedFile(
            partial=partial, final=final, sid_dir=sid_dir,
            secondary=secondary, **kw,
        )

    if not rows:
        # Phase 1 produced no surviving segments. Write the empty final so
        # resume marks it done — unless secondary already has a final.

        if secondary and os.path.exists(final):
            return None
        return _mk(write_empty=True)

    # ---- LID-based routing (S4) ----
    existing_rows: list[dict] = []
    if secondary:
        if os.path.exists(final):
            with open(final) as f:
                existing_rows = json.load(f)
            covered = {r.get("index") for r in existing_rows if r.get("index")}
            rows = [r for r in rows if r.get("index") not in covered]
        rows = [r for r in rows if asr.accepts((r.get("language") or "").strip().lower())]
        if not rows:
            return None
    else:
        rows = [r for r in rows
                if (lid := (r.get("language") or "").strip().lower()) in ("", "unknown")
                or asr.accepts(lid)]
        if not rows:
            # Nothing for the primary ASR here (all indic-only LID). Write an
            # empty final so resume marks the primary short-track pass done;
            # the secondary pass merges its rows in afterwards.
            if os.path.exists(final):
                return _mk(dialogue_only=True) if _has_phase1_windows(sid_dir) else None
            return _mk(write_empty=True)

    segments = [_segment_from_meta(r) for r in rows]

    # Concatenate the per-segment WAVs into one buffer; rel_segments carry
    # start/end within that buffer (the ASR slices it per segment).
    clips: list[np.ndarray] = []
    sample_rate = 0
    cursor = 0.0
    rel_segments: list[Segment] = []
    loaded = _read_segments_parallel(
        rows, sid_dir, target_sr=int(getattr(asr, "SR", 0) or 0) or None,
    )
    for seg, (wav, sr) in zip(segments, loaded):
        if sample_rate == 0:
            sample_rate = sr
        elif sr != sample_rate:
            raise RuntimeError(
                f"mixed sample rates in {partial}: {sr} vs {sample_rate}"
            )
        dur = len(wav) / sr
        clips.append(wav)
        rel_segments.append(Segment(
            start=cursor, end=cursor + dur,
            speaker=seg.speaker, index=seg.index,
            language=seg.language if seg.language and seg.language != "unknown" else None,
            extra=dict(seg.extra),
        ))
        cursor += dur

    big_buffer = np.concatenate(clips).astype(np.float32, copy=False)
    bundle = AudioBundle(waveform=big_buffer, sample_rate=sample_rate, name="phase2")
    return _mk(
        segments=segments, rows=rows, rel_segments=rel_segments,
        bundle=bundle, sample_rate=sample_rate, existing_rows=existing_rows,
    )


def _infer_one(asr: ASRModel, lf: _LoadedFile, batch_size: int, logger) -> None:
    """Single-file GPU step: bucket by language hint, one transcribe per bucket.

    Fills ``lf.transcribed`` (position i -> Segment). Stage 2's cross-file
    variant fills the same map.
    """
    buckets: dict[str | None, list[int]] = defaultdict(list)
    for i, rs in enumerate(lf.rel_segments):
        buckets[rs.language].append(i)
    for lang, idxs in buckets.items():
        subs = [lf.rel_segments[i] for i in idxs]
        out = _checked_transcribe(
            asr, lf.bundle, subs, language=lang, batch_size=batch_size,
            context=f"{lf.partial} lang={lang!r}",
        )
        for i, t in zip(idxs, out):
            lf.transcribed[i] = t


def _infer_batch(
    asr: ASRModel, lfs: list[_LoadedFile], batch_size: int, logger,
) -> None:
    """Stage 2: one ASR pass over SEVERAL files' segments at once.

    Concatenates each file's audio — already resampled to the adapter's working
    SR (``asr.SR``) — into one bundle, then offsets every file's segments into
    that shared timeline and buckets the whole lot by language for a single
    ``transcribe`` call per language. This keeps the vLLM engine fed across file
    boundaries instead of one small per-file batch at a time.

    **Output is bit-identical to ``_infer_one``.** Each file is resampled exactly
    as the per-file path would (same source buffer, same target SR, same cache),
    and the ``+0.5``-sample offset cancels float truncation, so for every segment
    the adapter slices precisely the same ``[ps:pe]`` window of that file's
    resampled buffer. The only thing that changes is how many segments share one
    ``transcribe`` call. (Relies on the adapter slicing ``int(seg.start*asr.SR)``,
    which qwen3 / indic both do.)

    Falls back to per-file when the adapter doesn't expose ``SR`` (can't pin the
    bundle SR) or the batch is a single file. Fills each ``lf.transcribed``.
    """
    target_sr = int(getattr(asr, "SR", 0) or 0)
    if target_sr <= 0 or len(lfs) == 1:
        for lf in lfs:
            _infer_one(asr, lf, batch_size, logger)
        return
    # ``get_at_sr(target_sr)`` resamples each file's buffer to target_sr from its
    # own source SR, so files with different source SRs can share one mega-bundle
    # without grouping — they're all at target_sr once concatenated.
    buffers: list[np.ndarray] = []
    mega_segs: list[Segment] = []
    routing: list[tuple] = []  # parallel to mega_segs: (lf, local_i)
    cursor = 0                 # running sample offset at target_sr
    for lf in lfs:
        buf = lf.bundle.get_at_sr(target_sr).astype(np.float32, copy=False)
        for li, rs in enumerate(lf.rel_segments):
            ps = int(rs.start * target_sr)
            pe = int(rs.end * target_sr)
            mega_segs.append(Segment(
                start=(cursor + ps + 0.5) / target_sr,
                end=(cursor + pe + 0.5) / target_sr,
                speaker=rs.speaker, index=rs.index,
                language=rs.language, extra=dict(rs.extra),
            ))
            routing.append((lf, li))
        buffers.append(buf)
        cursor += len(buf)
    mega = AudioBundle(
        waveform=np.concatenate(buffers).astype(np.float32, copy=False),
        sample_rate=target_sr, name="phase2_batch",
    )
    buckets: dict[str | None, list[int]] = defaultdict(list)
    for gi, ms in enumerate(mega_segs):
        buckets[ms.language].append(gi)
    for lang, gidxs in buckets.items():
        subs = [mega_segs[gi] for gi in gidxs]
        out = _checked_transcribe(
            asr, mega, subs, language=lang, batch_size=batch_size,
            context=f"phase2 batch lang={lang!r} files={len(lfs)}",
        )
        for gi, t in zip(gidxs, out):
            lf, li = routing[gi]
            lf.transcribed[li] = t


def _finalize_file(
    asr: ASRModel, lf: _LoadedFile, min_char: int, ratio_filter, filter_cfg, logger,
) -> None:
    """CPU/IO half: stitch ``lf.transcribed`` back, apply the 4 post-ASR
    filters, merge (secondary), atomic-write ``<sid>.json``, optional prune."""
    segments, rows = lf.segments, lf.rows
    zh_en_dnsmos_floor = float(
        filter_cfg.min_quality_overrides.get(
            "zh", filter_cfg.min_quality_overrides.get(
                "en", filter_cfg.min_quality,
            ),
        )
    )
    drop_fake_zh_en = bool(filter_cfg.drop_fake_zh_en)
    dialect_dnsmos_floor = float(filter_cfg.dialect_dnsmos_floor)

    final_rows: list[dict] = []
    kept = drop_by_lang = drop_by_text = drop_by_ratio = 0
    drop_by_post_asr_zh_en = drop_by_content = 0
    for i, (orig_seg, orig_row) in enumerate(zip(segments, rows)):
        if not orig_row.get("kept", True):
            continue
        t = lf.transcribed.get(i)
        if t is None:
            drop_by_text += 1
            continue

        detected = t.language or orig_seg.language
        text = (t.text or "").strip()
        st = asr_texts.SegText(
            text=text, language=detected,
            accepted=bool(asr.accepts(detected)),
            asr_model=_asr_model_name(asr), extra=dict(t.extra or {}),
        )
        seg_dnsmos = orig_seg.quality
        if seg_dnsmos is None:
            seg_dnsmos = orig_row.get("dnsmos")
        reason = asr_texts.short_verdict(
            st, seg_dnsmos,
            orig_seg.language,
            orig_seg.extra.get("lid_raw") or orig_row.get("lid_raw"),
            bool(orig_seg.extra.get("is_fake")) or bool(orig_row.get("is_fake")),
            float(orig_seg.extra.get("speech_s") or (orig_seg.end - orig_seg.start)),
            min_char=min_char, ratio_filter=ratio_filter,
            zh_en_floor=zh_en_dnsmos_floor, dialect_floor=dialect_dnsmos_floor,
            drop_fake_zh_en=drop_fake_zh_en, char_count=get_char_count,
            lid_is_dialect=_lid_is_dialect,
        )
        if reason is not None:
            if reason == "lang":
                drop_by_lang += 1
            elif reason in ("text", "no_result"):
                drop_by_text += 1
            elif reason == "post_asr_zh_en":
                drop_by_post_asr_zh_en += 1
            elif reason == "ratio":
                drop_by_ratio += 1
            else:
                drop_by_content += 1
            continue

        orig_seg.text = text
        orig_seg.language = detected
        _asr_raw = t.extra.get("asr_lang_raw")
        if _asr_raw:
            orig_seg.extra["asr_lang_raw"] = _asr_raw

        out_row = orig_seg.to_legacy_dict()
        for k, v in orig_row.items():
            if k not in out_row:
                out_row[k] = v
        final_rows.append(out_row)
        kept += 1

    logger.info(
        f"{os.path.basename(lf.final)}{' [secondary]' if lf.secondary else ''}: "
        f"kept={kept} drop_lang={drop_by_lang} drop_text={drop_by_text} "
        f"drop_ratio={drop_by_ratio} drop_post_asr_zh_en={drop_by_post_asr_zh_en} "
        f"drop_content={drop_by_content} in={len(segments)}"
    )

    own_rows = final_rows
    if lf.secondary:
        merged = lf.existing_rows + final_rows
        merged.sort(key=lambda r: r.get("start", 0))
        final_rows = merged

    _dump_asr_texts(asr, lf, segments, rows)

    _finalize_dialogue_windows(asr, lf, final_rows, ratio_filter, logger,
                               rescue=rescue_enabled())

    atomic_dump(final_rows, lf.final)

    _append_shorts_ledger(lf.sid_dir, os.path.basename(lf.final)[:-len(".json")],
                          own_rows, lf.secondary)

    # Delete loose audio of segments THIS pass DROPPED (not in final_rows) so
    # ASR-empty segments never accumulate as orphans.
    # SAFE: phase2 primary/secondary process DISJOINT phase1-LID partitions
    # (primary = ""/unknown/qwen3-accepted; secondary = indic-accepted), so a
    # segment one pass drops is never wanted by the other. Kept segments (in
    # final_rows — reuse anchors + packing sources) are preserved. Phase3 never
    # needs a dropped seg's loose audio: 3a reads all_shorts metadata, 3b anchors
    # ONLY on reuse (=phase2-kept) segments, 3c uses the window flac. Source tars
    # are read-only → reversible via re-run. Match by index (unambiguous) and
    # refuse to unlink anything outside the sid dir.
    kept_idx = {r.get("index") for r in final_rows}
    sid_dir_abs = os.path.abspath(lf.sid_dir)
    for r in rows:
        if r.get("index") in kept_idx:
            continue
        ap = _own_audio(r)
        if not ap:
            continue
        p = os.path.abspath(_resolve_path(ap, lf.sid_dir))
        if os.path.dirname(p) != sid_dir_abs:   # never escape the sid dir
            continue
        try:
            os.remove(p)
        except OSError:
            pass

    if lf.prune:
        for r in rows:
            ap = _own_audio(r)
            if not ap:
                continue
            p = _resolve_path(ap, lf.sid_dir)
            try:
                os.remove(p)
            except OSError:
                pass


def _iter_prefetched(work, load_fn, depth):
    """Yield ``(item, future)`` in completion order, ≤ ``depth`` loads in flight.

    A small thread pool runs ``load_fn(item)`` (the CPU/IO ``_load_file_for_asr``)
    ahead of the main thread's GPU inference, overlapping NAS reads with the ASR
    call. ``depth`` bounds both the pool size and the number of in-flight loaded
    buffers, so memory stays bounded regardless of the work-list length.
    """
    work_it = iter(work)
    with ThreadPoolExecutor(max_workers=depth, thread_name_prefix="p2load") as ex:
        inflight: dict = {}

        def _submit_next() -> bool:
            try:
                item = next(work_it)
            except StopIteration:
                return False
            inflight[ex.submit(load_fn, item)] = item
            return True

        for _ in range(depth):
            if not _submit_next():
                break
        while inflight:
            done, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                item = inflight.pop(fut)
                _submit_next()  # keep the pipeline full
                yield item, fut


# Internal pacing for _run_short_track (not caller knobs): files between
# lease-lock heartbeats, and between gc.collect()+cuda.empty_cache() sweeps.
_HEARTBEAT_EVERY = 20
_CLEANUP_EVERY = 32


def _run_short_track(
    work, asr: ASRModel, *, batch_size: int, min_char: int, ratio_filter,
    filter_cfg, prune_segments: bool, secondary: bool, logger,
    failed_log_path: str, heartbeat=None, desc: str = "tx",
    prefetch: int = 4, batch_files: int = 1,
) -> None:
    """Drive the short track over a prefix's ``work`` list.

    Stage 1 (``prefetch``): file I/O (``_load_file_for_asr``) runs on a small
    thread pool ahead of the GPU ASR call so NAS reads of the next file overlap
    inference. Stage 2 (``batch_files`` > 1): up to ``batch_files`` loaded files
    share ONE batched ASR call (``_infer_batch``) so the engine stays fed across
    file boundaries; ``batch_files == 1`` infers one file at a time
    (``_infer_one``). Both produce per-file output bit-identical to the
    sequential path.

    Resume stays per-file: every file is atomic-written on its own, so a crash
    mid-batch just redoes that batch's files next run. Failure isolation: a load
    error, a whole-batch infer error (falls back to per-file infer), or a
    finalize error are all logged to ``_failed_phase2.jsonl`` without killing the
    loop. CUDA cache is cleared every ``_CLEANUP_EVERY`` files, not per-file.
    """
    work = list(work)
    if not work:
        return
    batch_files = max(1, batch_files)
    depth = max(1, prefetch, batch_files)  # keep enough in flight to fill a batch

    def load_fn(item):
        return _load_file_for_asr(item[0], item[1], asr, secondary)

    pending: list = []  # loaded files awaiting inference (flushed as a batch)

    def _flush() -> None:
        if not pending:
            return
        successful: list = []
        if len(pending) == 1:
            lf = pending[0]
            try:
                _infer_one(asr, lf, batch_size, logger)
                successful.append(lf)
            except Exception as e:  # noqa: BLE001 - leave final absent for retry
                logger.exception(f"per-file infer failed {lf.partial}: {e}")
                _record_phase2_fail(failed_log_path, lf.partial, e)
        else:
            try:
                _infer_batch(asr, pending, batch_size, logger)
                successful.extend(pending)
            except Exception as e:  # noqa: BLE001 - batch -> isolated retries
                logger.exception(
                    f"batched infer failed ({len(pending)} files); per-file retry: {e}"
                )
                for lf in pending:
                    lf.transcribed.clear()
                    try:
                        _infer_one(asr, lf, batch_size, logger)
                        successful.append(lf)
                    except Exception as e2:  # noqa: BLE001
                        logger.exception(f"per-file infer failed {lf.partial}: {e2}")
                        _record_phase2_fail(failed_log_path, lf.partial, e2)
        for lf in successful:
            try:
                _finalize_file(asr, lf, min_char, ratio_filter, filter_cfg, logger)
            except Exception as e:  # noqa: BLE001 - keep the rest of the batch
                logger.exception(f"finalize failed on {lf.partial}: {e}")
                _record_phase2_fail(failed_log_path, lf.partial, e)
        pending.clear()

    n = 0
    pbar = tqdm.tqdm(total=len(work), desc=desc)
    try:
        for (partial, _final), fut in _iter_prefetched(work, load_fn, depth):
            try:
                lf = fut.result()
                if lf is None:
                    pass
                elif lf.dialogue_only:
                    try:
                        _finalize_dialogue_only(asr, lf, ratio_filter, logger)
                    except Exception as e:  # noqa: BLE001
                        logger.exception(f"dialogue-only finalize failed {lf.partial}: {e}")
                        _record_phase2_fail(failed_log_path, lf.partial, e)
                elif lf.write_empty:
                    lf.prune = prune_segments

                    if _has_phase1_windows(lf.sid_dir):
                        _finalize_dialogue_windows(
                            asr, lf, [], ratio_filter, logger,
                        )
                    # Resume marker is last; dialogue failure remains retryable.
                    atomic_dump([], lf.final)
                else:
                    lf.prune = prune_segments
                    pending.append(lf)
                    if len(pending) >= batch_files:
                        _flush()
            except Exception as e:  # noqa: BLE001 - load failure for this file
                logger.exception(f"transcribe failed on {partial}: {e}")
                _record_phase2_fail(failed_log_path, partial, e)
            n += 1
            pbar.update(1)
            if n % _CLEANUP_EVERY == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if heartbeat and n % _HEARTBEAT_EVERY == 0:
                heartbeat()
        _flush()  # tail batch (< batch_files)
    finally:
        pbar.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
