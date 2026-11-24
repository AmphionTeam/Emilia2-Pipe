"""Transcribe the dialogue windows Phase 1 selected, reusing short-track text where valid."""

from __future__ import annotations

import json
import os

import numpy as np

import asr_texts
from dialogue import phase2 as dialogue_phase2
from dialogue.select import Gates
from pipeline.io import atomic_dump
from pipeline.stages.base import ASRModel
from pipeline.types import AudioBundle, Segment
from transcription.carrier_io import _read_carrier_windows
from transcription.common import _LoadedFile, _checked_transcribe


DIALOGUE_MIN_CHAR = 1  # a dialogue turn needs at least this many characters
# Set per run by configure().
_DIALOGUE_RESCUE = False
_DIALOGUE_GATES = Gates()


def _dialogue_paths(lf: _LoadedFile) -> tuple[str, str]:
    def _sib(suffix: str) -> str:
        p = lf.partial.replace(".partial.json", suffix)
        return p if p != lf.partial else os.path.join(
            lf.sid_dir, os.path.basename(lf.sid_dir) + suffix)
    return _sib(".dialogue_chunk.json"), _sib(".dialogue.json")


def _has_phase1_windows(sid_dir: str) -> bool:
    sid = os.path.basename(os.path.normpath(sid_dir))
    try:
        with open(os.path.join(sid_dir, f"{sid}.dialogue_chunk.json")) as f:
            wins = json.load(f)
    except (OSError, ValueError):
        return False
    return any(dialogue_phase2.is_phase1_materialized(w) for w in wins)


def _finalize_dialogue_only(asr: ASRModel, lf: _LoadedFile, ratio_filter, logger) -> None:
    try:
        with open(lf.final) as f:
            final_rows = json.load(f)
    except (OSError, ValueError):
        final_rows = []
    _finalize_dialogue_windows(asr, lf, final_rows, ratio_filter, logger,
                               rescue=_DIALOGUE_RESCUE)


def _finalize_dialogue_windows(
    asr: ASRModel, lf: _LoadedFile, final_rows: list[dict], ratio_filter, logger,
    rescue: bool = False,
) -> None:
    if lf.secondary:
        return
    src, dst = _dialogue_paths(lf)

    try:
        with open(src) as f:
            wins = json.load(f)
    except (OSError, ValueError):
        return
    if not wins:
        atomic_dump([], dst)
        return
    if not any(dialogue_phase2.is_phase1_materialized(w) for w in wins):
        return

    text_map = {r["index"]: r["text"] for r in final_rows
                if r.get("index") and r.get("text")}
    part_idx = {r.get("index") for r in lf.rows}
    gates = _DIALOGUE_GATES

    sid = os.path.basename(lf.partial)[: -len(".partial.json")]
    table = asr_texts.load(asr_texts.path_for(lf.sid_dir, sid))

    kept: list[dict] = []
    n_from_table = 0
    todo: list[tuple[dict, list[dict]]] = []     # (window, units this pass transcribes)
    for w in wins:
        if not dialogue_phase2.is_phase1_materialized(w):
            kept.append(w)
            continue

        dialogue_phase2.prepare(w, text_map, part_idx, gates, final_pass=False)
        if table:
            n_from_table += dialogue_phase2.fill_from_table(
                w, table, min_char=DIALOGUE_MIN_CHAR, ratio_filter=ratio_filter)

        units = dialogue_phase2.pending_units(w)
        if units:
            todo.append((w, units))
        kept.append(w)

    if todo:
        _transcribe_dialogue_units(asr, lf, todo, ratio_filter, logger)

    if rescue:
        if table:
            for w in kept:
                if dialogue_phase2.is_phase1_materialized(w):
                    n_from_table += dialogue_phase2.rescue_from_table(w, table)
        todo_rescue = [(w, t) for w in kept
                       if dialogue_phase2.is_phase1_materialized(w)
                       for t in [dialogue_phase2.rescue_targets(w)] if t]
        if todo_rescue:
            _rescue_dialogue_units(asr, lf, todo_rescue, logger)

    out: list[dict] = []
    for w in kept:
        if not dialogue_phase2.is_phase1_materialized(w):
            out.append(w)
            continue

        for c in dialogue_phase2.split_window(w, gates):
            dialogue_phase2.finalize(c)
            out.append(c)
    if n_from_table:
        logger.info(f"dialogue: {n_from_table} units from asr_texts table (no ASR)")
    atomic_dump(out, dst)


def _transcribe_dialogue_units(
    asr: ASRModel, lf: _LoadedFile, todo: list, ratio_filter, logger,
) -> None:
    target_sr = int(getattr(asr, "SR", 0) or 0) or None
    win_rows = [{"sample_rate": w["sample_rate"], "carrier_path": w["carrier_path"],
                 "carrier_start_samples": w["carrier_start_samples"],
                 "carrier_end_samples": w["carrier_end_samples"]} for w, _ in todo]
    loaded = _read_carrier_windows(win_rows, lf.sid_dir, target_sr=target_sr)

    n_ok = n_unit = 0
    for (w, units), (wav, sr) in zip(todo, loaded):
        bundle = AudioBundle(
            waveform=wav.astype(np.float32, copy=False), sample_rate=sr,
            name=w["index"])
        w0 = w["start"]

        rel = [Segment(start=u["start"] - w0, end=u["end"] - w0,
                       speaker=u["speaker"], index="|".join(u["seg_indices"]),
                       language=None, extra={}) for u in units]
        checked = _checked_transcribe(
            asr, bundle, rel, language=None, batch_size=len(rel),
            context=f"dialogue {w['index']}",
        )
        res = {r.index: r for r in checked}
        for u in units:
            n_unit += 1
            if dialogue_phase2.apply_result(
                    u, res.get("|".join(u["seg_indices"])), asr,
                    DIALOGUE_MIN_CHAR, ratio_filter):
                n_ok += 1
    if n_unit:
        logger.info(f"dialogue units: {n_ok}/{n_unit} ok "
                    f"({len(todo)} windows, 1 carrier decode)")


def _rescue_dialogue_units(
    asr: ASRModel, lf: _LoadedFile, rescue: list, logger,
) -> None:
    target_sr = int(getattr(asr, "SR", 0) or 0) or None
    win_rows = [{"sample_rate": w["sample_rate"], "carrier_path": w["carrier_path"],
                 "carrier_start_samples": w["carrier_start_samples"],
                 "carrier_end_samples": w["carrier_end_samples"]} for w, _ in rescue]
    loaded = _read_carrier_windows(win_rows, lf.sid_dir, target_sr=target_sr)

    n_ok = n_try = 0
    for (w, units), (wav, sr) in zip(rescue, loaded):
        bundle = AudioBundle(
            waveform=wav.astype(np.float32, copy=False), sample_rate=sr,
            name=w["index"])
        w0 = w["start"]
        rel = [Segment(start=max(0.0, u["start"] - w0), end=u["end"] - w0,
                       speaker=u.get("speaker"),
                       index="|".join(u.get("seg_indices") or [str(i)]),
                       language=None, extra={})
               for i, u in enumerate(units)]
        checked = _checked_transcribe(
            asr, bundle, rel, language=None, batch_size=len(rel),
            context=f"dialogue rescue {w['index']}",
        )
        res = {r.index: r for r in checked}
        for u, s in zip(units, rel):
            n_try += 1
            if dialogue_phase2.apply_rescue(u, res.get(s.index)):
                n_ok += 1
    if n_try:
        logger.info(f"dialogue rescue: recovered {n_ok}/{n_try} "
                    f"({len(rescue)} windows)")


def configure(*, rescue: bool, gates: Gates) -> None:
    """Set the run's dialogue policy (``--dialogue-rescue`` and the config's gates)."""
    global _DIALOGUE_RESCUE, _DIALOGUE_GATES
    _DIALOGUE_RESCUE = rescue
    _DIALOGUE_GATES = gates


def rescue_enabled() -> bool:
    return _DIALOGUE_RESCUE
