"""Phase 2 — batched ASR over the partial manifests Phase 1 produced.

Phase 1 (``main.py``) walks each audio file through standardize → separator
→ diarizer → VAD → DNSMOS → quality-filter → LID and lands each surviving
recording as one M4A carrier plus a ``<sid>.partial.json`` manifest under
``<input>_processed/<sid>/``.

This script:

1. Scans ``<input>_processed/*/<sid>.partial.json`` for files not yet
   transcribed (skip when ``<sid>.json`` already exists).
2. Decodes each manifest's shared carrier once and slices one big batch indexed by
   ``(sid, seg_idx)``.
3. Runs the configured ASR adapter (default ``qwen3_asr``) on the batch.
   The ``language=`` hint per segment comes from the partial — if Phase 1
   tagged it ``"unknown"`` (LID confidence below threshold) the adapter
   sees ``None`` and auto-detects.
4. Applies the adapter's ``accepts(language)`` rule against ASR's
   *output* language. Utterances whose resolved language is unknown,
   unsupported, or explicitly skip-listed (e.g. ``ms``/``id`` for
   Qwen3-ASR) are dropped here.
5. Applies the text-length filter (``filter.min_char_count``) — the part
   of the legacy ``calculate_audio_stats`` that needs text.
6. Writes ``<sid>.json`` per file with the final segment list, augmented
   with ``text`` / ``language`` / ASR provenance. Per-segment WAVs are
   retained (re-runnable) unless ``--prune-segments`` is passed.

CLI:

    python transcribe.py --config_path config.json
    python transcribe.py --config_path config.json --processed-root processed
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import time
import warnings

import torch

from dialogue import phase2 as dialogue_phase2
from pipeline.config import PipelineConfig
from pipeline.io import nodes, work_queue
from pipeline.registry import build_adapter
from pipeline.stages.base import ASRModel
from transcription import dialogue_windows, long_track
from transcription.common import lid_unsupported
from transcription.discovery import (
    _claim_lock, _mark_drained, _prefix_dirs, _prefix_drained, _prefix_fingerprint,
    _queue_prefix, _scan_long_chunks, _scan_partials, _scan_prefix)
from transcription.failures import (
    _P2_MAX_FAILS, _poisoned, _poisoned_all, _record_phase2_fail, _root_poison, _touch)
from transcription.long_track import _transcribe_long_chunks
from transcription.short_track import _run_short_track
from utils.logger import Logger
from utils.tool import detect_gpu

warnings.filterwarnings("ignore")


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dialogue-rescue", action="store_true",
        help="enable permissive dialogue rescue for existing data; accept any "
             "nonempty transcription. Disabled by default",
    )
    ap.add_argument(
        "--config_path", type=str, default="config.json",
        help="Same config Phase 1 used. Phase 2 reads ``stages.asr`` "
             "(must be set), ``filter.min_char_count``, and ``runtime`` knobs.",
    )
    ap.add_argument(
        "--processed-root", type=str, default="",
        help="Override the directory holding ``<sid>/<sid>.partial.json`` "
             "files. Defaults to ``<input_folder_path>_processed`` derived "
             "from the config.",
    )
    ap.add_argument(
        "--prune-segments", action="store_true",
        help="Delete per-segment WAVs of files that finish Phase 2 "
             "successfully. Off by default so failures stay re-runnable.",
    )
    ap.add_argument(
        "--secondary", action="store_true",
        help=(
            "Fill-in pass: don't skip files where ``<sid>.json`` already "
            "exists. Instead, for each ``partial.json`` row whose index "
            "isn't already in ``<sid>.json``, run the configured ASR if "
            "its ``accepts()`` returns True for that row's language, and "
            "merge the resulting rows into ``<sid>.json`` (re-sorted by "
            "start time). Intended workflow: run primary pass with one "
            "ASR (e.g. Qwen3-ASR), then run secondary pass with another "
            "(e.g. IndicConformer for Bengali) so each ASR handles only "
            "the languages it knows."
        ),
    )
    ap.add_argument(
        "--loop", action="store_true",
        help=(
            "Synchronous-pipeline mode: load vLLM ONCE, then repeatedly scan "
            "for new partials and transcribe them, consuming Phase-1's output "
            "as it streams in. Idles when there's no new work; exits when "
            "``_PHASE1_DONE`` exists AND the backlog is drained. Without this "
            "flag, runs a single pass over existing partials and exits "
            "(use that after Phase-1 has fully finished)."
        ),
    )
    ap.add_argument(
        "--phase1-nodes", type=int, default=1,
        help=(
            "minimum number of phase-one nodes to wait for in --loop mode. "
            "RUN.json provides the fixed topology when present; this value "
            "supports older output trees without that file."
        ),
    )
    ap.add_argument(
        "--num-shards", type=int, default=1,
        help=(
            "Split the start-up check for pending work across this many "
            "instances (default 1: check the whole tree). With --shard, this "
            "instance checks only the top-level dirs where int(name, 16) %% "
            "num-shards == shard (md5 of the name when it is not hex), and exits "
            "before loading the ASR model if none of them has pending work. This "
            "does not split the transcription: a running instance takes any "
            "unclaimed prefix, and per-prefix claim locks stop two instances from "
            "transcribing the same one. --loop, --secondary, --no-prescan, batch "
            "and queue runs skip the start-up check, so the shard has no effect there."
        ),
    )
    ap.add_argument(
        "--shard", type=int, default=0,
        help="This instance's index in [0, num-shards) for the start-up check. "
             "See --num-shards.",
    )
    ap.add_argument(
        "--prefetch", type=int, default=4,
        help=(
            "Stage 1: how many files' I/O (read partial + segment WAVs, build "
            "the bundle) to prefetch on a background thread pool ahead of the "
            "GPU ASR call, so NAS reads overlap inference. 0/1 = one loader "
            "(still one-ahead); higher hides more NAS latency at more memory. "
            "Default 4."
        ),
    )
    ap.add_argument(
        "--batch-files", type=int, default=1,
        help=(
            "Stage 2: how many files' segments to feed the ASR in ONE batched "
            "call (so the engine stays fed across file boundaries instead of a "
            "small per-file batch). 1 (default) = one file per call (= Stage 1 "
            "only). Output is bit-identical regardless; this only trades a "
            "larger in-flight memory footprint for higher GPU utilization. Try "
            "4-16; set proportional to your KV-cache / memory budget."
        ),
    )
    ap.add_argument(
        "--batch-base", type=str, default="",
        help=(
            "batch-mode parent directory containing independent phase-one output "
            "trees. Phase two scans unfinished batches and writes _PHASE2_DONE "
            "when a sealed batch has no pending work. Mutually exclusive with "
            "--processed-root; phase one uses --batch-recordings"
        ),
    )
    ap.add_argument(
        "--sealed-only", action="store_true",
        help=(
            "in batch mode, process only batches marked _SEALED. A single-pass "
            "run exits after all sealed batches are complete; newly sealed batches "
            "are picked up during the next scan"
        ),
    )
    ap.add_argument(
        "--use-queue", action="store_true",
        help=(
            "discover phase-two work from the optional phase-one queue. Disabled "
            "by default; falls back to scanning when the queue is not enabled"
        ),
    )
    ap.add_argument(
        "--no-prescan", action="store_true",
        help=(
            "Skip the single-shot pre-warmup full-tree scan (``_scan_partials`` "
            "/ ``_scan_long_chunks``) that decides whether to bother loading the "
            "ASR. That scan globs every prefix + stats every partial across the "
            "whole processed tree — slow on NAS and redundant when N instances "
            "each do it. The per-prefix claim loop already discovers work "
            "incrementally (1× total, distributed), so for a single-pass cycle "
            "over a huge tree, skip the pre-scan and go straight to warmup + one "
            "claim round. Cost: warmup runs even if there turns out to be no "
            "work. No effect with --loop or --secondary (they skip it already)."
        ),
    )
    ap.add_argument(
        "--tracks", choices=("both", "short", "long"), default="both",
        help=(
            "Which track(s) to clean. ``both`` (default) transcribes shorts "
            "(<sid>.json) and longs (<sid>.long.json). ``short`` only does the "
            "short track; ``long`` only the long track. Resume is independent "
            "per track, so you can run the two separately (even on different "
            "GPUs/instances). Applies identically to the --secondary (indic) "
            "pass."
        ),
    )
    return ap


def _check_args(args: argparse.Namespace) -> None:
    """Reject flag combinations that Phase 2 cannot honour."""
    if args.num_shards < 1 or not (0 <= args.shard < args.num_shards):
        raise SystemExit(
            f"bad sharding: --shard {args.shard} --num-shards {args.num_shards} "
            "(need num_shards >= 1 and 0 <= shard < num_shards)"
        )
    if args.batch_base and args.secondary:
        raise SystemExit(
            "--batch-base cannot be used with --secondary: _PHASE2_DONE does not "
            "track separate passes. Run secondary against one output tree with "
            "--processed-root."
        )
    if args.batch_base and args.tracks != "both":
        raise SystemExit(
            "--batch-base requires --tracks both: _PHASE2_DONE does not track "
            "short and long tracks separately."
        )
    if args.sealed_only and not args.batch_base:
        raise SystemExit("--sealed-only requires --batch-base")


def _processed_root(args: argparse.Namespace, cfg: PipelineConfig) -> str:
    """The batch base, or the single Phase 1 output tree to transcribe."""
    if args.batch_base:
        root = args.batch_base
        if not os.path.isdir(root):
            raise SystemExit(f"batch base not found: {root}")
        return root
    root = args.processed_root or cfg.io.input_folder_path.rstrip("/") + "_processed"
    if not os.path.isdir(root):
        raise SystemExit(f"processed root not found: {root}")
    return root


def _prescan_finds_work(args: argparse.Namespace, processed_root: str, logger) -> bool:
    """Whether a single-shot primary pass has anything to do before ASR warmup.

    Worth it for the PRIMARY pass: the scan is cheap next to the qwen3 warmup
    it guards. NOT worth it for the SECONDARY (indic) pass: a full pre-scan
    would walk every partial.json in the dataset, so N concurrent instances each
    pay an N× redundant full-tree scan (tens of minutes) just to answer "is
    there any work?", which the claim loop already discovers per prefix
    (distributed across instances, 1× total). Secondary, ``--loop``,
    ``--no-prescan``, batch and queue runs therefore skip it.
    """
    if args.loop or args.secondary or args.no_prescan:
        return True
    if args.batch_base:
        logger.info("batch mode: skip full-tree prescan; batches discover work")
        return True
    if args.use_queue and work_queue.is_enabled(processed_root):
        logger.info("queue mode: skip full-tree prescan; the main loop discovers work")
        return True
    shorts = (_scan_partials(processed_root, args.shard, args.num_shards)
              if args.tracks in ("both", "short") else [])
    longs = (_scan_long_chunks(processed_root, args.shard, args.num_shards)
             if args.tracks in ("both", "long") else [])
    if not shorts and not longs:
        logger.info(f"nothing to do under {processed_root}")
        return False
    return True


LOOP_SLEEP = 30    # seconds to idle when no new partials yet
LEASE_S = 1800.0   # prefix-claim lease; a lock older than this is stolen


class _Phase2Run:
    """Drain every pending prefix, round after round, until there is no work left.

    Dynamic prefix claiming (self-balancing, elastic): each instance claims one
    fan-out prefix dir at a time via an O_EXCL lease lock, drains its backlog,
    releases, and moves on. Fast instances naturally claim more prefixes; a
    slow/remote node claims fewer; a killed instance's lock is stolen after the
    lease. Scans stay cheap (one ~1/256 prefix per claim, not the whole tree)
    and instances never duplicate each other. Add/stop/move instances freely;
    there is no fixed partition to reconfigure. Primary and secondary (indic)
    use separate claim dirs so they don't block each other.
    """

    def __init__(self, args: argparse.Namespace, cfg: PipelineConfig, asr: ASRModel,
                 logger, processed_root: str) -> None:
        self.args = args
        self.cfg = cfg
        self.asr = asr
        self.logger = logger
        self.processed_root = processed_root
        self.batch_base = args.batch_base or ""
        if self.batch_base:
            from pipeline.io import batches
            self.batches = batches
        self._states: dict = {}

    def phase1_done(self) -> bool:
        return nodes.phase1_done(self.processed_root, min_nodes=self.args.phase1_nodes)

    def active_roots(self) -> list:
        """``(batch index, root)`` pairs to drain this round."""
        if not self.batch_base:
            return [(None, self.processed_root)]
        roots = list(self.batches.active_roots(self.batch_base))
        if self.args.sealed_only:
            roots = [(i, r) for i, r in roots if self.batches.is_sealed(self.batch_base, i)]
        return roots

    def root_state(self, root: str) -> dict:
        """Per-root claim dir, failure log and discovery mode, created on first use."""
        st = self._states.get(root)
        if st is None:
            args = self.args
            cd = os.path.join(
                root, "_phase2_claims" + ("_sec" if args.secondary else ""))
            try:
                os.makedirs(cd, exist_ok=True)
            except OSError:
                pass
            qm = (args.use_queue and not args.secondary
                  and work_queue.is_enabled(root))
            st = self._states[root] = {
                "claims_dir": cd,
                "failed": os.path.join(root, "_failed_phase2.jsonl"),
                "queue_mode": qm,
                "reconciled": not qm,
                "scan_override": False,
            }
            self.logger.info(f"phase-2 root {root}: "
                             f"discovery={'QUEUE' if qm else 'SCAN'}")
        return st

    def drain_root(self, root: str, st: dict, round_i: int) -> bool:
        """Claim each prefix of ``root`` in turn and transcribe its work."""
        args, logger = self.args, self.logger
        claims_dir = st["claims_dir"]
        st["contested"] = False
        use_queue = st["queue_mode"] and not st["scan_override"]
        want_tracks = [t for t in ("short", "long") if args.tracks in ("both", t)]
        prefixes = (
            work_queue.prefixes(root, want_tracks) if use_queue
            else _prefix_dirs(root)
        )
        random.shuffle(prefixes)
        did_any = False
        for prefix in prefixes:
            pdir = os.path.join(root, prefix)
            pend_short = pend_long = []
            if use_queue:
                if args.tracks in ("both", "short"):
                    pend_short = work_queue.pending(root, prefix, "short")
                if args.tracks in ("both", "long"):
                    pend_long = work_queue.pending(root, prefix, "long")
                if not (pend_short or pend_long):
                    continue
            else:
                # Per-track marker: short and long instances share this claims dir,
                # so a long-drained prefix must not make a short instance skip.
                marker = os.path.join(claims_dir, f"{prefix}.done_{args.tracks}")
                if _prefix_drained(marker, pdir):
                    continue  # peer proved it empty + nothing new landed (1 read+1 stat)
            lock = os.path.join(claims_dir, prefix + ".lock")
            if not _claim_lock(lock, LEASE_S):
                st["contested"] = True
                continue  # another instance owns this prefix
            ack_short: list = []
            ack_long: list = []
            try:
                if use_queue:
                    work, long_work, ack_short, ack_long, stale = _queue_prefix(
                        root, prefix, args.tracks, pend_short, pend_long
                    )
                    if stale:
                        logger.info(
                            f"[round {round_i}] prefix {prefix}: cleared {stale} stale "
                            f"entries (final output exists or source is missing)"
                        )
                else:
                    # Capture BEFORE the scan so phase-1 output landing mid-scan
                    # invalidates the marker instead of being skipped.
                    fp_before = _prefix_fingerprint(pdir)
                    work, long_work = _scan_prefix(
                        root, prefix, args.secondary, args.tracks
                    )
                n_poison = len(work) + len(long_work)
                work = [t for t in work if not _poisoned(t[0])]
                long_work = [t for t in long_work if not _poisoned(t[0])]
                n_poison -= len(work) + len(long_work)
                if not (work or long_work):
                    if not use_queue and not n_poison:
                        _mark_drained(marker, fp_before)
                    continue
                did_any = True
                self.transcribe_prefix(prefix, work, long_work, lock, st["failed"], round_i)
            finally:
                if ack_short:
                    work_queue.ack_finished(root, "short", ack_short)
                if ack_long:
                    work_queue.ack_finished(root, "long", ack_long)
                try:
                    os.remove(lock)  # release (re-claimable for new partials)
                except OSError:
                    pass
        return did_any

    def transcribe_prefix(self, prefix: str, work: list, long_work: list, lock: str,
                          failed_log_path: str, round_i: int) -> None:
        """Run the short track, then each long chunk, for one claimed prefix."""
        args, logger, cfg = self.args, self.logger, self.cfg
        logger.info(
            f"[round {round_i}] prefix {prefix}: {len(work)} short"
            f"{' SECONDARY' if args.secondary else ''}, {len(long_work)} long"
        )
        _run_short_track(
            work, self.asr, batch_size=cfg.runtime.batch_size,
            min_char=cfg.filter.min_char_count,
            ratio_filter=cfg.filter.ratio_filter, filter_cfg=cfg.filter,
            prune_segments=args.prune_segments, secondary=args.secondary,
            logger=logger, failed_log_path=failed_log_path,
            heartbeat=lambda lk=lock: _touch(lk), desc=f"tx {prefix}",
            prefetch=args.prefetch, batch_files=args.batch_files,
        )
        for src, dst in long_work:
            sid_dir = os.path.dirname(src)
            try:
                _transcribe_long_chunks(
                    src_path=src, dst_path=dst, sid_dir=sid_dir,
                    asr=self.asr, logger=logger, ratio_filter=cfg.filter.ratio_filter,
                )
            except Exception as e:  # noqa: BLE001 - keep loop alive
                logger.exception(f"long-track failed on {src}: {e}")
                _record_phase2_fail(failed_log_path, src, e, key="long_chunks")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _touch(lock)

    def finish_batch(self, bi: int, root: str, st: dict, worked: bool) -> bool:
        """Mark a sealed, drained batch ``_PHASE2_DONE``; True if anything changed."""
        batches, logger = self.batches, self.logger
        if worked or not batches.is_sealed(self.batch_base, bi):
            return False

        def blocked() -> bool:
            if st.get("contested"):
                logger.info(
                    f"batch {batches.batch_name(bi)}: a prefix is locked by another "
                    f"worker; its status is unknown, so sealing is deferred"
                )
                return True
            bad = _root_poison(root)
            if bad:
                logger.warning(
                    f"batch {batches.batch_name(bi)}: {len(bad)} files failed "
                    f"{_P2_MAX_FAILS} times; _PHASE2_DONE is withheld pending "
                    f"restart or review. Examples: {bad[:3]}"
                )
                return True
            return False

        if st["scan_override"]:
            if blocked():
                return False
            batches.mark_phase2_done(self.batch_base, bi)
            logger.info(f"batch {batches.batch_name(bi)} complete: _PHASE2_DONE")
            self._states.pop(root, None)
            return True
        if not st["reconciled"]:
            st["scan_override"] = True
            st["reconciled"] = True
            logger.info(f"batch {batches.batch_name(bi)} queue empty; running fallback scan")
            return True
        if blocked():
            return False
        batches.mark_phase2_done(self.batch_base, bi)
        logger.info(f"batch {batches.batch_name(bi)} complete: _PHASE2_DONE")
        self._states.pop(root, None)
        return True

    def run(self) -> None:
        args, logger = self.args, self.logger
        reconciled = True
        scan_override = False
        if not self.batch_base:
            reconciled = self.root_state(self.processed_root)["reconciled"]

        def needs_reconcile() -> bool:
            nonlocal reconciled, scan_override
            if reconciled:
                return False
            logger.info("queue drained; running a fallback scan for missed work")
            reconciled = True
            scan_override = True
            return True

        round_i = 0
        while True:
            round_i += 1
            did_any = False
            try:
                round_roots = self.active_roots()
            except OSError as ex:
                logger.warning(f"cannot read batch metadata; will not exit this round: {ex}")
                if args.loop:
                    time.sleep(LOOP_SLEEP)
                    continue
                raise
            for bi, root in round_roots:
                st = self.root_state(root)
                if not self.batch_base:
                    st["scan_override"] = scan_override
                worked = self.drain_root(root, st, round_i)
                did_any = did_any or worked
                if self.batch_base and self.finish_batch(bi, root, st, worked):
                    did_any = True

            # ---- loop control ----
            if self.batch_base:
                if not args.loop:
                    if did_any:
                        continue
                    break
                phase1_done = self.phase1_done()
                remaining_roots = None
                if phase1_done and not did_any:
                    try:
                        remaining_roots = self.active_roots()
                    except OSError as ex:
                        logger.warning(f"cannot recheck batch metadata; will not exit: {ex}")
                        time.sleep(LOOP_SLEEP)
                        continue
                if phase1_done and not did_any and remaining_roots == []:
                    logger.info("phase1 and all batches complete; exiting --loop")
                    break
                if phase1_done and not did_any and _poisoned_all():
                    logger.warning(
                        "phase1 complete but repeatedly failed files remain; exiting --loop "
                        "with their batches unsealed for retry or review"
                    )
                    break
                if not did_any:
                    time.sleep(LOOP_SLEEP)
                continue

            if scan_override:
                scan_override = False
                if did_any:
                    reconciled = False
                    continue
            if not args.loop:
                if needs_reconcile():
                    continue
                break  # single-shot: one pass over the backlog, then exit
            if self.phase1_done() and not did_any:
                if needs_reconcile():
                    continue
                logger.info("phase1 done + no claimable work → exiting --loop")
                break  # synchronous pipeline: phase1 finished and nothing pending
            if not did_any:
                time.sleep(LOOP_SLEEP)  # idle: wait for phase1 to produce more

        bad = _poisoned_all()
        if bad:
            logger.warning(
                f"this process skipped {len(bad)} files after {_P2_MAX_FAILS} failures "
                f"(not finalized or acknowledged; restart to retry): "
                + ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
            )


def main() -> None:
    args = _build_parser().parse_args()
    _check_args(args)
    cfg = PipelineConfig.load(args.config_path)
    if cfg.stages.asr is None:
        raise SystemExit(
            "config.stages.asr is None — Phase 2 needs an ASR adapter. "
            "Set ``stages.asr`` in the config and re-run."
        )

    logger = Logger.get_logger()
    device = "cuda" if detect_gpu() else "cpu"
    logger.info(f"phase-2 device: {device}")
    if args.num_shards > 1:
        logger.info(
            f"phase-2 shard {args.shard}/{args.num_shards}: the start-up check "
            "covers this shard's dirs only; prefix claims share out the work"
        )
    processed_root = _processed_root(args, cfg)
    if not _prescan_finds_work(args, processed_root, logger):
        return

    asr: ASRModel = build_adapter("asr", cfg.stages.asr, device)
    t0 = time.time()
    asr.warmup()  # vLLM loaded ONCE; the loop below reuses it, never reloads.
    logger.info(f"asr warmup {time.time() - t0:.1f}s")

    dialogue_phase2.set_lid_unsupported(lid_unsupported)
    dialogue_windows.configure(rescue=bool(args.dialogue_rescue),
                               gates=cfg.dialogue_window.gates())
    # An explicitly set environment variable wins over the config.
    long_track.configure(
        max_gap_s=(None if "EMILIA2_DATA_LONG_MAX_GAP_S" in os.environ
                   else cfg.long_chunk.max_gap_s),
        min_dur_s=(None if "EMILIA2_DATA_LONG_MIN_DUR_S" in os.environ
                   else cfg.long_chunk.min_duration_s))
    _Phase2Run(args, cfg, asr, logger, processed_root).run()


if __name__ == "__main__":
    main()
