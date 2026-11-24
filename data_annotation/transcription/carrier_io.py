"""Read segment audio: decode carriers once, slice windows, load standalone files."""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from pipeline.types import Segment
from utils.logger import Logger


# Process-global, bounded pool for per-segment audio reads. One <sid> can
# carry 50–2000 segment files; reading them serially over NFS (one blocking
# ``sf.read`` each, latency-bound) starves the GPU whenever a big sid is the
# file being loaded — the other prefetch slots sit idle behind that one
# serial loop. We fan a file's reads across this shared pool so a big sid
# issues many NFS RPCs at once, and bound it PROCESS-GLOBALLY so total
# in-flight reads stay near the NFS sweet spot no matter how many files
# prefetch concurrently. Reassembly is by original order, so the concatenated
# buffer + per-segment offsets preserve the serial path's order. Tune via
# ``EMILIA2_DATA_P2_SEG_READERS`` (raise if
# the NFS mount has read headroom; lower if many instances contend).
_SEG_READERS = max(1, int(os.environ.get("EMILIA2_DATA_P2_SEG_READERS", "8")))


_CARRIER_TAIL_SLACK = 1024

_SEG_READ_POOL = ThreadPoolExecutor(
    max_workers=_SEG_READERS, thread_name_prefix="p2segread"
)


def _own_audio(r: dict) -> str | None:
    ap = r.get("audio_path")
    if not ap:
        return None
    if r.get("carrier_path"):
        return None
    return ap


def _resolve_path(path: str, base_dir: str) -> str:
    """Resolve a manifest path (absolute or relative to ``base_dir``)."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _load_segment_audio(seg_meta: dict, base_dir: str) -> tuple[np.ndarray, int]:
    """Read a per-segment WAV referenced by a partial-manifest row."""
    full = _resolve_path(seg_meta["audio_path"], base_dir)
    waveform, sr = sf.read(full, dtype="float32", always_2d=False)
    if waveform.ndim == 2:  # defensive — phase 1 writes mono
        waveform = waveform.mean(axis=1).astype(np.float32)
    return waveform, int(sr)


def _read_carrier_windows(
    rows: list[dict], sid_dir: str, target_sr: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Decode one carrier sequentially and retain only requested windows.

    PCM is streamed from ffmpeg in bounded chunks, so a five-hour recording
    never becomes a multi-GB temporary numpy array. Overlapping windows are
    supported and result order matches ``rows``.
    """
    if not rows:
        return []
    carrier_names = {r.get("carrier_path") for r in rows}
    source_rates = {int(r.get("sample_rate") or 0) for r in rows}
    if None in carrier_names or len(carrier_names) != 1:
        raise RuntimeError(f"mixed/missing carrier_path in one sid: {carrier_names}")
    if 0 in source_rates or len(source_rates) != 1:
        raise RuntimeError(f"mixed/missing carrier sample rates: {source_rates}")
    source_sr = next(iter(source_rates))
    decode_sr = int(target_sr or source_sr)
    if decode_sr <= 0:
        raise ValueError(f"invalid carrier decode rate: {decode_sr}")

    carrier = _resolve_path(next(iter(carrier_names)), sid_dir)
    windows: list[tuple[int, int, int]] = []
    outputs: list[np.ndarray] = []
    for i, row in enumerate(rows):
        try:
            source_a = int(row["carrier_start_samples"])
            source_b = int(row["carrier_end_samples"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid carrier window in row {i}: {row}") from exc
        source_a = max(0, source_a)
        source_b = max(source_a, source_b)
        # Integer floor matches AudioBundle's int(seconds * sample_rate) cut.
        a = source_a * decode_sr // source_sr
        b = source_b * decode_sr // source_sr
        max_window_s = float(os.environ.get("EMILIA2_DATA_MAX_CARRIER_WINDOW_S", "21600"))
        if b - a > int(max_window_s * decode_sr):
            raise RuntimeError(
                f"carrier window {i} is {(b-a)/decode_sr:.1f}s, "
                f"above safety cap {max_window_s:.1f}s"
            )
        windows.append((a, b, i))
        outputs.append(np.empty(max(0, b - a), dtype=np.float32))

    ordered = sorted(windows, key=lambda x: (x[0], x[1], x[2]))
    filled = [0] * len(rows)
    max_requested_end = max(b for _, b, _ in ordered)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-i", carrier,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ar", str(decode_sr), "-ac", "1", "pipe:1",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    active: list[tuple[int, int, int]] = []
    next_window = 0
    frame_pos = 0
    remainder = b""
    completed_early = False
    try:
        while True:
            raw = proc.stdout.read(4 * 262_144)
            if not raw:
                break
            raw = remainder + raw
            usable = len(raw) - (len(raw) % 4)
            remainder = raw[usable:]
            if usable == 0:
                continue
            chunk = np.frombuffer(raw[:usable], dtype=np.float32)
            chunk_start = frame_pos
            chunk_end = frame_pos + int(chunk.shape[0])
            while next_window < len(ordered) and ordered[next_window][0] < chunk_end:
                active.append(ordered[next_window])
                next_window += 1
            still_active = []
            for a, b, idx in active:
                lo = max(a, chunk_start)
                hi = min(b, chunk_end)
                if hi > lo:
                    outputs[idx][lo - a:hi - a] = chunk[lo - chunk_start:hi - chunk_start]
                    filled[idx] += hi - lo
                if b > chunk_end:
                    still_active.append((a, b, idx))
            active = still_active
            frame_pos = chunk_end
            if frame_pos >= max_requested_end and next_window == len(ordered) and not active:
                completed_early = True
                if proc.poll() is None:
                    proc.terminate()  # no need to decode the carrier tail
                # Closing our read end is what actually stops it. SIGTERM alone
                # deadlocks whenever the undecoded tail exceeds the 64 KiB pipe
                # buffer: ffmpeg blocks writing into the pipe we just stopped
                # draining while we block on stderr's EOF, and the GPU idles
                # until someone notices. The close makes that write fail.
                proc.stdout.close()
                break
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        returncode = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if returncode != 0 and not completed_early:
        raise RuntimeError(
            f"ffmpeg carrier decode failed for {carrier}: "
            f"{stderr.decode(errors='replace').strip()[:600]}"
        )
    for i, ((a, b, _), got) in enumerate(zip(windows, filled)):
        if got == b - a:
            continue

        available = max(0, min(b, frame_pos) - a)
        short = (b - a) - got
        if got > 0 and got == available and 0 < short <= _CARRIER_TAIL_SLACK:
            outputs[i][got:] = 0.0
            Logger.get_logger().warning(
                "carrier window %d is short by %d samples (%.2f ms); padded: %s",
                i, short, short / decode_sr * 1000.0, carrier,
            )
            continue
        raise RuntimeError(
            f"carrier window {i} [{a}, {b}) exceeds decoded "
            f"length {frame_pos} for {carrier} (got {got}/{b - a})"
        )
    return [(wav, decode_sr) for wav in outputs]


def _read_segments_parallel(
    rows: list[dict], sid_dir: str, target_sr: int | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Read legacy loose clips, or decode one carrier for all new rows."""
    if rows and any(r.get("carrier_path") for r in rows):
        if not all(r.get("carrier_path") for r in rows):
            raise RuntimeError("mixed loose-audio and carrier rows in one sid")
        return _read_carrier_windows(rows, sid_dir, target_sr=target_sr)
    if len(rows) > 1:
        return list(_SEG_READ_POOL.map(lambda r: _load_segment_audio(r, sid_dir), rows))
    return [_load_segment_audio(rows[0], sid_dir)]


def _segment_from_meta(meta: dict) -> Segment:
    """Rebuild a Segment object from a partial-manifest row.

    ``to_legacy_dict`` flattens speaker/language/quality/extra into the
    JSON; ``from_legacy_dict`` reverses that. Anything Phase 2 wrote on
    its own (``audio_path``, ``sample_rate``) goes into ``extra``.
    """
    return Segment.from_legacy_dict(meta)
