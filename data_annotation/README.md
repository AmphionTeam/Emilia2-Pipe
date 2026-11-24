# Data annotation

Data annotation runs in two passes, each in its own environment:

| Pass | Entry | Does |
|---|---|---|
| Phase 1 | `main.py` | Standardizes audio, separates vocals, diarizes speakers, detects speech and scores quality. Writes one audio carrier plus short, long and dialogue manifests per recording. |
| Phase 2 (ASR) | `transcribe.py` | Transcribes those manifests. The entry point schedules work; the `transcription/` package holds the short, long and dialogue tracks, carrier audio I/O and work discovery. |

Both passes skip recordings that are already complete. Run them through
`scripts/run_data_annotation.py` from the repository root. For a corpus that
ships as metadata plus tar files, use `run_pipeline_multi.py`, which needs
explicit metadata, output and config arguments.

Synthetic-speech detection is optional. With `stages.deepfake` set to `null`,
the default, `is_fake` and `deepfake_score` stay unset and the fake-related
filters are skipped. An unset value means the segment was not evaluated.
