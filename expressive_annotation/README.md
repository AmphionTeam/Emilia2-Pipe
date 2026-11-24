# Expressive annotation

This package, `emilia2_expressive`, adds three kinds of label to speech clips:

| Pass | Command | Adds |
|---|---|---|
| Tags | `emilia2-annotate` | Voice properties such as age, sex, dialect, pitch and speed |
| Captions | `emilia2-caption` | Instruction captions built from the tags |
| NV | `emilia2-nv` | Nonverbal vocalizations (NV) |

It reads its own JSONL manifest and does not need the data annotation stage.
Run it through `scripts/run_expressive_annotation.py` from the repository
root.

## Manifest rows

Each row needs a unique `id` and an audio source, given in one of these ways:

- `wav_path`, a standalone audio file
- `source_tar` plus `source_member`, a member of a tar file
- `wav_path` plus `carrier_start_samples`, `carrier_end_samples` and
  `sample_rate`, a range of a shared carrier

See [examples/](examples/) for a sample manifest.

## Optional inputs

- **Emotion.** `--emotion-jsonl FILE --emotion-tau T` merges external
  per-clip emotion predictions, keeping those whose confidence passes `T`
  (between 0 and 1). Without them, captions carry no emotion label.
- **Verified NV.** The default `--nv-mode verified` needs the NV-Bench and
  PANNs sources, their weights, and an AudioSet label CSV.
  `--nv-mode direct` skips the verification.
