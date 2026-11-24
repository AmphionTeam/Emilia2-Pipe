# Emilia2-Pipe

Emilia2-Pipe turns raw speech recordings into annotated training data. It has
two stages, and you can run either one alone or both in sequence.

| Stage | Input | What it adds |
|---|---|---|
| **Data annotation** | A folder of recordings | Vocal separation, diarization, VAD, quality scores and ASR. Each recording is split into short utterances, long speech and dialogue. |
| **Expressive annotation** | A JSONL manifest of clips | Voice tags (age, sex, dialect, pitch, speed and so on), instruction captions and nonverbal vocalization (NV) labels |

```text
recordings ──▶ data annotation ──▶ short / long / dialogue JSON + audio carrier
                                           │
                                           ▼  (automatic bridge)
                JSONL manifest ──▶ expressive annotation ──▶ final.jsonl
```

| Entry point | Runs |
|---|---|
| `scripts/run_data_annotation.py` | Data annotation |
| `scripts/run_expressive_annotation.py` | Expressive annotation |
| `scripts/run_all.py` | Both stages, converting the data output into an expressive manifest |

All three scripts accept `--dry-run`, which prints each command without
running it.

## Requirements

- Python 3.10 or 3.11
- A CUDA GPU and a matching driver
- `ffmpeg` and `ffprobe` on `PATH`

Model weights and third-party source trees are not part of this repository.

## Installation

### 1. Create the environments

The stages have conflicting dependencies, so each runs in its own
environment:

| Environment | Used for |
|---|---|
| `data_annotation/.venv-phase1` | Separation, diarization, VAD, quality, carrier export |
| `.venv-data-asr` | Qwen3-ASR transcription (vLLM) |
| `.venv-expressive-tags` | Voice tags |
| `.venv-expressive-caption` | Instruction captions (vLLM) |
| `.venv-expressive-nv` | NV detection and verification |

Phase 1 pins PyTorch 2.5.1. Check that it suits your CUDA driver before
running its installer. In the other GPU environments, install a PyTorch build
that matches your driver first.

```bash
# Data annotation
bash data_annotation/setup_phase1.sh
python3.10 -m venv .venv-data-asr
.venv-data-asr/bin/pip install -r requirements/data-asr.txt

# Expressive annotation: one environment per pass
for pass in tags caption nv; do
  python3.10 -m venv .venv-expressive-$pass
  .venv-expressive-$pass/bin/pip install -r requirements/expressive-$pass.txt
  .venv-expressive-$pass/bin/pip install -e ./expressive_annotation --no-deps
done
bash expressive_annotation/scripts/setup_models.sh
```

`setup_phase1.sh` fetches the DiariZen and FireRedASR2S sources, and
`setup_models.sh` fetches Vox-Profile and Voxlect. Both put them in an ignored
`third_party/` directory. [requirements/README.md](requirements/README.md)
explains why the environments must stay separate.

### 2. Configure the models

**Data annotation.** Copy the template and fill in the VAD, LID, separator and
quality model locations:

```bash
cp data_annotation/examples/qwen3_asr.json data_annotation/config.json
```

Git ignores `config.json`. For FireRedVAD, set `model_dir` to the `VAD`
subdirectory of the public checkpoint, which holds both `cmvn.ark` and
`model.pth.tar`.

**Voice tags.** Download the WavLM age/sex checkpoint with its
`microsoft/wavlm-large` backbone, the Voxlect dialect checkpoint for your
language, and PENN's `fcnf0++`. An offline run needs all of them cached.

**NV.** Set these environment variables to the external sources and checkpoints:

| Variable | Needed for |
|---|---|
| `EMILIA2_EXPRESSIVE_NVBENCH_REPO` | All NV runs (NV-Bench source) |
| `EMILIA2_EXPRESSIVE_NVASR_DIR` | All NV runs (NVASR checkpoint) |
| `AUDIOLDM_ROOT`, `CKPT_ROOT` | `--nv-mode verified` only |
| `EMILIA2_EXPRESSIVE_PANNS_LABELS` | `--nv-mode verified` only (AudioSet label CSV) |

`--nv-mode direct` runs NVASR alone. `--nv-mode verified`, the default, adds
a sung-audio filter and PANNs-based verification.

## Usage

### Data annotation

```bash
python scripts/run_data_annotation.py \
  --input audio --config data_annotation/config.json
```

The output goes to `audio_processed/`, next to the input folder. Each
recording gets short, long and dialogue JSON files and one shared audio
carrier. Phase 1 and ASR run in separate interpreters. Use
`--stages phase1` or `--stages asr` to run only one of them. Both resume from
completed recordings.

### Expressive annotation

```bash
python scripts/run_expressive_annotation.py \
  --manifest examples/manifest.example.jsonl \
  --out-dir runs/expressive --nv-mode direct
```

The manifest has one JSON object per clip. A row points either to its own
audio file or to a sample range in a shared carrier:

```json
{"id":"clip_1","wav_path":"audio/clip_1.wav","txt":"Hello","lang":"en"}
{"id":"clip_2","wav_path":"audio/carrier.m4a","carrier_start_samples":44100,"carrier_end_samples":88200,"sample_rate":44100,"txt":"Hi","lang":"en"}
```

Relative paths resolve against `--audio-root` if you pass it. Otherwise they
resolve against the manifest's directory.

- **Components.** All three run by default. Use `--components` to pick a
  subset, for example `--components nv`. Captions are built from the tags, so
  a caption-only run needs an existing `annotations.jsonl`.
- **Resume.** A rerun skips clips whose IDs are already done. Changing the
  manifest for an existing output directory is an error.
- **Chinese captions.** The `zh` caption mode uses English instructions that
  ask the model to answer in Simplified Chinese.

### Both stages

```bash
python scripts/run_all.py \
  --input audio --data-config data_annotation/config.json \
  --out-dir runs/complete --nv-mode direct
```

By default each finished short utterance becomes one expressive clip. To also
annotate long-speech members and dialogue turns, pass
`--tracks short,long,dialogue`. Clip IDs record the source track and
recording. Carrier audio is decoded once per recording and sliced at its
original sample rate.

- `--expressive-manifest FILE` annotates a different corpus instead of the
  data output. The data stage still runs.
- `--data-stages none --processed-root DIR` skips data inference and reuses
  an existing output tree.

## Outputs

| File | Written by | Contents |
|---|---|---|
| `<input>_processed/…/*.json`, `*.long.json`, `*.dialogue.json` | Data annotation | Segments with transcripts and quality metadata |
| `<out-dir>/data_manifest.jsonl` | `run_all.py` | The data output converted to an expressive manifest |
| `<out-dir>/annotations.jsonl` | Tags, then captions | Voice tags and instruction captions per clip |
| `<out-dir>/nv_direct.jsonl` or `nv_verified/final.jsonl` | NV | NV labels |
| `<out-dir>/final.jsonl` | Merge | Input fields joined with every stage's results by ID |

The merge fails if a clip is missing or duplicated, so no clip is dropped
silently.

## Tests

```bash
pip install pytest numpy soundfile 'pydantic>=2,<3'
python scripts/check_release.py        # no machine paths, secrets or binaries
python -m pytest -q tests              # workflow tests
cd expressive_annotation && python -m pytest -q tests
```

The tests need no GPU or model weights.

## Repository layout

```text
scripts/                 entry points and the data → expressive bridge
data_annotation/         Phase 1 (main.py) and ASR (transcribe.py + transcription/)
expressive_annotation/   the emilia2_expressive package, tools and tests
requirements/            one requirements file per environment
examples/                a minimal expressive manifest
tests/                   workflow tests
```

## License

Apache-2.0. See [LICENSE](LICENSE).
