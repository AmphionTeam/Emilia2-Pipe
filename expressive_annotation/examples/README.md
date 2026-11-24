# Example manifest

`manifest.example.jsonl` shows two kinds of row:

| Rows | Fields | Effect |
|---|---|---|
| `clip_0001`, `clip_0002` | `id`, `wav_path`, `txt`, `lang` | Speaking rate comes from `txt`, so Whisper is never loaded |
| `clip_0003` | `id`, `wav_path` | Whisper transcribes the clip and detects its language |

If even one row lacks `txt`, Whisper is loaded for the whole run. Fill in
`txt` on every row or on none.

The caption language is chosen independently of the audio's language, so an
English clip can still receive a Chinese instruction.

To try it on your own audio with the tags pass alone:

```bash
emilia2-annotate --manifest examples/manifest.example.jsonl --audio-root corpus --limit 5
```

When you call `emilia2-annotate` directly, relative paths resolve against
`--audio-root`, and paths outside it are rejected. Without `--audio-root`,
they resolve against the working directory. `emilia2-annotate` can fill in a
missing `id` from the file name. `scripts/run_expressive_annotation.py` needs
every row to have an `id`, and resolves relative paths against the
manifest's directory.
