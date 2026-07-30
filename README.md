# ocr-pipeline

A long-running service that watches an Obsidian vault for PDFs containing **handwritten**
Russian/English text and produces a sibling Markdown transcription for each one using the
OpenAI API.

## Contract

For every `foo.pdf` there is exactly one `foo.md`. The **first line** of `foo.md` records
the MD5 of the PDF it was generated from:

```
<!-- ocr-md5: 9f86d081884c7d659a2feaa0c55ad015 -->
```

(An HTML comment, so it does not render in Obsidian.) A PDF is reprocessed whenever its
`.md` is missing, the first line is unparseable, or the recorded hash no longer matches the
PDF's current bytes.

## How it works

1. **Startup reconciliation** — recursively scans the vault and enqueues every PDF whose
   `.md` is missing or stale.
2. **Live monitoring** — a cross-platform `watchdog` observer reacts to PDF created /
   modified / moved events (debounced to absorb sync bursts).
3. **Pipeline** (LangGraph) — `validate → split → OCR → Markdown → verify → merge → write`.
   Each stage has one job:
   - OCR produces a literal transcription with page and uncertainty markers.
   - Markdown formatting transforms only that transcription into Obsidian Markdown.
   - Verification compares the candidate Markdown with the original pages and corrects
     source-supported discrepancies.

   PDFs are split into quality-oriented chunks (10 pages by default), each stage streams
   through the Responses API, and the final transcription is written atomically.

Processing is **serial**: at most one OpenAI job runs at a time.

## PDF versus rendered PNG

The default `OCR_INPUT_MODE=pdf` sends each PDF chunk with `detail=high`. This retains any
embedded text layer and is the best starting point.

Set `OCR_INPUT_MODE=png` to render pages on the fly with PyMuPDF. The temporary PNGs are
created at `OCR_IMAGE_DPI` (default 300), uploaded as ordered `input_image` parts with
`detail=original`, and deleted after each OCR or verification request. They are never
written into the Obsidian vault.

PNG is not assumed to be universally better: compare both modes on representative pages.
It is most likely to help pure scans with small handwriting; PDF mode can win when the PDF
contains a useful text layer.

The OCR and verification passes use `OCR_MODEL` by default; formatting can use a cheaper
text model:

```dotenv
OCR_MODEL=gpt-5.6-terra
OCR_FORMAT_MODEL=gpt-5.6-luna
OCR_VERIFY_MODEL=gpt-5.6-terra
OCR_REASONING_EFFORT=low
```

`OCR_VERIFY_MODE=uncertain` verifies only chunks containing `[?]` or `[illegible]`, plus
documents with human `## OCR Instructions`. Use `always` to verify every chunk or `off` to
disable visual verification. The verifier returns only changed numbered lines through a
strict JSON schema; the service applies those replacements deterministically instead of
paying for another complete Markdown output.

`OCR_FORMATTING_ENABLED=false` disables the formatting pass.

## Correcting a transcription

When a transcription has mistakes, you don't have to re-do anything by hand. Add an
instructions section at the **end** of the `.md` file:

```markdown
## OCR Instructions

- The diagram on page 2 is a UML sequence diagram, not a flowchart.
- "Пётр" is a proper name — keep the capitalization.
```

On save, the service notices the change (the `instr:` hash in the first line no longer
matches), re-runs the pipeline for that PDF with your notes supplied to every stage,
and rewrites the `.md` — **keeping your instructions section intact** so it keeps applying
on future passes. Editing the `.md` alone is enough to trigger a rerun; the PDF doesn't
need to change. Writing the file back does not cause a loop, because once written the
recorded hashes match the file again.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # then edit OPENAI_API_KEY and OCR_VAULT_ROOT
ocr-pipeline
```

Run the tests with `pytest`.

Prometheus metrics are exposed on a separate monitoring port (default `50000`,
override with `--monitoring-port`): `http://127.0.0.1:50000/metrics`.

## Running under systemd

A unit file is provided in [`deploy/ocr-pipeline.service`](deploy/ocr-pipeline.service).
It assumes the project is installed in `/opt/ocr-pipeline` with its venv and `.env`
alongside — adjust the paths, `User=`, and `ReadWritePaths=` (must match
`OCR_VAULT_ROOT`) for your machine, then:

```bash
sudo cp deploy/ocr-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-pipeline
journalctl -u ocr-pipeline -f   # follow logs
```

Environment configuration comes from the same `.env` file: the unit passes it via
`EnvironmentFile=`, and pydantic-settings would also pick it up from
`WorkingDirectory` on its own, so both stay in sync automatically.
