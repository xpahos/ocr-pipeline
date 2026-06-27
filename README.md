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
3. **Pipeline** (LangGraph) — `validate → split → recognize → merge → write`. Oversized
   PDFs are split to fit OpenAI's 100-page / 32 MB per-request limits, then merged. The
   transcription streams token-by-token from the Responses API and is written atomically.

Processing is **serial**: at most one OpenAI job runs at a time.

## Correcting a transcription

When a transcription has mistakes, you don't have to re-do anything by hand. Add an
instructions section at the **end** of the `.md` file:

```markdown
## OCR Instructions

- The diagram on page 2 is a UML sequence diagram, not a flowchart.
- "Пётр" is a proper name — keep the capitalization.
```

On save, the service notices the change (the `instr:` hash in the first line no longer
matches), re-runs recognition for that PDF with your notes appended to the system prompt,
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
