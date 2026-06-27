from __future__ import annotations

from ocr_pipeline.mdfile import md_path_for, write_md
from ocr_pipeline.reconcile import is_ignored, iter_pdfs, reconcile
from pathlib import Path


def test_is_ignored_globs():
    assert is_ignored(Path(".obsidian/app.json"), [".obsidian/**"]) is True
    assert is_ignored(Path(".obsidian"), [".obsidian/**"]) is True
    assert is_ignored(Path("notes/a.pdf"), [".obsidian/**"]) is False


def test_iter_pdfs_skips_ignored(tmp_path, pdf_factory):
    (tmp_path / ".obsidian").mkdir()
    pdf_factory("keep.pdf")
    hidden = tmp_path / ".obsidian" / "skip.pdf"
    pdf_factory("skip.pdf")  # create then move into ignored dir
    (tmp_path / "skip.pdf").rename(hidden)

    found = {p.name for p in iter_pdfs(tmp_path, [".obsidian/**"])}
    assert found == {"keep.pdf"}


def test_reconcile_enqueues_only_stale(tmp_path, pdf_factory):
    fresh = pdf_factory("fresh.pdf")
    write_md(fresh, "already done")  # up to date

    stale_missing = pdf_factory("missing.pdf")  # no md at all

    stale_wrong = pdf_factory("wrong.pdf")
    md_path_for(stale_wrong).write_text(
        "<!-- ocr-md5: %s -->\n\nold" % ("a" * 32), encoding="utf-8"
    )

    submitted: list[Path] = []
    summary = reconcile(tmp_path, [], submitted.append)

    names = {p.name for p in submitted}
    assert names == {"missing.pdf", "wrong.pdf"}
    assert summary.total == 3
    assert summary.queued == 2
    assert summary.up_to_date == 1


def test_reconcile_warns_orphan_md(tmp_path):
    # A transcription-looking md with no sibling pdf.
    orphan = tmp_path / "ghost.md"
    orphan.write_text("<!-- ocr-md5: %s -->\n\nbody" % ("b" * 32), encoding="utf-8")
    # A plain note (no hash header) must NOT count as an orphan.
    (tmp_path / "note.md").write_text("just notes", encoding="utf-8")

    summary = reconcile(tmp_path, [], lambda p: None)
    assert summary.orphans == 1
