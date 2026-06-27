from __future__ import annotations

import hashlib

from ocr_pipeline.hashing import md5_of_file


def test_md5_matches_hashlib(tmp_path):
    data = b"hello handwritten world" * 1000
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    assert md5_of_file(p) == hashlib.md5(data).hexdigest()


def test_md5_empty_file(tmp_path):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    assert md5_of_file(p) == hashlib.md5(b"").hexdigest()
