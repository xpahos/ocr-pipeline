"""MD5 hashing of PDF files."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def md5_of_file(path: Path) -> str:
    """Return the lowercase hex MD5 digest of a file, read in chunks."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()
