"""Утилиты скачивания файлов (модель YuNet) с проверкой SHA256."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path


def _sha256(path: Path) -> str:
    """Считает SHA256 файла на диске."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dst: Path,
    *,
    expected_sha256: str | None = None,
    timeout_s: float = 60.0,
) -> Path:
    """Скачивает файл по URL в `dst` (атомарно) и опционально проверяет SHA256."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    req = urllib.request.Request(url, headers={"User-Agent": "gaze-tracker/1.0"})
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)

    if expected_sha256 is not None:
        got = _sha256(tmp)
        if got.lower() != expected_sha256.lower():
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise RuntimeError(f"SHA256 не совпал для {dst.name}: ожидали {expected_sha256}, получили {got}")

    os.replace(tmp, dst)
    return dst
