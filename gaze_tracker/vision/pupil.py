"""Best‑effort поиск центра зрачка (без обучения).

Это не полноценный сегментатор зрачка, а практичная эвристика для веб‑камеры:
мы ищем «самые тёмные» пиксели в центральной части глаза и берём взвешенный центр масс.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


@dataclass(frozen=True)
class PupilResult:
    """Нормированные координаты зрачка внутри crop'а глаза."""

    x_norm: float
    y_norm: float
    confidence: float


def find_pupil(gray_eye: np.ndarray) -> PupilResult | None:
    """Ищет центр зрачка на сером crop'е глаза.

    Алгоритм (эвристика):
    - размытие (Gaussian blur) для подавления шума;
    - берём тёмные пиксели по процентилю внутри центральной маски;
    - считаем взвешенный центр (чем темнее — тем больше вес).

    Возвращает нормированные `(x, y)` в `[0..1]` внутри crop'а.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV не установлен. Установи: python -m pip install opencv-python")

    if gray_eye.ndim != 2:
        raise ValueError("gray_eye должен быть 2D (grayscale)")

    h, w = gray_eye.shape[:2]
    if h < 10 or w < 10:
        return None

    img = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Исключаем края: веки/ресницы/артефакты crop'а.
    # Асимметрия по Y помогает при взгляде «вниз» (сохраняем больше нижней части).
    base = int(min(h, w))
    m_x = max(3, int(round(base * 0.12)))
    m_top = max(3, int(round(base * 0.18)))
    m_bottom = max(2, int(round(base * 0.06)))

    cap = int(base // 3)
    m_x = min(m_x, cap, w - 2)
    m_top = min(m_top, cap, h - 2)
    m_bottom = min(m_bottom, cap, h - 2)

    if (h - m_top - m_bottom) < 6 or (w - 2 * m_x) < 6:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[m_top : h - m_bottom, m_x : w - m_x] = True

    vals = img[mask].astype(np.float32)
    if vals.size < 50:
        return None

    img_f = img.astype(np.float32)
    mask_count = float(np.count_nonzero(mask))
    if mask_count <= 0:
        return None

    best: tuple[float, float, float] | None = None
    for pct in (10.0, 15.0, 20.0):
        thr = float(np.percentile(vals, pct))
        sel = (img_f <= thr) & mask
        ys, xs = np.where(sel)
        if xs.size < 20:
            continue

        weights = (thr - img_f[ys, xs] + 1.0)
        denom = float(np.sum(weights))
        if not np.isfinite(denom) or denom <= 1e-6:
            continue

        cx = float(np.sum(xs.astype(np.float32) * weights) / denom)
        cy = float(np.sum(ys.astype(np.float32) * weights) / denom)
        nx = cx / float(w)
        ny = cy / float(h)

        if nx < 0.02 or nx > 0.98 or ny < 0.02 or ny > 0.98:
            continue

        frac = float(xs.size) / mask_count
        # Предпочитаем не слишком большие области (случаи «всё тёмное» обычно плохие).
        if best is None or frac < best[2]:
            best = (nx, ny, frac)

    if best is None:
        return None

    nx, ny, frac = best
    # Грубая оценка уверенности: «насколько маленькое и аккуратное тёмное пятно мы нашли».
    conf = float(max(0.0, min(1.0, frac / 0.20)))
    return PupilResult(x_norm=float(nx), y_norm=float(ny), confidence=conf)
