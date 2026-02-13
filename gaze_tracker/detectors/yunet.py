"""Детектор лица и ориентиров (YuNet, OpenCV Zoo, ONNX).

YuNet выдаёт:
- bbox лица;
- 5 ориентиров: левый/правый глаз, нос, левый/правый угол рта.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

from gaze_tracker.types import BBox
from gaze_tracker.utils.download import download_file


YUNET_MODEL_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
# SHA256 взят из Git‑LFS указателя файла в репозитории OpenCV Zoo.
YUNET_MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


@dataclass(frozen=True)
class FaceDetection:
    """Результат детекции одного лица."""

    bbox: BBox
    score: float
    # Порядок: left_eye, right_eye, nose, mouth_left, mouth_right.
    landmarks: dict[str, tuple[float, float]]


def default_model_dir() -> Path:
    """Папка кэша модели YuNet в кэше пользователя."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "gaze_tracker" / "models"


def ensure_yunet_model(model_dir: Path | None = None) -> Path:
    """Гарантирует наличие модели YuNet на диске (скачает при необходимости)."""
    model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
    path = model_dir / "face_detection_yunet_2023mar.onnx"
    if path.exists():
        return path
    return download_file(YUNET_MODEL_URL, path, expected_sha256=YUNET_MODEL_SHA256)


class YuNetDetector:
    """Обёртка над `cv2.FaceDetectorYN` с выбором «лучшего» лица."""

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        score_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV не установлен. Установи: python -m pip install opencv-python")

        model_path = Path(model_path) if model_path is not None else ensure_yunet_model()
        self.model_path = model_path

        # Размер входа переопределяем на каждом кадре через setInputSize().
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )

    def detect_best(self, bgr: np.ndarray) -> FaceDetection | None:
        """Детектит лица и возвращает самое «уверенное/крупное» (эвристика)."""
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return None

        faces = np.asarray(faces, dtype=np.float32)
        # Формат FaceDetectorYN:
        # [x, y, w, h, l0x, l0y, l1x, l1y, l2x, l2y, l3x, l3y, l4x, l4y, score]
        scores = faces[:, 14]
        areas = faces[:, 2] * faces[:, 3]
        idx = int(np.lexsort((areas, scores))[-1])
        f = faces[idx]

        x, y, fw, fh = [float(v) for v in f[:4]]
        score = float(f[14])
        bbox = BBox(x, y, x + fw, y + fh).clamp(w, h)

        pts = [float(v) for v in f[4:14]]
        lm = {
            "left_eye": (pts[0], pts[1]),
            "right_eye": (pts[2], pts[3]),
            "nose": (pts[4], pts[5]),
            "mouth_left": (pts[6], pts[7]),
            "mouth_right": (pts[8], pts[9]),
        }
        return FaceDetection(bbox=bbox, score=float(score), landmarks=lm)
