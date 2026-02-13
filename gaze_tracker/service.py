"""Сервис: YuNet → ориентиры лица → области глаз → зрачок → признаки → калибровка.

Здесь находится основная «логика продукта»:
- детект лица и 5 ориентиров (предобученная модель YuNet);
- эвристический поиск центра зрачка на кадре веб‑камеры;
- персональная калибровка (линейная ridge‑регрессия) в координаты «экрана».
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

from gaze_tracker.calibration import LinearCalibrator
from gaze_tracker.detectors.yunet import (
    FaceDetection,
    YuNetDetector,
    default_model_dir,
    ensure_yunet_model,
)
from gaze_tracker.types import BBox
from gaze_tracker.vision.pupil import PupilResult, find_pupil


def _bbox_json(bb: BBox) -> dict[str, float]:
    """Преобразует `BBox` в JSON‑словарь."""
    return {
        "x0": float(bb.x0),
        "y0": float(bb.y0),
        "x1": float(bb.x1),
        "y1": float(bb.y1),
    }


def _pupil_json(p: PupilResult | None) -> dict[str, float] | None:
    """Преобразует `PupilResult` в JSON‑словарь (или `None`)."""
    if p is None:
        return None
    return {"x": float(p.x_norm), "y": float(p.y_norm), "conf": float(p.confidence)}


def _avg_pupil(left: PupilResult | None, right: PupilResult | None) -> tuple[float, float, float] | None:
    """Усредняет координаты зрачка (если есть левый/правый глаз).

    Возвращает `(x_norm, y_norm, confidence)` в нормированных координатах области глаза `[0..1]`.
    Если нет ни одного зрачка — возвращает `None`.
    """
    xs: list[float] = []
    ys: list[float] = []
    cs: list[float] = []
    if left is not None:
        xs.append(float(left.x_norm))
        ys.append(float(left.y_norm))
        cs.append(float(left.confidence))
    if right is not None:
        xs.append(float(right.x_norm))
        ys.append(float(right.y_norm))
        cs.append(float(right.confidence))
    if not xs:
        return None
    return (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(cs)))


def default_state_dir() -> Path:
    """Папка состояния (калибровки) в кэше пользователя."""
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    return base / "gaze_tracker" / "state"


def get_screen_size() -> tuple[int, int]:
    """Пытается получить размер экрана (в пикселях).

    Используется только для удобной визуализации `x_px/y_px`. Если получить размер нельзя,
    возвращает безопасный дефолт `1920x1080`.
    """
    try:
        import tkinter as tk  # стандартная библиотека

        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return 1920, 1080


@dataclass(frozen=True)
class FeatureDebug:
    """Отладочные данные одного кадра: что нашли и какие признаки собрали."""

    face: FaceDetection
    eye_left_bb: BBox
    eye_right_bb: BBox
    pupil_left: PupilResult | None
    pupil_right: PupilResult | None
    features: np.ndarray | None


class GazeService:
    """Сервис трекинга: детект лица/ориентиров, зрачок и персональная калибровка."""

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        calib_path: Path | None = None,
        load_calibration: bool = False,
        feature_dim: int = 14,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV не установлен. Установи: python -m pip install opencv-python")

        self.model_dir = Path(model_dir) if model_dir is not None else default_model_dir()
        self.detector = YuNetDetector(model_path=ensure_yunet_model(self.model_dir))

        self.feature_dim = int(feature_dim)
        self.calibrator = LinearCalibrator(feature_dim=self.feature_dim, ridge_lambda=1e-4)

        self.state_dir = default_state_dir()
        self.calib_path = Path(calib_path) if calib_path is not None else (self.state_dir / "calibration_v2.npz")
        self.saved_calib_exists = bool(self.calib_path.exists())
        self.calibration_loaded = False

        if load_calibration and self.calib_path.exists():
            try:
                self.calibrator = LinearCalibrator.load(self.calib_path)
                self.calibration_loaded = True
            except Exception:
                # Если калибровка битая — просто стартуем без неё.
                pass

        self.screen_w, self.screen_h = get_screen_size()

    def reset_calibration(self) -> None:
        """Сбрасывает калибровку и удаляет сохранённый файл (если был)."""
        self.calibrator.W = None
        self.calibrator.reset_samples()
        try:
            self.calib_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _eye_boxes_from_landmarks(self, face: FaceDetection, img_w: int, img_h: int) -> tuple[BBox, BBox]:
        """Строит bounding‑box'ы глаз по ориентирам YuNet."""
        le = face.landmarks["left_eye"]
        re = face.landmarks["right_eye"]
        dx = float(re[0] - le[0])
        dy = float(re[1] - le[1])
        d = float((dx * dx + dy * dy) ** 0.5)
        d = max(20.0, min(d, float(max(img_w, img_h))))

        # Эвристика: размер бокса глаза зависит от расстояния между глазами.
        ew = 0.72 * d
        eh = 0.65 * d
        # Добавляем чуть больше нижнего века: помогает при взгляде «вниз».
        y_off = 0.02 * d

        le_bb = BBox(
            le[0] - ew / 2.0,
            le[1] - eh / 2.0 + y_off,
            le[0] + ew / 2.0,
            le[1] + eh / 2.0 + y_off,
        ).clamp(img_w, img_h)
        re_bb = BBox(
            re[0] - ew / 2.0,
            re[1] - eh / 2.0 + y_off,
            re[0] + ew / 2.0,
            re[1] + eh / 2.0 + y_off,
        ).clamp(img_w, img_h)
        return le_bb, re_bb

    def extract_features(self, bgr: np.ndarray) -> FeatureDebug | None:
        """Извлекает признаки из кадра.

        Возвращает `FeatureDebug` (включая отладочную информацию) или `None`, если лицо не найдено.
        """
        h, w = bgr.shape[:2]
        face = self.detector.detect_best(bgr)
        if face is None:
            return None

        le_bb, re_bb = self._eye_boxes_from_landmarks(face, w, h)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        def crop_gray(bb: BBox) -> np.ndarray:
            """Вырезает серый кадр по bbox (с округлением координат)."""
            x0, y0, x1, y1 = bb.as_int_tuple()
            return gray[y0:y1, x0:x1]

        le_crop = crop_gray(le_bb)
        re_crop = crop_gray(re_bb)

        pupil_l = find_pupil(le_crop) if le_crop.size else None
        pupil_r = find_pupil(re_crop) if re_crop.size else None

        has_pupil = not (pupil_l is None and pupil_r is None)

        # Нормализованные параметры лица (помогают компенсировать движение головы).
        fb = face.bbox
        fc_x, fc_y = fb.center()
        face_cx = float(fc_x) / float(w)
        face_cy = float(fc_y) / float(h)
        face_w = float(fb.width()) / float(w)
        face_h = float(fb.height()) / float(h)

        # Зрачок в нормированных координатах внутри каждого crop'а.
        # Если зрачок не найден — используем 0.5/0.5 как нейтральный центр.
        le_x = float(pupil_l.x_norm) if pupil_l is not None else 0.5
        le_y = float(pupil_l.y_norm) if pupil_l is not None else 0.5
        re_x = float(pupil_r.x_norm) if pupil_r is not None else 0.5
        re_y = float(pupil_r.y_norm) if pupil_r is not None else 0.5

        # Межзрачковое расстояние (нормированное).
        le_pt = face.landmarks["left_eye"]
        re_pt = face.landmarks["right_eye"]
        dx_eye = float(re_pt[0] - le_pt[0])
        dy_eye = float(re_pt[1] - le_pt[1])
        d_eye_px = float((dx_eye * dx_eye + dy_eye * dy_eye) ** 0.5)
        inter = float(d_eye_px) / float(max(w, h))

        # Прокси‑признаки позы/поворота головы по ориентирам (нормируем на d_eye_px).
        d = max(1.0, d_eye_px)
        mid_eye_x = (float(le_pt[0]) + float(re_pt[0])) * 0.5
        mid_eye_y = (float(le_pt[1]) + float(re_pt[1])) * 0.5

        nose = face.landmarks["nose"]
        ml = face.landmarks["mouth_left"]
        mr = face.landmarks["mouth_right"]
        mouth_cx = (float(ml[0]) + float(mr[0])) * 0.5
        mouth_cy = (float(ml[1]) + float(mr[1])) * 0.5

        # Поджимаем координаты: YuNet иногда возвращает точки чуть за пределами кадра.
        nx = float(np.clip(float(nose[0]), 0.0, float(w)))
        ny = float(np.clip(float(nose[1]), 0.0, float(h)))
        mx = float(np.clip(float(mouth_cx), 0.0, float(w)))
        my = float(np.clip(float(mouth_cy), 0.0, float(h)))

        nose_dx = (nx - mid_eye_x) / d
        nose_dy = (ny - mid_eye_y) / d
        mouth_dx = (mx - mid_eye_x) / d
        mouth_dy = (my - mid_eye_y) / d

        roll = float(np.arctan2(dy_eye, dx_eye) / np.pi)  # примерно [-1..1]

        feat = None
        if has_pupil:
            feat = np.asarray(
                [
                    le_x,
                    le_y,
                    re_x,
                    re_y,
                    face_cx,
                    face_cy,
                    face_w,
                    face_h,
                    inter,
                    float(nose_dx),
                    float(nose_dy),
                    float(mouth_dx),
                    float(mouth_dy),
                    float(roll),
                ],
                dtype=np.float32,
            )
        return FeatureDebug(
            face=face,
            eye_left_bb=le_bb,
            eye_right_bb=re_bb,
            pupil_left=pupil_l,
            pupil_right=pupil_r,
            features=feat,
        )

    def predict(self, bgr: np.ndarray) -> dict:
        """Обрабатывает кадр и возвращает JSON‑совместимый результат."""
        dbg = self.extract_features(bgr)
        if dbg is None:
            return {"ok": False, "error": "no_face"}

        warnings: list[str] = []
        if dbg.features is None:
            warnings.append("no_pupil")

        out: dict[str, object] = {
            "ok": True,
            "calibrated": self.calibrator.W is not None,
            "warnings": warnings,
            "face": {
                "bbox": _bbox_json(dbg.face.bbox),
                "score": float(dbg.face.score),
                "landmarks": dbg.face.landmarks,
            },
            "eyes": {
                "left_bbox": _bbox_json(dbg.eye_left_bb),
                "right_bbox": _bbox_json(dbg.eye_right_bb),
            },
            "pupil": {
                "left": _pupil_json(dbg.pupil_left),
                "right": _pupil_json(dbg.pupil_right),
            },
        }

        avg = _avg_pupil(dbg.pupil_left, dbg.pupil_right)
        if avg is None:
            out["gaze_dir"] = None
        else:
            px, py, pconf = avg
            # Вектор направления в [-1, 1], где +x = вправо, +y = вверх.
            dx = float(np.clip((px - 0.5) * 2.0, -1.0, 1.0))
            dy = float(np.clip((0.5 - py) * 2.0, -1.0, 1.0))
            out["gaze_dir"] = {"x": dx, "y": dy, "conf": float(pconf)}

        if self.calibrator.W is not None and dbg.features is not None:
            xy = self.calibrator.predict(dbg.features)
            x = float(np.clip(xy[0], 0.0, 1.0))
            y = float(np.clip(xy[1], 0.0, 1.0))
            out["screen"] = {
                "x_norm": x,
                "y_norm": y,
                "x_px": x * float(self.screen_w),
                "y_px": y * float(self.screen_h),
                "screen_w_px": int(self.screen_w),
                "screen_h_px": int(self.screen_h),
            }
        else:
            out["screen"] = None

        return out

    def add_calibration_sample(self, bgr: np.ndarray, target_x: float, target_y: float) -> dict:
        """Добавляет один сэмпл калибровки (кадр + целевая точка в норм. координатах)."""
        dbg = self.extract_features(bgr)
        if dbg is None:
            return {"ok": False, "error": "no_face"}
        if dbg.features is None:
            return {"ok": False, "error": "no_pupil"}

        tx = float(np.clip(target_x, 0.0, 1.0))
        ty = float(np.clip(target_y, 0.0, 1.0))
        self.calibrator.add_sample(dbg.features, (tx, ty))
        return {"ok": True, "num_samples": self.calibrator.num_samples}

    def fit_calibration(self) -> dict:
        """Обучает калибратор и сохраняет его на диск."""
        W = self.calibrator.fit()
        try:
            self.calibrator.save(self.calib_path)
        except Exception:
            pass
        return {
            "ok": True,
            "num_samples": self.calibrator.num_samples,
            "W_shape": list(W.shape),
            "calib_path": str(self.calib_path),
        }
