"""Простая персональная калибровка: линейная ridge‑регрессия признаков в координаты экрана.

Это «классический» подход для прототипа: после того как пользователь посмотрел на 9 точек,
мы аппроксимируем зависимость `features -> (x_norm, y_norm)` аффинным преобразованием.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LinearCalibrator:
    """Линейный калибратор (ridge) для преобразования признаков в (x, y)."""

    feature_dim: int
    ridge_lambda: float = 1e-4
    W: np.ndarray | None = None  # (feature_dim+1, 2), включая смещение (bias)

    def __post_init__(self) -> None:
        """Инициализирует буфер сэмплов."""
        self._X: list[np.ndarray] = []
        self._Y: list[np.ndarray] = []

    def reset_samples(self) -> None:
        """Очищает накопленные сэмплы (без изменения матрицы `W`)."""
        self._X = []
        self._Y = []

    @property
    def num_samples(self) -> int:
        """Количество накопленных сэмплов калибровки."""
        return len(self._X)

    def add_sample(self, features: np.ndarray, target_xy: tuple[float, float]) -> None:
        """Добавляет сэмпл: `features` и целевую точку `target_xy` в `[0..1]`."""
        f = np.asarray(features, dtype=np.float32).reshape(-1)
        if f.shape[0] != self.feature_dim:
            raise ValueError(f"Ожидали feature_dim={self.feature_dim}, получили {f.shape[0]}")
        self._X.append(f)
        self._Y.append(np.asarray([float(target_xy[0]), float(target_xy[1])], dtype=np.float32))

    def fit(self) -> np.ndarray:
        """Обучает `W` по накопленным данным и возвращает матрицу весов."""
        if len(self._X) < 3:
            raise RuntimeError("Нужно минимум 3 сэмпла, чтобы обучить аффинное преобразование")

        X = np.stack(self._X, axis=0).astype(np.float32)
        Y = np.stack(self._Y, axis=0).astype(np.float32)

        # Добавляем признак смещения (единицы).
        ones = np.ones((X.shape[0], 1), dtype=np.float32)
        Xa = np.concatenate([X, ones], axis=1)  # (N, D+1)

        d = Xa.shape[1]
        lam = float(self.ridge_lambda)
        XtX = Xa.T @ Xa
        XtX += (lam * np.eye(d, dtype=np.float32))
        XtY = Xa.T @ Y
        W = np.linalg.solve(XtX, XtY).astype(np.float32)  # (D+1,2)

        self.W = W
        return W

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Предсказывает `(x, y)` по признакам (требует обученную `W`)."""
        if self.W is None:
            raise RuntimeError("Калибратор не обучен")
        f = np.asarray(features, dtype=np.float32).reshape(-1)
        if f.shape[0] != self.feature_dim:
            raise ValueError(f"Ожидали feature_dim={self.feature_dim}, получили {f.shape[0]}")
        xa = np.concatenate([f, np.array([1.0], dtype=np.float32)], axis=0)  # (D+1,)
        return xa @ self.W  # (2,)

    def save(self, path: Path) -> None:
        """Сохраняет калибровку в `.npz`."""
        if self.W is None:
            raise RuntimeError("Нечего сохранять (W is None)")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W=self.W.astype(np.float32),
            feature_dim=np.array([int(self.feature_dim)], dtype=np.int32),
            ridge_lambda=np.array([float(self.ridge_lambda)], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> "LinearCalibrator":
        """Загружает калибровку из `.npz`."""
        path = Path(path)
        npz = np.load(path)
        feature_dim = int(npz["feature_dim"][0])
        ridge_lambda = float(npz.get("ridge_lambda", np.array([1e-4], dtype=np.float32))[0])
        cal = cls(feature_dim=feature_dim, ridge_lambda=ridge_lambda)
        cal.W = npz["W"].astype(np.float32)
        return cal
