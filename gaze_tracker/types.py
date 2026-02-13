"""Базовые типы и структуры данных проекта."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    """Ограничивающий прямоугольник (bounding box) в координатах изображения."""

    x0: float
    y0: float
    x1: float
    y1: float

    def clamp(self, w: int, h: int) -> "BBox":
        """Ограничивает bbox размерами изображения `w x h`."""
        x0 = max(0.0, min(float(w), float(self.x0)))
        y0 = max(0.0, min(float(h), float(self.y0)))
        x1 = max(0.0, min(float(w), float(self.x1)))
        y1 = max(0.0, min(float(h), float(self.y1)))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return BBox(x0, y0, x1, y1)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        """Возвращает `(x0, y0, x1, y1)` с округлением до целых (для slicing)."""
        return (int(round(self.x0)), int(round(self.y0)), int(round(self.x1)), int(round(self.y1)))

    def width(self) -> float:
        """Ширина bbox."""
        return max(0.0, float(self.x1) - float(self.x0))

    def height(self) -> float:
        """Высота bbox."""
        return max(0.0, float(self.y1) - float(self.y0))

    def center(self) -> tuple[float, float]:
        """Центр bbox."""
        return ((float(self.x0) + float(self.x1)) / 2.0, (float(self.y0) + float(self.y1)) / 2.0)
