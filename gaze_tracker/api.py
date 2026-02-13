"""HTTP API + Web UI для локального трекера взгляда.

Этот модуль поднимает FastAPI‑приложение и отдаёт:
- Web UI (статические файлы: HTML/CSS/JS);
- JSON API для предсказания и калибровки.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from gaze_tracker.service import GazeService


def decode_image(data: bytes) -> np.ndarray | None:
    """Декодирует изображение (JPEG/PNG bytes) в BGR‑кадр OpenCV.

    Возвращает `None`, если декодирование не удалось.
    """
    # Локальный импорт: чтобы при отсутствии OpenCV ошибка была понятнее.
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def _web_dir() -> Path:
    """Возвращает путь к папке Web UI внутри пакета."""
    return Path(__file__).resolve().parent / "web"


def create_app(
    *,
    model_dir: Path | None = None,
    calib_path: Path | None = None,
    load_calibration: bool = False,
) -> FastAPI:
    """Создаёт FastAPI‑приложение с подключённым Web UI и API."""
    web_dir = _web_dir()
    static_dir = web_dir / "static"
    index_path = web_dir / "index.html"
    if not index_path.exists():
        raise RuntimeError(f"Не найден Web UI: {index_path}")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Инициализация сервиса на старте приложения."""
        _app.state.gaze = GazeService(
            model_dir=model_dir,
            calib_path=calib_path,
            load_calibration=load_calibration,
        )
        yield

    app = FastAPI(title="Gaze Tracker API", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        """Главная страница (Web UI)."""
        return FileResponse(index_path)

    @app.get("/api/health")
    def health() -> dict:
        """Проверка доступности сервиса."""
        return {"ok": True}

    @app.get("/api/calibration/status")
    def calibration_status() -> dict:
        """Текущее состояние калибровки."""
        service: GazeService = app.state.gaze
        return {
            "ok": True,
            "calibrated": service.calibrator.W is not None,
            "num_samples": service.calibrator.num_samples,
            "calib_path": str(service.calib_path),
            "saved_calib_exists": bool(service.calib_path.exists()),
            "calibration_loaded": bool(getattr(service, "calibration_loaded", False)),
        }

    @app.post("/api/calibration/reset")
    def calibration_reset() -> dict:
        """Сбросить калибровку (в памяти и на диске)."""
        service: GazeService = app.state.gaze
        service.reset_calibration()
        return {"ok": True}

    @app.post("/api/calibration/capture")
    async def calibration_capture(
        file: UploadFile = File(...),
        target_x: float = Form(...),
        target_y: float = Form(...),
    ) -> JSONResponse:
        """Добавить один сэмпл калибровки (кадр + координата целевой точки)."""
        service: GazeService = app.state.gaze
        data = await file.read()
        img = decode_image(data)
        if img is None:
            return JSONResponse({"ok": False, "error": "bad_image"}, status_code=400)
        return JSONResponse(service.add_calibration_sample(img, target_x=float(target_x), target_y=float(target_y)))

    @app.post("/api/calibration/fit")
    def calibration_fit() -> dict:
        """Обучить линейный калибратор по накопленным сэмплам."""
        service: GazeService = app.state.gaze
        try:
            return service.fit_calibration()
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/predict")
    async def predict(file: UploadFile = File(...)) -> JSONResponse:
        """Предсказать направление взгляда и/или координату на «экране» (после калибровки)."""
        service: GazeService = app.state.gaze
        data = await file.read()
        img = decode_image(data)
        if img is None:
            return JSONResponse({"ok": False, "error": "bad_image"}, status_code=400)
        return JSONResponse(service.predict(img))

    return app


app = create_app()


def main(argv: list[str]) -> int:
    """CLI‑точка входа для запуска API + Web UI."""
    parser = argparse.ArgumentParser(description="Запуск API + Web UI трекера взгляда.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-dir", type=Path, default=None, help="Куда кэшировать/скачивать предобученные модели")
    parser.add_argument("--calib-path", type=Path, default=None, help="Куда сохранять калибровку (.npz)")
    parser.add_argument("--load-calibration", action="store_true", help="Загрузить существующую калибровку (.npz) при старте")
    args = parser.parse_args(argv)

    app = create_app(
        model_dir=args.model_dir,
        calib_path=args.calib_path,
        load_calibration=bool(args.load_calibration),
    )
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


def cli() -> None:
    """Точка входа для команды `gaze-tracker`."""
    raise SystemExit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
