# Gaze Tracker

Локальный трекер направления взгляда для ПК (без собственного датасета обучения):

- **YuNet (ONNX)** — предобученная нейросеть для лица + 5 ориентиров.
- Best‑effort поиск центра зрачка (classic CV).
- **Калибровка пользователя** (9 точек) → маппинг в координаты «экрана».
- **API‑first** (FastAPI) + Web UI для визуализации и отладки.

> Важно: без обучения на датасете это не «идеальный eye‑tracking», а практичный базовый прототип:
> стрелка направления до калибровки и точка на «экране» после калибровки.

## Содержание

- [Быстрый старт](#быстрый-старт)
- [Калибровка](#калибровка)
- [Запуск в Docker](#запуск-в-docker)
- [API](#api)
- [Параметры запуска](#параметры-запуска)
- [Структура проекта](#структура-проекта)
- [Траблшутинг](#траблшутинг)

## Быстрый старт

### Windows (PowerShell)

```powershell
python -m pip install -r requirements.txt
python -m gaze_tracker --port 8000
```

Открыть в браузере: `http://127.0.0.1:8000/`

Первый запуск может скачать модель YuNet (ONNX) в кэш пользователя.

### Linux / macOS

```bash
python3 -m pip install -r requirements.txt
python3 -m gaze_tracker --port 8000
```

## Калибровка

1) Нажми `Start`
2) Нажми `Calibrate (9 pts)` — откроется полноэкранный оверлей
3) На каждой точке: смотри на оранжевую точку и нажимай `Capture` (можно `Space`)
4) После 9 точек нажми `Fit` — появится точка на «экране»

Совет по качеству:
- нормальный свет и открытые глаза сильно улучшают стабилизацию зрачка и калибровку.

## Запуск в Docker

### Build + Run

```powershell
docker build -t gaze-tracker .
docker run --rm -p 8000:8000 gaze-tracker
```

Открыть: `http://localhost:8000/`

### Docker Compose

```powershell
docker compose up --build
```

## API

Эндпоинты:
- `GET  /api/health`
- `POST /api/predict` (multipart: `file`)
- `POST /api/calibration/reset`
- `POST /api/calibration/capture` (multipart: `file`, `target_x`, `target_y`)
- `POST /api/calibration/fit`
- `GET  /api/calibration/status`

Пример (PowerShell): healthcheck

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

## Параметры запуска

```powershell
python -m gaze_tracker --help
```

Полезные флаги:
- `--host 127.0.0.1` — адрес бинда
- `--port 8000` — порт (если `winerror 10048`, поставь другой)
- `--model-dir <path>` — куда кэшировать/скачивать модели (по умолчанию кэш пользователя)
- `--calib-path <path>` — куда сохранять калибровку (`.npz`)
- `--load-calibration` — загрузить существующую калибровку при старте

## Структура проекта

```
gaze_tracker/
  api.py                 # FastAPI + статика (Web UI)
  service.py             # пайплайн: YuNet → глаза → зрачок → признаки → калибровка
  calibration.py         # линейная регрессия (ridge) для калибровки
  detectors/yunet.py     # YuNet (OpenCV Zoo, ONNX) + автоскачивание
  vision/pupil.py        # эвристический центр зрачка (classic CV)
  web/                   # UI (HTML/CSS/JS)
docs/
  requirements.md        # требования (таблица)
  requirements.xlsx      # те же требования в Excel
```

## Траблшутинг

### `winerror 10048` (порт занят)

Запусти на другом порту:

```powershell
python -m gaze_tracker --port 8010
```

### Камера не работает в браузере

- Разреши доступ к камере для `http://127.0.0.1:8000/`
- Открой именно `localhost/127.0.0.1` (для удалённых адресов браузер может требовать HTTPS)

### «Точка не двигается»

- До калибровки отображается **стрелка направления** (на видео‑оверлее).
- **Точка на “экране”** появляется только после калибровки `Calibrate → Capture ×9 → Fit`.
