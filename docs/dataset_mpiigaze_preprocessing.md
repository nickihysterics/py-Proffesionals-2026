# Обработка MPIIGaze для проекта

Этот документ фиксирует, как в проекте подготавливается датасет `MPIIGaze`.

## Что берём из исходников

Используем распакованный корень датасета:

- `MPIIGaze/Annotation Subset/pXX.txt`
- `MPIIGaze/Data/Original/pXX/dayYY/*.jpg`

`Annotation Subset` содержит координаты ключевых точек. На их основе строятся кропы лица и глаз.

## Формат строки аннотации

Строка из `Annotation Subset/pXX.txt`:

```text
day13/0203.jpg 614 355 684 357 769 368 842 379 643 524 768 535 640 356 803 375
```

В коде интерпретируем это так:

1. путь к кадру `dayXX/NNNN.jpg`
2. левый глаз: два уголка
3. правый глаз: два уголка
4. рот: левый и правый угол
5. левый и правый центр зрачка

## Где находится код

Конвертер:

- `gaze_tracker/datasets/mpiigaze_preprocess.py`

CLI команда (после `pip install -e .`):

- `gaze-mpiigaze-preprocess`

Также можно запускать без установки:

```powershell
python -m gaze_tracker.datasets.mpiigaze_preprocess --help
```

## Что делает конвертер

1. Читает все `pXX.txt` в `Annotation Subset`.
2. Для каждой строки ищет исходный кадр в `Data/Original/pXX/dayYY/`.
3. Строит:
   - `left_eye_bbox`
   - `right_eye_bbox`
   - `face_bbox`
4. Вырезает кропы:
   - `images/left_eye/*.jpg`
   - `images/right_eye/*.jpg`
   - `images/face/*.jpg`
5. Пишет `metadata.csv` с путями, bbox и нормированными координатами зрачка.

## Команды запуска

### Быстрый smoke-test

```powershell
python -m gaze_tracker.datasets.mpiigaze_preprocess `
  --mpiigaze-root MPIIGaze `
  --output-dir data/mpiigaze_processed_sample `
  --max-samples 200
```

### Полная обработка

```powershell
python -m gaze_tracker.datasets.mpiigaze_preprocess `
  --mpiigaze-root MPIIGaze `
  --output-dir data/mpiigaze_processed
```

### Только `metadata.csv` (без сохранения кропов)

```powershell
python -m gaze_tracker.datasets.mpiigaze_preprocess `
  --mpiigaze-root MPIIGaze `
  --output-dir data/mpiigaze_processed_meta_only `
  --no-images
```

## Формат результата

В `metadata.csv` пишутся:

- `sample_id`, `subject`
- `source_image_rel`, `source_image_abs`
- `face_image_rel`, `left_eye_image_rel`, `right_eye_image_rel`
- `left_pupil_x_norm`, `left_pupil_y_norm`
- `right_pupil_x_norm`, `right_pupil_y_norm`
- координаты `face_bbox`, `left_eye_bbox`, `right_eye_bbox`

Нормированные координаты зрачка считаются в системе соответствующего eye crop:

```python
left_pupil_x_norm = (left_pupil_x - left_eye_bbox_x0) / (left_eye_bbox_x1 - left_eye_bbox_x0)
left_pupil_y_norm = (left_pupil_y - left_eye_bbox_y0) / (left_eye_bbox_y1 - left_eye_bbox_y0)
```

После расчёта значения ограничиваются диапазоном `[0.0, 1.0]`.

## Важные замечания

- `MPIIGaze/` исключён из Git (`.gitignore`) и из Docker context (`.dockerignore`).
- Для конвертации нужен `opencv-python` (уже в `requirements.txt`).
- Если в аннотации строка битая или исходный кадр не найден, она пропускается и попадает в счётчик `skipped_*`.

