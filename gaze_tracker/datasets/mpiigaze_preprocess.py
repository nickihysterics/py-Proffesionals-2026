"""Конвертация MPIIGaze в формат кропов и metadata.csv для проекта.

Пайплайн использует `Annotation Subset`:
- читает строки `pXX.txt`;
- находит исходный кадр в `Data/Original/pXX/...`;
- строит bbox глаз и лица по аннотациям;
- сохраняет кропы `left_eye`, `right_eye`, `face`;
- пишет `metadata.csv` с путями и нормированными координатами зрачка.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None

from gaze_tracker.types import BBox


ANNOTATION_FILES_GLOB = "p*.txt"


@dataclass(frozen=True)
class AnnotationRow:
    """Одна строка аннотации MPIIGaze (Annotation Subset)."""

    subject: str
    rel_image_path: str
    left_eye_corner_a: tuple[float, float]
    left_eye_corner_b: tuple[float, float]
    right_eye_corner_a: tuple[float, float]
    right_eye_corner_b: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]
    left_pupil: tuple[float, float]
    right_pupil: tuple[float, float]


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])
    return float(math.sqrt(dx * dx + dy * dy))


def _center(p1: tuple[float, float], p2: tuple[float, float]) -> tuple[float, float]:
    return ((float(p1[0]) + float(p2[0])) * 0.5, (float(p1[1]) + float(p2[1])) * 0.5)


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def parse_annotation_line(subject: str, raw_line: str) -> AnnotationRow | None:
    """Парсит строку `Annotation Subset/pXX.txt`."""
    line = raw_line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) != 17:
        return None

    rel_image_path = parts[0]
    values = [float(x) for x in parts[1:]]

    return AnnotationRow(
        subject=subject,
        rel_image_path=rel_image_path,
        left_eye_corner_a=(values[0], values[1]),
        left_eye_corner_b=(values[2], values[3]),
        right_eye_corner_a=(values[4], values[5]),
        right_eye_corner_b=(values[6], values[7]),
        mouth_left=(values[8], values[9]),
        mouth_right=(values[10], values[11]),
        left_pupil=(values[12], values[13]),
        right_pupil=(values[14], values[15]),
    )


def compute_eye_box(
    corner_a: tuple[float, float],
    corner_b: tuple[float, float],
    width: int,
    height: int,
) -> BBox:
    """Строит bbox глаза по двум уголкам глаза."""
    cx, cy = _center(corner_a, corner_b)
    eye_width = max(10.0, 2.4 * _distance(corner_a, corner_b))
    eye_height = max(8.0, 1.8 * _distance(corner_a, corner_b))

    # Небольшая асимметрия по Y: оставляем больше нижнего века.
    x0 = cx - eye_width * 0.5
    x1 = cx + eye_width * 0.5
    y0 = cy - eye_height * 0.45
    y1 = cy + eye_height * 0.55
    return BBox(x0, y0, x1, y1).clamp(width, height)


def compute_face_box(row: AnnotationRow, width: int, height: int) -> BBox:
    """Строит bbox лица по глазным и ротовым точкам."""
    left_eye_center = _center(row.left_eye_corner_a, row.left_eye_corner_b)
    right_eye_center = _center(row.right_eye_corner_a, row.right_eye_corner_b)
    inter_eye = max(20.0, _distance(left_eye_center, right_eye_center))

    points = [
        row.left_eye_corner_a,
        row.left_eye_corner_b,
        row.right_eye_corner_a,
        row.right_eye_corner_b,
        row.mouth_left,
        row.mouth_right,
    ]

    x_values = [float(p[0]) for p in points]
    y_values = [float(p[1]) for p in points]

    x0 = min(x_values) - 0.35 * inter_eye
    x1 = max(x_values) + 0.35 * inter_eye
    y0 = min(y_values) - 0.65 * inter_eye
    y1 = max(y_values) + 0.55 * inter_eye
    return BBox(x0, y0, x1, y1).clamp(width, height)


def crop_by_bbox(image, bbox: BBox):
    """Вырезает изображение по `BBox`."""
    x0, y0, x1, y1 = bbox.as_int_tuple()
    return image[y0:y1, x0:x1]


def preprocess_mpiigaze(
    mpiigaze_root: Path,
    output_dir: Path,
    *,
    max_samples: int | None = None,
    save_images: bool = True,
    jpeg_quality: int = 95,
) -> dict[str, int]:
    """Обрабатывает MPIIGaze и сохраняет `metadata.csv` + кропы."""
    if cv2 is None:
        raise RuntimeError("OpenCV не установлен. Установи: python -m pip install opencv-python")

    annotation_dir = mpiigaze_root / "Annotation Subset"
    original_dir = mpiigaze_root / "Data" / "Original"
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Не найдена папка аннотаций: {annotation_dir}")
    if not original_dir.exists():
        raise FileNotFoundError(f"Не найдена папка оригинальных кадров: {original_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    face_dir = output_dir / "images" / "face"
    left_eye_dir = output_dir / "images" / "left_eye"
    right_eye_dir = output_dir / "images" / "right_eye"

    if save_images:
        face_dir.mkdir(parents=True, exist_ok=True)
        left_eye_dir.mkdir(parents=True, exist_ok=True)
        right_eye_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    processed = 0
    skipped_missing_image = 0
    skipped_bad_row = 0

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "subject",
                "source_image_rel",
                "source_image_abs",
                "face_image_rel",
                "left_eye_image_rel",
                "right_eye_image_rel",
                "left_pupil_x_norm",
                "left_pupil_y_norm",
                "right_pupil_x_norm",
                "right_pupil_y_norm",
                "face_bbox_x0",
                "face_bbox_y0",
                "face_bbox_x1",
                "face_bbox_y1",
                "left_eye_bbox_x0",
                "left_eye_bbox_y0",
                "left_eye_bbox_x1",
                "left_eye_bbox_y1",
                "right_eye_bbox_x0",
                "right_eye_bbox_y0",
                "right_eye_bbox_x1",
                "right_eye_bbox_y1",
            ],
        )
        writer.writeheader()

        annotation_files = sorted(annotation_dir.glob(ANNOTATION_FILES_GLOB))
        for annotation_file in annotation_files:
            subject = annotation_file.stem
            with annotation_file.open("r", encoding="utf-8") as af:
                for raw_line in af:
                    if max_samples is not None and processed >= max_samples:
                        break

                    row = parse_annotation_line(subject, raw_line)
                    if row is None:
                        skipped_bad_row += 1
                        continue

                    source_image = original_dir / subject / row.rel_image_path
                    if not source_image.exists():
                        skipped_missing_image += 1
                        continue

                    image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
                    if image is None:
                        skipped_missing_image += 1
                        continue

                    image_h, image_w = image.shape[:2]
                    left_eye_box = compute_eye_box(row.left_eye_corner_a, row.left_eye_corner_b, image_w, image_h)
                    right_eye_box = compute_eye_box(row.right_eye_corner_a, row.right_eye_corner_b, image_w, image_h)
                    face_box = compute_face_box(row, image_w, image_h)

                    left_crop = crop_by_bbox(image, left_eye_box)
                    right_crop = crop_by_bbox(image, right_eye_box)
                    face_crop = crop_by_bbox(image, face_box)

                    if left_crop.size == 0 or right_crop.size == 0 or face_crop.size == 0:
                        skipped_bad_row += 1
                        continue

                    day_part = Path(row.rel_image_path).parent.name
                    frame_name = Path(row.rel_image_path).stem
                    sample_id = f"{subject}_{day_part}_{frame_name}"

                    face_rel = f"images/face/{sample_id}.jpg"
                    left_rel = f"images/left_eye/{sample_id}.jpg"
                    right_rel = f"images/right_eye/{sample_id}.jpg"

                    if save_images:
                        cv2.imwrite(
                            str(output_dir / face_rel),
                            face_crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                        )
                        cv2.imwrite(
                            str(output_dir / left_rel),
                            left_crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                        )
                        cv2.imwrite(
                            str(output_dir / right_rel),
                            right_crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                        )

                    lx0, ly0, lx1, ly1 = left_eye_box.as_int_tuple()
                    rx0, ry0, rx1, ry1 = right_eye_box.as_int_tuple()
                    fx0, fy0, fx1, fy1 = face_box.as_int_tuple()

                    left_w = max(1.0, float(lx1 - lx0))
                    left_h = max(1.0, float(ly1 - ly0))
                    right_w = max(1.0, float(rx1 - rx0))
                    right_h = max(1.0, float(ry1 - ry0))

                    left_pupil_x = _clamp01((float(row.left_pupil[0]) - float(lx0)) / left_w)
                    left_pupil_y = _clamp01((float(row.left_pupil[1]) - float(ly0)) / left_h)
                    right_pupil_x = _clamp01((float(row.right_pupil[0]) - float(rx0)) / right_w)
                    right_pupil_y = _clamp01((float(row.right_pupil[1]) - float(ry0)) / right_h)

                    writer.writerow(
                        {
                            "sample_id": sample_id,
                            "subject": subject,
                            "source_image_rel": row.rel_image_path,
                            "source_image_abs": str(source_image.resolve()),
                            "face_image_rel": face_rel if save_images else "",
                            "left_eye_image_rel": left_rel if save_images else "",
                            "right_eye_image_rel": right_rel if save_images else "",
                            "left_pupil_x_norm": f"{left_pupil_x:.6f}",
                            "left_pupil_y_norm": f"{left_pupil_y:.6f}",
                            "right_pupil_x_norm": f"{right_pupil_x:.6f}",
                            "right_pupil_y_norm": f"{right_pupil_y:.6f}",
                            "face_bbox_x0": fx0,
                            "face_bbox_y0": fy0,
                            "face_bbox_x1": fx1,
                            "face_bbox_y1": fy1,
                            "left_eye_bbox_x0": lx0,
                            "left_eye_bbox_y0": ly0,
                            "left_eye_bbox_x1": lx1,
                            "left_eye_bbox_y1": ly1,
                            "right_eye_bbox_x0": rx0,
                            "right_eye_bbox_y0": ry0,
                            "right_eye_bbox_x1": rx1,
                            "right_eye_bbox_y1": ry1,
                        }
                    )
                    processed += 1

                if max_samples is not None and processed >= max_samples:
                    break

    return {
        "processed": processed,
        "skipped_missing_image": skipped_missing_image,
        "skipped_bad_row": skipped_bad_row,
    }


def build_parser() -> argparse.ArgumentParser:
    """Возвращает CLI parser для конвертера MPIIGaze."""
    parser = argparse.ArgumentParser(description="Конвертация MPIIGaze в формат кропов + metadata.csv")
    parser.add_argument(
        "--mpiigaze-root",
        type=Path,
        default=Path("MPIIGaze"),
        help="Путь к корню распакованного MPIIGaze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mpiigaze_processed"),
        help="Путь, куда сохранить metadata.csv и кропы",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Ограничение на число обработанных строк (для smoke-теста)",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Считать bbox/лейблы и писать только metadata.csv, без сохранения кропов",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Качество JPEG для сохранённых кропов (1..100)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint конвертера MPIIGaze."""
    args = build_parser().parse_args(argv)

    if not (1 <= int(args.jpeg_quality) <= 100):
        raise ValueError("--jpeg-quality должен быть в диапазоне 1..100")

    stats = preprocess_mpiigaze(
        mpiigaze_root=Path(args.mpiigaze_root),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        save_images=not bool(args.no_images),
        jpeg_quality=int(args.jpeg_quality),
    )

    print("MPIIGaze preprocessing completed")
    print(f"processed={stats['processed']}")
    print(f"skipped_missing_image={stats['skipped_missing_image']}")
    print(f"skipped_bad_row={stats['skipped_bad_row']}")
    print(f"output={Path(args.output_dir).resolve()}")
    return 0


def cli() -> None:
    """Console script entrypoint."""
    raise SystemExit(main())


if __name__ == "__main__":
    cli()

