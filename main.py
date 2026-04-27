import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch
import torch.fx
import distutils.version

try:
    from openpyxl import load_workbook
except ImportError:  # pragma: no cover
    load_workbook = None

# Ensure project-root imports work when running this file directly.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
print(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from myscripts.seg_infer import (
    DatasetConfig,
    Detectron2SegInference,
    IOUType,
    ModelConfig,
    OutputOptions,
    SearchSpace,
    SlidingWindowConfig,
)


@dataclass
class InferenceSettings:
    project_root: str = PROJECT_ROOT
    model_config_rel: str = "./config_v1.yaml"
    checkpoint_rel: str = "./model_final.pth"
    device: str = "cuda:0"
    imports: tuple = ("maskdino",)
    tile: int = 1024
    overlap: int = 128
    resize: Optional[Tuple[int, int]] = None
    iou_type: IOUType = IOUType.IOU_Min
    score_thr: float = 0.25
    nms_iou_thr: float = 0.3
    base_keep_score: float = 0.01
    polygon_epsilon: float = 2.0


ERROR_PATH_COLUMN = "文件夹路径"
ERROR_PATH_PREFIX = "/media/ubuntu/农场数据采集硬盘1"


class InstanceSegService:
    def __init__(self, settings: InferenceSettings):
        self.settings = settings
        self.runner = self._build_runner(settings)
        self.ready = False

    def _build_runner(self, settings: InferenceSettings) -> Detectron2SegInference:
        dataset = DatasetConfig(yolo_root="")
        model = ModelConfig(
            project_root=settings.project_root,
            config_rel=settings.model_config_rel,
            checkpoint_rel=settings.checkpoint_rel,
            device=settings.device,
            imports=settings.imports,
        )
        sw = SlidingWindowConfig(tile=settings.tile, overlap=settings.overlap, pad_value=(114, 114, 114))
        search = SearchSpace(base_keep_score=settings.base_keep_score)
        outputs = OutputOptions(vis=False, excel=False, draw_scatter=False, out_dir="")
        runner = Detectron2SegInference(dataset, model, sw, search, outputs)

        base_infer_tile = runner.infer_tile

        def infer_tile_with_resize(tile_img: np.ndarray):
            orig_h, orig_w = tile_img.shape[:2]
            if settings.resize is None:
                resized_img = tile_img
                resize_w, resize_h = orig_w, orig_h
            else:
                resize_w, resize_h = int(settings.resize[0]), int(settings.resize[1])
                if resize_w <= 0 or resize_h <= 0:
                    raise ValueError(f"Invalid resize setting: {settings.resize}")
                if (orig_w, orig_h) == (resize_w, resize_h):
                    resized_img = tile_img
                else:
                    resized_img = cv2.resize(tile_img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

            instances = base_infer_tile(resized_img)
            sx = float(orig_w) / float(resize_w)
            sy = float(orig_h) / float(resize_h)

            restored = []
            for ins in instances:
                x1, y1, x2, y2 = ins["bbox"]
                mapped_bbox = [
                    float(max(0.0, min(orig_w, x1 * sx))),
                    float(max(0.0, min(orig_h, y1 * sy))),
                    float(max(0.0, min(orig_w, x2 * sx))),
                    float(max(0.0, min(orig_h, y2 * sy))),
                ]

                mask = ins["mask"].astype(np.uint8)
                if mask.shape[:2] != (orig_h, orig_w):
                    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                restored.append(
                    {
                        "bbox": mapped_bbox,
                        "score": float(ins.get("score", 0.0)),
                        "label": int(ins.get("label", 0)),
                        "mask": mask.astype(bool),
                    }
                )
            return restored

        runner.infer_tile = infer_tile_with_resize
        return runner

    def ensure_model(self):
        if self.ready:
            return
        self.runner.load_model()
        self.ready = True

    def infer_image(self, img_path: str):
        self.ensure_model()
        start = time.time()
        proposals = self.runner._proposals_for_image(img_path)
        final_dets = self.runner._apply_thresh_nms(
            proposals,
            self.settings.score_thr,
            self.settings.nms_iou_thr,
            self.settings.iou_type,
        )
        elapsed = time.time() - start
        return final_dets, elapsed

    @staticmethod
    def render_overlay(img_path: str, dets: List[Any]) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        overlay = img.copy()

        rng = np.random.default_rng(2026)
        for d in dets:
            color = tuple(int(x) for x in rng.integers(50, 256, size=3))
            y1, y2 = d.top, int(d.top + d.mask.shape[0])
            x1, x2 = d.left, int(d.left + d.mask.shape[1])
            roi = overlay[y1:y2, x1:x2]
            if roi.shape[:2] != d.mask.shape:
                continue
            m = d.mask
            roi[m] = (0.35 * roi[m] + 0.65 * np.array(color, dtype=np.uint8)).astype(np.uint8)

            contours, _ = cv2.findContours((m.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay[y1:y2, x1:x2], contours, -1, color, 2)

        return overlay

    @staticmethod
    def _mask_to_polygons(local_mask: np.ndarray, left: int, top: int, epsilon: float) -> List[List[float]]:
        contours, _ = cv2.findContours(
            (local_mask.astype(np.uint8) * 255),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons: List[List[float]] = []
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
            if approx.shape[0] < 3:
                approx = cnt

            pts = approx.reshape(-1, 2).astype(np.float32)
            pts[:, 0] += float(left)
            pts[:, 1] += float(top)
            poly = pts.flatten().tolist()
            if len(poly) >= 6:
                polygons.append(poly)
        return polygons

    def serialize_detections(self, dets: List[Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for i, d in enumerate(dets, start=1):
            bw = int(d.mask.shape[1])
            bh = int(d.mask.shape[0])
            items.append(
                {
                    "id": i,
                    "category_id": int(d.label),
                    "score": float(d.score),
                    "bbox": [float(d.left), float(d.top), float(bw), float(bh)],
                    "area": float(int(d.area)),
                    "segmentation": self._mask_to_polygons(
                        d.mask,
                        d.left,
                        d.top,
                        epsilon=self.settings.polygon_epsilon,
                    ),
                }
            )
        return items


def parse_bool(v: str) -> bool:
    s = v.strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def write_log_line(log_path: str, stem: str, status_code: int):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{stem}#{status_code}\n")


def normalize_error_path(raw_path: str) -> Optional[Path]:
    path_text = raw_path.strip().replace("\\", "/")
    if not path_text:
        return None

    if path_text == ERROR_PATH_PREFIX:
        return None
    if path_text.startswith(ERROR_PATH_PREFIX + "/"):
        path_text = path_text[len(ERROR_PATH_PREFIX) + 1 :]

    return Path(path_text.lstrip("/"))


def candidate_dirs_from_rel_path(root: Path, rel_path: Path) -> List[Path]:
    """Build candidate directories under root for a normalized error-file path."""
    root_resolved = root.resolve(strict=False)
    candidates: List[Path] = []
    seen: set[Path] = set()

    def _push(p: Path):
        p = p.resolve(strict=False)
        if p in seen:
            return
        seen.add(p)
        candidates.append(p)

    # Default behavior: treat rel_path as relative to scan root.
    _push(root_resolved / rel_path)

    # If input_dir is already a child root (e.g. .../2025定兴基地玉米收集数据),
    # trim the leading matching segment from rel_path.
    parts = rel_path.parts
    if parts:
        matched_indices = [i for i, part in enumerate(parts) if part == root_resolved.name]
        for i in matched_indices:
            trimmed = Path(*parts[i + 1 :]) if i + 1 < len(parts) else Path(".")
            _push(root_resolved / trimmed)

    return candidates


def load_skip_dirs_from_excel(error_file: str, root: Path) -> set[Path]:
    if load_workbook is None:
        print("openpyxl is not installed, skip error-file filtering.")
        return set()

    excel_path = Path(error_file).expanduser()
    if not excel_path.is_absolute():
        excel_path = (Path.cwd() / excel_path).resolve(strict=False)

    if not excel_path.exists():
        print(f"error_file not found: {excel_path}, skip filtering.")
        return set()

    try:
        wb = load_workbook(excel_path, read_only=True, data_only=True)
    except Exception as exc:
        print(f"failed to read error_file {excel_path}: {exc}")
        return set()

    ws = wb.active
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
    if not header:
        wb.close()
        print(f"{excel_path} has empty header, skip filtering.")
        return set()

    title_to_idx = {
        str(col).strip(): idx for idx, col in enumerate(header) if col is not None and str(col).strip()
    }
    if ERROR_PATH_COLUMN not in title_to_idx:
        wb.close()
        print(f"{excel_path} missing column '{ERROR_PATH_COLUMN}', skip filtering.")
        return set()

    col_idx = title_to_idx[ERROR_PATH_COLUMN]
    root_resolved = root.resolve(strict=False)
    skip_dirs: set[Path] = set()
    ignored_rows = 0

    for row in ws.iter_rows(min_row=2, values_only=True):
        value = row[col_idx] if col_idx < len(row) else None
        if value is None:
            continue

        rel_path = normalize_error_path(str(value))
        if rel_path is None:
            ignored_rows += 1
            continue

        matched = False
        for candidate in candidate_dirs_from_rel_path(root, rel_path):
            try:
                candidate.relative_to(root_resolved)
            except ValueError:
                continue

            if candidate.exists() and candidate.is_dir():
                skip_dirs.add(candidate)
                matched = True
                break

        if not matched:
            ignored_rows += 1

    wb.close()
    print(f"Loaded {len(skip_dirs)} skipped directories from error_file (ignored rows: {ignored_rows}).")
    return skip_dirs


def is_under_skipped_dir(path: Path, scan_root: Path, skip_dirs: set[Path]) -> bool:
    current = path.resolve(strict=False)
    root_resolved = scan_root.resolve(strict=False)
    while True:
        if current in skip_dirs:
            return True
        if current == root_resolved or current.parent == current:
            return False
        current = current.parent


def list_images_in_dir_with_prune(root: Path, skip_dirs: set[Path], valid_extensions: set[str]) -> List[str]:
    root_resolved = root.resolve(strict=False)
    image_paths: List[str] = []

    for current_dir, dirs, files in os.walk(root_resolved, topdown=True):
        current_path = Path(current_dir).resolve(strict=False)

        # Prune directories listed in error_file so their subtree is never scanned.
        if is_under_skipped_dir(current_path, root_resolved, skip_dirs):
            dirs[:] = []
            continue

        dirs[:] = [d for d in dirs if not is_under_skipped_dir(current_path / d, root_resolved, skip_dirs)]

        for file_name in files:
            file_path = current_path / file_name
            if file_path.suffix.lower() not in valid_extensions:
                continue
            image_paths.append(str(file_path))

    image_paths.sort()
    return image_paths


def _format_hms(seconds: float) -> str:
    secs = max(0, int(seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_infer_progress(
    current: int,
    total: int,
    success: int,
    failed: int,
    width: int = 28,
    *,
    last_sec: Optional[float] = None,
    elapsed_sec: Optional[float] = None,
    eta_sec: Optional[float] = None,
):
    if total <= 0:
        return

    elapsed_text = _format_hms(elapsed_sec or 0.0) if elapsed_sec is not None else "--:--"
    eta_text = _format_hms(eta_sec or 0.0) if eta_sec is not None else "--:--"
    per_text = f"{last_sec:.2f}s/img" if last_sec is not None else "--.--s/img"

    speed_text = "--.--it/s"
    if elapsed_sec is not None and elapsed_sec > 0 and current > 0:
        speed_text = f"{(current / elapsed_sec):.2f}it/s"

    if sys.stdout.isatty():
        ratio = max(0.0, min(1.0, float(current) / float(total)))
        filled = int(round(width * ratio))
        bar = "=" * filled + "-" * max(0, width - filled)
        line = (
            f"\rInfer progress [{bar}] {current}/{total} "
            f"({ratio * 100:5.1f}%) ok:{success} fail:{failed} "
            f"| {per_text} | elapsed {elapsed_text} | eta {eta_text} | {speed_text}"
        )
        end = "\n" if current >= total else ""
        print(line, end=end, flush=True)
    else:
        print(
            f"Infer progress {current}/{total} ok:{success} fail:{failed} "
            f"| {per_text} | elapsed {elapsed_text} | eta {eta_text} | {speed_text}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch instance segmentation runner")
    parser.add_argument("-i", "--input_dir", required=True, help="Input image directory")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory")
    parser.add_argument("-f", "--format", default="", help="Reserved argument; ignored")
    parser.add_argument(
        "-s",
        "--save_images",
        type=parse_bool,
        default=True,
        help="Whether to save overlay images, default: true",
    )
    parser.add_argument(
        "-r",
        "--error_file",
        default=None,
        help="Optional Excel file used to skip listed directories",
    )

    parser.add_argument("--model_config_rel", default=InferenceSettings.model_config_rel)
    parser.add_argument("--checkpoint_rel", default=InferenceSettings.checkpoint_rel)
    parser.add_argument("--device", default=InferenceSettings.device)
    parser.add_argument("--tile", type=int, default=InferenceSettings.tile)
    parser.add_argument("--overlap", type=int, default=InferenceSettings.overlap)
    parser.add_argument("--score_thr", type=float, default=InferenceSettings.score_thr)
    parser.add_argument("--nms_iou_thr", type=float, default=InferenceSettings.nms_iou_thr)
    parser.add_argument("--base_keep_score", type=float, default=InferenceSettings.base_keep_score)
    parser.add_argument("--polygon_epsilon", type=float, default=InferenceSettings.polygon_epsilon)
    return parser


def main():
    args, _ = build_parser().parse_known_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    log_path = os.path.join(output_dir, "app.log")
    output_json_path = os.path.join(output_dir, "output.json")
    output_csv_path = os.path.join(output_dir, "output.csv")

    save_images = bool(args.save_images)
    saved_images_dir = os.path.join(output_dir, "saved_images")
    if save_images:
        os.makedirs(saved_images_dir, exist_ok=True)
    input_root_name = os.path.basename(os.path.normpath(input_dir))

    settings = InferenceSettings(
        project_root=PROJECT_ROOT,
        model_config_rel=args.model_config_rel,
        checkpoint_rel=args.checkpoint_rel,
        device=args.device,
        tile=args.tile,
        overlap=args.overlap,
        score_thr=args.score_thr,
        nms_iou_thr=args.nms_iou_thr,
        base_keep_score=args.base_keep_score,
        polygon_epsilon=args.polygon_epsilon,
    )

    service = InstanceSegService(settings)

    # Filter images by supported extensions, case-insensitive
    valid_extensions = {".jpg", ".jpeg", ".png"}
    skip_dirs: set[Path] = set()
    if args.error_file:
        skip_dirs = load_skip_dirs_from_excel(args.error_file, Path(input_dir))

    image_paths = list_images_in_dir_with_prune(Path(input_dir), skip_dirs, valid_extensions)
    if skip_dirs:
        print(f"Collected {len(image_paths)} images after pruning directories from error_file.")

    records = []
    total_images = len(image_paths)
    success_count = 0
    failed_count = 0
    batch_start_ts = time.time()
    if total_images > 0:
        print_infer_progress(0, total_images, success_count, failed_count, elapsed_sec=0.0, eta_sec=None)

    for idx, img_path in enumerate(image_paths, start=1):
        iter_start_ts = time.time()
        abs_img_path = os.path.abspath(img_path)
        # Keep input_dir as the root so saved_images starts from input folder name.
        rel_path = os.path.relpath(abs_img_path, start=input_dir)
        rel_dir = os.path.dirname(rel_path)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        write_log_line(log_path, stem, 1)

        try:
            dets, elapsed = service.infer_image(img_path)
            saved_overlay = ""
            if save_images:
                overlay = service.render_overlay(img_path, dets)
                target_dir = os.path.join(saved_images_dir, input_root_name, rel_dir)
                os.makedirs(target_dir, exist_ok=True)
                ext = os.path.splitext(os.path.basename(img_path))[1] or ".jpg"
                saved_overlay = os.path.join(target_dir, f"{stem}{ext}")
                cv2.imwrite(saved_overlay, overlay)

            record = {
                "filename": stem,
                "counts": len(dets),
                "infer_time_sec": round(float(elapsed), 3),
                "status": "success",
                "saved_image": saved_overlay,
                "detections": service.serialize_detections(dets),
            }
            records.append(record)
            write_log_line(log_path, stem, 2)
            success_count += 1
        except Exception as e:
            records.append(
                {
                    "filename": stem,
                    "counts": "",
                    "infer_time_sec": "",
                    "status": "failed",
                    "error": str(e),
                    "saved_image": "",
                    "detections": [],
                }
            )
            write_log_line(log_path, stem, 3)
            failed_count += 1

        iter_elapsed = max(0.0, time.time() - iter_start_ts)
        elapsed = max(0.0, time.time() - batch_start_ts)
        avg_per_image = (elapsed / idx) if idx > 0 else 0.0
        eta = avg_per_image * max(0, total_images - idx)
        print_infer_progress(
            idx,
            total_images,
            success_count,
            failed_count,
            last_sec=iter_elapsed,
            elapsed_sec=elapsed,
            eta_sec=eta,
        )

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"results": records}, f, ensure_ascii=False, indent=2)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["filename", "counts", "infer_time_sec", "status", "error", "saved_image"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "filename": r.get("filename", ""),
                    "counts": r.get("counts", ""),
                    "infer_time_sec": r.get("infer_time_sec", ""),
                    "status": r.get("status", ""),
                    "error": r.get("error", ""),
                    "saved_image": r.get("saved_image", ""),
                }
            )


if __name__ == "__main__":
    main()

