import csv
import json
import os
import random
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

# Make project root importable when running from tools/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    """GUI default settings; update these paths to your local environment."""

    project_root: str = PROJECT_ROOT
    model_config_rel: str = "/public/home/zncui/pycharm_project/MaskDINO/output/maskdino_swinl_maize_rice_baselr_1e-5/config_v1.yaml"
    checkpoint_rel: str = "/public/home/zncui/pycharm_project/MaskDINO/output/maskdino_swinl_maize_rice_baselr_1e-5/model_final.pth"
    device: str = "cuda:0"
    imports: tuple = ("maskdino",)
    tile: int = 1024 # 800
    overlap: int = 128 # 256
    resize: Optional[Tuple[int, int]] = None  # None -> infer at original tile size
    iou_type: IOUType = IOUType.IOU_Min
    score_thr: float = 0.25
    nms_iou_thr: float = 0.3
    base_keep_score: float = 0.01


class InstanceSegService:
    """Reuse seg_infer.py inference pipeline without CLI."""

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

        # Keep seg_infer pipeline, but resize model input and map outputs back to original tile coords.
        base_infer_tile = runner.infer_tile

        def infer_tile_with_resize(tile_img: np.ndarray):
            orig_h, orig_w = tile_img.shape[:2]
            if settings.resize is None:
                resized_img = tile_img
                resize_w, resize_h = orig_w, orig_h
            else:
                if len(settings.resize) != 2:
                    raise ValueError(f"Invalid resize setting: {settings.resize}")
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

    def infer_and_save(self, img_path: str, out_dir: str) -> tuple[int, str, float]:
        self.ensure_model()
        start_time = time.time()

        proposals = self.runner._proposals_for_image(img_path)
        final_dets = self.runner._apply_thresh_nms(
            proposals,
            self.settings.score_thr,
            self.settings.nms_iou_thr,
            self.settings.iou_type,
        )
        count = len(final_dets)

        base = os.path.splitext(os.path.basename(img_path))[0]
        overlay_dir = os.path.join(out_dir, "overlay")
        json_dir = os.path.join(out_dir, "json")
        os.makedirs(overlay_dir, exist_ok=True)
        os.makedirs(json_dir, exist_ok=True)

        overlay_path = os.path.join(overlay_dir, f"{base}_overlay.png")
        json_path = os.path.join(json_dir, f"{base}.json")

        vis = self._render_overlay(img_path, final_dets)
        cv2.imwrite(overlay_path, vis)
        self._save_coco_like_json(img_path, final_dets, json_path)

        elapsed = time.time() - start_time
        self._append_csv(out_dir, base, count, elapsed)
        return count, overlay_path, elapsed

    @staticmethod
    def _mask_to_polygons(local_mask: np.ndarray, left: int, top: int) -> list[list[float]]:
        m8 = (local_mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons: list[list[float]] = []
        for cnt in contours:
            if cnt.shape[0] < 3:
                continue
            pts = cnt.reshape(-1, 2).astype(np.float32)
            pts[:, 0] += float(left)
            pts[:, 1] += float(top)
            poly = pts.flatten().tolist()
            if len(poly) >= 6:
                polygons.append(poly)
        return polygons

    @classmethod
    def _save_coco_like_json(cls, img_path: str, dets: list, json_path: str):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        h, w = img.shape[:2]
        image_name = os.path.basename(img_path)

        categories = sorted({int(d.label) for d in dets})
        coco = {
            "images": [
                {
                    "id": 1,
                    "file_name": image_name,
                    "width": int(w),
                    "height": int(h),
                }
            ],
            "annotations": [],
            "categories": [{"id": cid, "name": f"class_{cid}"} for cid in categories],
        }

        for ann_id, d in enumerate(dets, start=1):
            bw = int(d.mask.shape[1])
            bh = int(d.mask.shape[0])
            bbox = [float(d.left), float(d.top), float(bw), float(bh)]
            polygons = cls._mask_to_polygons(d.mask, d.left, d.top)
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": 1,
                    "category_id": int(d.label),
                    "score": float(d.score),
                    "bbox": bbox,
                    "area": float(int(d.area)),
                    "iscrowd": 0,
                    "segmentation": polygons,
                }
            )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _render_overlay(img_path: str, dets: list) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)

        rng = random.Random(2026)
        overlay = img.copy()

        for d in dets:
            color = (
                rng.randint(50, 255),
                rng.randint(50, 255),
                rng.randint(50, 255),
            )
            y1, y2 = d.top, int(d.top + d.mask.shape[0])
            x1, x2 = d.left, int(d.left + d.mask.shape[1])
            roi = overlay[y1:y2, x1:x2]
            if roi.shape[:2] != d.mask.shape:
                continue
            m = d.mask
            roi[m] = (0.35 * roi[m] + 0.65 * np.array(color, dtype=np.uint8)).astype(np.uint8)

            m8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay[y1:y2, x1:x2], contours, -1, color, 2)

        return overlay

    @staticmethod
    def _append_csv(out_dir: str, filename_no_ext: str, count: int, elapsed: float):
        csv_path = os.path.join(out_dir, "results.csv")
        rows = []
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)

        # Upsert by filename
        updated = False
        for r in rows:
            if r.get("filename") == filename_no_ext:
                r["counts"] = str(count)
                r["infer_time_sec"] = f"{elapsed:.3f}"
                updated = True
                break

        if not updated:
            rows.append(
                {
                    "filename": filename_no_ext,
                    "counts": str(count),
                    "infer_time_sec": f"{elapsed:.3f}",
                }
            )

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "counts", "infer_time_sec"])
            writer.writeheader()
            writer.writerows(rows)


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("MaskDINO Instance Segmentation Tool")
        self.root.geometry("1400x900")

        self.tk_img_orig = None
        self.tk_img_overlay = None
        self.raw_img_orig = None
        self.raw_img_overlay = None

        self.current_path = ""
        self.image_paths: List[str] = []
        self.current_index = -1
        self.output_dir = ""

        self.model: Optional[InstanceSegService] = None
        self.settings = InferenceSettings()

        self.setup_ui()

    def setup_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, pady=4, padx=8)

        btn_style = {
            "font": ("Arial", 14),
            "padx": 8,
            "pady": 4,
            "height": 1,
        }
        pack_opts = {"side": tk.LEFT, "padx": 3}

        tk.Button(top_frame, text="Select Image", command=self.choose_image, **btn_style).pack(**pack_opts)
        tk.Button(top_frame, text="Select Directory", command=self.choose_directory, **btn_style).pack(**pack_opts)
        tk.Frame(top_frame, width=16).pack(side=tk.LEFT)
        tk.Button(top_frame, text="Select Output Directory", command=self.choose_output_dir, **btn_style).pack(**pack_opts)
        tk.Button(top_frame, text="Inference (Current)", command=self.run_inference, bg="#e1f5fe", **btn_style).pack(**pack_opts)
        tk.Button(top_frame, text="Inference (Total)", command=self.run_inference_all, bg="#ffebee", **btn_style).pack(**pack_opts)

        self.main_paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashwidth=4)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        top_content = tk.Frame(self.main_paned)
        self.main_paned.add(top_content, height=500)

        list_frame = tk.Frame(top_content, width=220)
        list_frame.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(list_frame, text="File List").pack(anchor=tk.W)

        list_scroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(list_frame, yscrollcommand=list_scroll.set, width=28)
        list_scroll.config(command=self.listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        images_frame = tk.Frame(top_content)
        images_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)

        def create_img_container(parent, title, bg_color):
            frame = tk.Frame(parent, bg="white", bd=1, relief=tk.SUNKEN)
            frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
            tk.Label(frame, text=title, bg=bg_color).pack(fill=tk.X)
            lbl = tk.Label(frame, bg="gray")
            lbl.pack(expand=True, fill=tk.BOTH)
            return lbl

        self.orig_label = create_img_container(images_frame, "Original Image", "#eeeeee")
        self.overlay_label = create_img_container(images_frame, "Instance Overlay", "#e3f2fd")

        bottom_content = tk.Frame(self.main_paned)
        self.main_paned.add(bottom_content)
        tk.Label(bottom_content, text="Results.csv Data Preview").pack(anchor=tk.W)

        cols = ("filename", "counts", "infer_time_sec")
        self.tree = ttk.Treeview(bottom_content, columns=cols, show="headings", height=8)
        self.tree.heading("filename", text="filename")
        self.tree.heading("counts", text="counts")
        self.tree.heading("infer_time_sec", text="infer_time_sec")
        self.tree.column("filename", width=300)
        self.tree.column("counts", width=120)
        self.tree.column("infer_time_sec", width=140)

        tree_scroll = tk.Scrollbar(bottom_content, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, pady=2, padx=10)
        self.path_label = tk.Label(status_frame, text="Current Image: None", fg="blue")
        self.path_label.pack(side=tk.LEFT)
        self.status_label = tk.Label(status_frame, text="", fg="purple")
        self.status_label.pack(side=tk.RIGHT)

    def ensure_model(self):
        if self.model is not None:
            return
        try:
            self.status_label.config(text="Initializing model...")
            self.root.update()
            self.model = InstanceSegService(self.settings)
            self.model.ensure_model()
            self.status_label.config(text="Model ready")
        except Exception as e:
            self.model = None
            messagebox.showerror("Error", f"Model initialization failed: {e}")

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.image_paths = [path]
        self.current_index = 0
        self.refresh_listbox()
        self.load_image_by_index(0)

    def choose_directory(self):
        d = filedialog.askdirectory()
        if not d:
            return
        files = sorted(
            [
                os.path.join(d, f)
                for f in os.listdir(d)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
        )
        if not files:
            messagebox.showinfo("Tip", "No image file found in selected directory.")
            return
        self.image_paths = files
        self.current_index = 0
        self.refresh_listbox()
        self.load_image_by_index(0)

    def choose_output_dir(self):
        d = filedialog.askdirectory()
        if not d:
            return
        self.output_dir = d
        self.status_label.config(text=f"Output directory: {d}")
        self.refresh_csv_table()
        if self.current_index >= 0:
            self.load_image_by_index(self.current_index)

    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for p in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(p))
        if self.image_paths:
            self.listbox.select_set(0)

    def on_listbox_select(self, event):
        sel = self.listbox.curselection()
        if sel:
            self.current_index = sel[0]
            self.load_image_by_index(self.current_index)

    def load_image_by_index(self, idx: int):
        if idx < 0 or idx >= len(self.image_paths):
            return
        path = self.image_paths[idx]
        self.current_path = path
        self.path_label.config(text=f"Current: {path}")

        try:
            self.raw_img_orig = Image.open(path)
            self.update_single_image(self.raw_img_orig, self.orig_label, is_result=False)
        except Exception as e:
            messagebox.showerror("Error", f"Load image failed: {e}")
            return

        self.raw_img_overlay = None
        if self.output_dir:
            filename_no_ext = os.path.splitext(os.path.basename(path))[0]
            overlay_path = os.path.join(self.output_dir, "overlay", f"{filename_no_ext}_overlay.png")
            if os.path.exists(overlay_path):
                try:
                    self.raw_img_overlay = Image.open(overlay_path)
                except Exception:
                    self.raw_img_overlay = None

        self.update_single_image(self.raw_img_overlay, self.overlay_label, is_result=True)

    def update_single_image(self, pil_img, label_widget, is_result=False):
        if pil_img is None:
            label_widget.config(image="", text="No Result" if is_result else "Load Failed")
            label_widget.image = None
            return

        w = label_widget.winfo_width()
        h = label_widget.winfo_height()
        if w < 10 or h < 10:
            w, h = 300, 300

        img_w, img_h = pil_img.size
        scale = min(w / img_w, h / img_h)
        new_w, new_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)

        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img
        if label_widget == self.orig_label:
            self.tk_img_orig = tk_img
        elif label_widget == self.overlay_label:
            self.tk_img_overlay = tk_img

    def refresh_csv_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.output_dir:
            return

        csv_path = os.path.join(self.output_dir, "results.csv")
        if not os.path.exists(csv_path):
            return

        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for row in reversed(rows):
                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        row.get("filename", ""),
                        row.get("counts", ""),
                        row.get("infer_time_sec", ""),
                    ),
                )
        except Exception as e:
            messagebox.showerror("Error", f"Read CSV failed: {e}")

    def _load_completed_filenames(self) -> set[str]:
        """Load completed image stems from results.csv (filename column)."""
        completed: set[str] = set()
        if not self.output_dir:
            return completed

        csv_path = os.path.join(self.output_dir, "results.csv")
        if not os.path.exists(csv_path):
            return completed

        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = (row.get("filename") or "").strip()
                    if name:
                        completed.add(name)
        except Exception as e:
            messagebox.showwarning("Warning", f"Read results.csv failed, continue without skip: {e}")

        return completed

    def run_inference(self):
        if not self.current_path or not self.output_dir:
            messagebox.showwarning("Warning", "Please select image and output directory first.")
            return

        self.ensure_model()
        if not self.model:
            return

        try:
            self.status_label.config(text="Inferencing current image...")
            self.root.update()
            count, _, elapsed = self.model.infer_and_save(self.current_path, self.output_dir)
            self.status_label.config(text=f"Done. Count={count}, time={elapsed:.2f}s")
            self.load_image_by_index(self.current_index)
            self.refresh_csv_table()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_inference_all(self):
        if not self.image_paths or not self.output_dir:
            messagebox.showwarning("Warning", "Please select images and output directory first.")
            return

        self.ensure_model()
        if not self.model:
            return

        completed = self._load_completed_filenames()

        ok = 0
        skipped = 0
        failed = 0
        total = len(self.image_paths)
        for i, path in enumerate(self.image_paths, 1):
            base = os.path.basename(path)
            stem = os.path.splitext(base)[0]

            if stem in completed:
                skipped += 1
                self.status_label.config(text=f"Skipping ({i}/{total}): {base}")
                self.root.update()
                continue

            self.status_label.config(text=f"Processing ({i}/{total}): {base}")
            self.root.update()
            try:
                self.model.infer_and_save(path, self.output_dir)
                ok += 1
                completed.add(stem)
            except Exception as e:
                failed += 1
                print(f"[ERROR] {base}: {e}")

        self.status_label.config(
            text=f"Batch completed: processed={ok}, skipped={skipped}, failed={failed}, total={total}."
        )
        self.refresh_csv_table()
        if self.current_index >= 0:
            self.load_image_by_index(self.current_index)


if __name__ == "__main__":
    app = GUI()
    app.root.mainloop()

