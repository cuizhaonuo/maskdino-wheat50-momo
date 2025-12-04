import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np

from typing import List

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from demo.predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image

from data_register import uav_register_datasets

def setup_cfg():
    config_file: str = '/public/home/qwang/DJI/MaskDINO/configs/coco/panoptic-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048_uav.yaml'
    opts: List[str] = [
        'MODEL.WEIGHTS',
        '/public/home/qwang/DJI/MaskDINO/train_result/baseline/output_all_20251119/model_best_PQ.pth'
    ]
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


# ==============================================================================
# 1. 分割模型
# ==============================================================================
class SegModel:
    def __init__(self):
        self.cfg = setup_cfg()
        self.visualizer = VisualizationDemo(self.cfg)

    def count_spike_from_panoptic(self, predictions) -> int:
        """
        从 predictions['panoptic_seg'] 中统计 spike 实例数量（必须有 segments_info）
        """
        if "panoptic_seg" not in predictions:
            return 0

        panoptic_seg, segments_info = predictions["panoptic_seg"]

        spike_class_id = 0  # 这里假设穗的类别 id 为 0

        count = 0
        for seg in segments_info:
            if seg.get("isthing", False) and seg.get("category_id", -1) == spike_class_id:
                count += 1

        return count

    def infer_and_save(self, img_path: str, out_file: str, opacity: float = 0.5):
        img = read_image(img_path, format="BGR")
        predictions, visualized_output = self.visualizer.run_on_image(img)

        # 准备各种路径
        dir_path = os.path.dirname(out_file)
        if dir_path == "":
            dir_path = os.path.dirname(img_path)
        if dir_path == "":
            dir_path = "."

        base_name = os.path.basename(out_file)
        filename_no_ext = os.path.splitext(base_name)[0]

        overlay_path = os.path.join(dir_path, f"{filename_no_ext}_overlay.png")
        mask_path = os.path.join(dir_path, f"{filename_no_ext}_mask.png")  # 若后续想用可保留
        csv_path = os.path.join(dir_path, "results.csv")

        # 这里 overlay 与 mask 先都保存为可视化图（你之后想改 mask 逻辑可以再改）
        visualized_output.save(overlay_path)
        visualized_output.save(mask_path)

        counts = self.count_spike_from_panoptic(predictions)

        # 写入 CSV：注意这里只保存 filename 和 counts
        new_data_df = pd.DataFrame({
            "filename": [filename_no_ext],
            "counts": [counts]
        })

        if not os.path.exists(csv_path):
            # 如果不存在 CSV，则新建
            new_data_df.to_csv(csv_path, index=False)
        else:
            # 追加写入（若文件是空的也要处理）
            try:
                existing_df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                existing_df = pd.DataFrame(columns=["filename", "counts"])

            if "filename" not in existing_df.columns:
                existing_df["filename"] = []
            if "counts" not in existing_df.columns:
                existing_df["counts"] = []

            # 避免重复写入同名文件
            if filename_no_ext not in existing_df["filename"].values:
                new_data_df.to_csv(csv_path, mode="a", header=False, index=False)


# ==============================================================================
# 2. GUI 主程序
# ==============================================================================
class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('禾本科作物可控环境生物量测量工具')
        self.root.geometry("1400x900")  # 稍微加大窗口尺寸以容纳图和表格

        # --- 状态变量 ---
        # Tk Image 对象
        self.tk_img_orig = None
        self.tk_img_overlay = None

        # PIL Image 对象 (用于 resize)
        self.raw_img_orig = None
        self.raw_img_overlay = None

        self.current_path = ""
        self.image_paths = []
        self.current_index = -1
        self.output_dir = ""

        # 模型（MaskDINO）
        self.model = None

        self.setup_ui()

    def setup_ui(self):
        # ===== 1. 顶部按钮区域 =====
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, pady=5, padx=10)

        # 将样式参数(传给Button)和布局参数(传给pack)分开
        btn_style = {'font': ("Arial", 10)}
        pack_opts = {'side': tk.LEFT, 'padx': 5}

        tk.Button(top_frame, text="选择图片", command=self.choose_image, **btn_style).pack(**pack_opts)
        tk.Button(top_frame, text="选择目录", command=self.choose_directory, **btn_style).pack(**pack_opts)

        # 间隔
        tk.Frame(top_frame, width=30).pack(side=tk.LEFT)

        tk.Button(top_frame, text="选择输出目录", command=self.choose_output_dir, **btn_style).pack(**pack_opts)

        # 特殊颜色的按钮
        tk.Button(top_frame, text="推理 (当前)", command=self.run_inference, bg="#e1f5fe", **btn_style).pack(**pack_opts)
        tk.Button(top_frame, text="推理 (全部)", command=self.run_inference_all, bg="#ffebee", **btn_style).pack(**pack_opts)

        # ===== 2. 主体区域 (PanedWindow 分割：上部分是图，下部分是表) =====
        self.main_paned = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashwidth=4)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- 上半部分：图片展示区 (左列表 + 右两图：原图 + overlay) ---
        top_content_frame = tk.Frame(self.main_paned)
        self.main_paned.add(top_content_frame, height=600)  # 初始高度

        # 左侧列表
        list_frame = tk.Frame(top_content_frame, width=200)
        list_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(list_frame, text="文件列表").pack(anchor=tk.W)

        list_scroll = tk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(list_frame, yscrollcommand=list_scroll.set, width=25)
        list_scroll.config(command=self.listbox.yview)

        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        # 右侧两图区域（原图 + overlay）
        images_frame = tk.Frame(top_content_frame)
        images_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=5)

        # 辅助函数：创建图片Label容器
        def create_img_container(parent, title, bg_color):
            frame = tk.Frame(parent, bg="white", bd=1, relief=tk.SUNKEN)
            frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=2)
            tk.Label(frame, text=title, bg=bg_color).pack(fill=tk.X)
            lbl = tk.Label(frame, bg="gray")
            lbl.pack(expand=True, fill=tk.BOTH)
            return lbl

        self.orig_label = create_img_container(images_frame, "原图", "#eeeeee")
        self.overlay_label = create_img_container(images_frame, "推理叠加 (Overlay)", "#e3f2fd")

        # --- 下半部分：CSV 数据展示区 ---
        bottom_content_frame = tk.Frame(self.main_paned)
        self.main_paned.add(bottom_content_frame)

        tk.Label(bottom_content_frame, text="Results.csv 数据预览").pack(anchor=tk.W)

        # Treeview 表格：显示 filename 和 counts
        cols = ("filename", "counts")
        self.tree = ttk.Treeview(bottom_content_frame, columns=cols, show='headings', height=8)
        self.tree.heading("filename", text="文件名")
        self.tree.heading("counts", text="穗数")

        self.tree.column("filename", width=300)
        self.tree.column("counts", width=150)

        tree_scroll = tk.Scrollbar(bottom_content_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ===== 3. 底部状态栏 =====
        status_frame = tk.Frame(self.root)
        status_frame.pack(fill=tk.X, pady=2, padx=10)

        self.path_label = tk.Label(status_frame, text="当前图片：无", fg="blue")
        self.path_label.pack(side=tk.LEFT)

        self.status_label = tk.Label(status_frame, text="", fg="purple")
        self.status_label.pack(side=tk.RIGHT)

    # ================= 逻辑功能 =================

    def ensure_model(self):
        """确保有一个 MaskDINO 分割模型实例"""
        if self.model is None:
            try:
                self.status_label.config(text="正在初始化 MaskDINO 模型...")
                self.root.update()
                self.model = SegModel()
                self.status_label.config(text="MaskDINO 模型初始化完成")
            except Exception as e:
                messagebox.showerror("错误", f"模型初始化失败：{e}")

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.image_paths = [path]
            self.current_index = 0
            self.refresh_listbox()
            self.load_image_by_index(0)

    def choose_directory(self):
        d = filedialog.askdirectory()
        if d:
            files = sorted([os.path.join(d, f) for f in os.listdir(d)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            if not files:
                messagebox.showinfo("提示", "该目录下无图片")
                return
            self.image_paths = files
            self.current_index = 0
            self.refresh_listbox()
            self.load_image_by_index(0)

    def choose_output_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_dir = d
            self.status_label.config(text=f"输出目录: {d}")
            # 选择目录后，尝试加载一下CSV
            self.refresh_csv_table()
            # 如果当前有图，尝试重新加载结果图
            if self.current_index >= 0:
                self.load_image_by_index(self.current_index)

    def show_next(self):
        if self.image_paths:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
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

    # --- 核心：加载图片与结果 ---
    def load_image_by_index(self, idx):
        if idx < 0 or idx >= len(self.image_paths):
            return

        path = self.image_paths[idx]
        self.current_path = path
        self.path_label.config(text=f"当前: {path}")

        # 1. 加载原图
        try:
            self.raw_img_orig = Image.open(path)
            self.update_single_image(self.raw_img_orig, self.orig_label, is_result=False)
        except Exception as e:
            print(f"原图加载失败: {e}")

        # 2. 尝试寻找 Overlay
        self.raw_img_overlay = None

        if self.output_dir:
            filename_no_ext = os.path.basename(path).split('.')[0]
            overlay_path = os.path.join(self.output_dir, f"{filename_no_ext}_overlay.png")

            if os.path.exists(overlay_path):
                try:
                    self.raw_img_overlay = Image.open(overlay_path)
                except Exception:
                    pass

        # 更新 overlay 显示 (如果有图就显示，没图就清空)
        self.update_single_image(self.raw_img_overlay, self.overlay_label, is_result=True)

    def update_single_image(self, pil_img, label_widget, is_result=False):
        """通用更新单个Label图片的函数"""
        if pil_img is None:
            label_widget.config(image="", text="无结果" if is_result else "无法加载")
            label_widget.image = None  # keep ref
            return

        # 自动缩放
        w = label_widget.winfo_width()
        h = label_widget.winfo_height()
        if w < 10 or h < 10:
            # 窗口还没渲染出来时，给个默认大小
            w, h = 300, 300

        img_w, img_h = pil_img.size
        scale = min(w / img_w, h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        if new_w < 1:
            new_w = 1
        if new_h < 1:
            new_h = 1

        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)

        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img  # 必须保留引用防止被垃圾回收

        # 保存到类属性中方便后续使用
        if label_widget == self.orig_label:
            self.tk_img_orig = tk_img
        elif label_widget == self.overlay_label:
            self.tk_img_overlay = tk_img

    # --- CSV 处理 ---
    def refresh_csv_table(self):
        """读取 results.csv 并填充表格"""
        # 清空旧数据
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.output_dir:
            return

        csv_path = os.path.join(self.output_dir, "results.csv")
        if not os.path.exists(csv_path):
            return

        try:
            df = pd.read_csv(csv_path)
            # 倒序显示，让最新的在最上面
            for index, row in df[::-1].iterrows():
                # 这里使用 counts 字段
                self.tree.insert("", tk.END, values=(row['filename'], row['counts']))
        except Exception as e:
            print(f"读取 CSV 失败: {e}")

    # --- 推理 ---
    def run_inference(self):
        if not self.current_path or not self.output_dir:
            messagebox.showwarning("警告", "请先选择图片和输出目录")
            return

        self.ensure_model()
        if not self.model:
            return

        try:
            base = os.path.basename(self.current_path)
            out_file_base = os.path.join(self.output_dir, base)

            self.status_label.config(text="正在推理中...")
            self.root.update()

            self.model.infer_and_save(self.current_path, out_file_base)

            self.status_label.config(text="推理完成")

            # 刷新结果显示
            self.load_image_by_index(self.current_index)
            # 刷新表格
            self.refresh_csv_table()

        except Exception as e:
            messagebox.showerror("错误", str(e))

    def run_inference_all(self):
        if not self.image_paths or not self.output_dir:
            messagebox.showwarning("警告", "请先选择图片列表和输出目录")
            return

        self.ensure_model()
        if not self.model:
            return

        count = 0
        total = len(self.image_paths)

        for i, path in enumerate(self.image_paths):
            base = os.path.basename(path)
            out_file_base = os.path.join(self.output_dir, base)

            self.status_label.config(text=f"正在处理 ({i + 1}/{total}): {base}")
            self.root.update()

            try:
                self.model.infer_and_save(path, out_file_base)
                count += 1
            except Exception as e:
                print(f"Error {base}: {e}")

        self.status_label.config(text=f"批量推理完成，共 {count} 张")
        self.refresh_csv_table()
        # 刷新当前显示
        if self.current_index >= 0:
            self.load_image_by_index(self.current_index)


if __name__ == '__main__':
    app = GUI()
    app.root.mainloop()
