#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import yaml
from ultralytics import YOLO


def safe_get_float(d, key, default):
    """
    从字典中安全获取 float:
    - 若不存在 / 为 None / 转换失败，则返回 default
    """
    v = d.get(key, default)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def safe_get_imgsz(args, default=640):
    """
    从 args 中获取 imgsz:
    - 如果是单个数，直接用
    - 如果是列表/元组，取第一个
    - 失败则用默认值
    """
    v = args.get("imgsz", default)
    if isinstance(v, (list, tuple)) and len(v) > 0:
        v = v[0]
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = default
    return v


def load_model(cfg_path, weights_path):
    """
    加载 ultralytics YOLO 模型：
    1) 优先使用 args.yaml 中的 model 配置初始化，再加载权重；
    2) 如果没有可用的 model 配置，则直接用权重初始化。
    """
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")

    with open(cfg_path, "r") as f:
        args = yaml.safe_load(f) or {}

    model_cfg = args.get("model", None)

    # 情况一：有 model 配置（yolov8n.yaml、自定义.yaml 等）
    if model_cfg is not None and str(model_cfg).strip() != "":
        model_cfg_path = str(model_cfg)
        if not os.path.isabs(model_cfg_path):
            model_cfg_path = os.path.join(os.path.dirname(cfg_path), model_cfg_path)

        if os.path.isfile(model_cfg_path):
            print(f"[INFO] 使用模型配置初始化: {model_cfg_path}")
            model = YOLO(model_cfg_path)
            print(f"[INFO] 加载权重: {weights_path}")
            model.load(weights_path)
        else:
            print(f"[WARN] 未找到模型配置文件 {model_cfg_path} ，改为直接用权重初始化。")
            model = YOLO(weights_path)
    else:
        # 情况二：args.yaml 中没有有效的 model 字段
        print("[INFO] args.yaml 中未指定有效的 model 字段，直接用权重初始化模型。")
        model = YOLO(weights_path)

    return model, args


def run_inference(model, args, source_dir, project_root, save_name, label_dir_name):
    """
    对 source_dir 中所有图片做推理：
    - 可视化结果保存到: project_root/save_name
    - YOLO txt 结果保存到: project_root/label_dir_name
      文件名与图片同名，格式:
      class x_center y_center width height  (归一化坐标)
    - 仅保留 conf >= 0.7 的预测框
    """
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {source_dir}")

    project_root = Path(project_root)
    project_root.mkdir(parents=True, exist_ok=True)

    label_dir = project_root / label_dir_name
    label_dir.mkdir(parents=True, exist_ok=True)

    # 推理参数
    imgsz = safe_get_imgsz(args, default=640)
    conf = 0.7  # 固定置信度阈值，过滤掉 conf < 0.7
    iou = safe_get_float(args, "iou", 0.45)
    if not (0.0 <= iou <= 1.0):
        print(f"[WARN] args.yaml 中 iou={iou} 非法，重置为 0.45")
        iou = 0.45

    print(f"[INFO] 开始推理目录: {source_dir}")
    print(f"[INFO] 可视化输出目录: {project_root / save_name}")
    print(f"[INFO] YOLO标签输出目录: {label_dir}")
    print(f"[INFO] imgsz={imgsz}, conf={conf}, iou={iou}")

    # 使用 stream=True 一边推理一边写入标签文件
    results = model.predict(
        source=str(source_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,            # 保存可视化图片到 project/name
        save_conf=True,       # 在框上显示置信度（仅影响可视化）
        project=str(project_root),
        name=save_name,       # 最终可视化输出目录: project_root/save_name
        exist_ok=True,
        verbose=True,
        stream=True           # 逐张返回结果，方便自定义保存 txt
    )

    for r in results:
        img_path = Path(r.path)
        stem = img_path.stem
        txt_path = label_dir / f"{stem}.txt"

        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes
            # 归一化坐标 (x_center, y_center, w, h)
            xywhn = boxes.xywhn.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            # 这里不再按 conf 过滤，因为在 model.predict 已经用 conf=0.7 过滤过
            for c, (x, y, w, h) in zip(clss, xywhn):
                lines.append(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # 即使没有框也创建空文件，方便后续统计
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

    print(f"[INFO] 推理完成: {save_name}, 标签已保存到: {label_dir_name}")


def main():
    # 路径按你给定
    args_yaml_path = "/public/home/zncui/pycharm_project/ultralytics/ultralytics/runs/detect/train/args.yaml"
    weights_path = "/public/home/zncui/pycharm_project/ultralytics/ultralytics/runs/detect/train/weights/best.pt"

    dataset_root = "/public/home/zncui/datasets/maize_rice_organ_seg_dataset"
    maize_dir = os.path.join(dataset_root, "maize")
    rice_dir = os.path.join(dataset_root, "rice")

    # 可视化输出目录名（在 dataset_root 下）
    maize_save_name = "maize_yolo_infer"
    rice_save_name = "rice_yolo_infer"

    # 标签输出目录名（在 dataset_root 下）
    maize_label_dir_name = "maize_yolo_labels"
    rice_label_dir_name = "rice_yolo_labels"

    # 1. 加载模型（配置 + 权重）
    model, args = load_model(args_yaml_path, weights_path)

    # 2. 推理 maize
    run_inference(
        model=model,
        args=args,
        source_dir=maize_dir,
        project_root=dataset_root,
        save_name=maize_save_name,
        label_dir_name=maize_label_dir_name
    )

    # 3. 推理 rice
    run_inference(
        model=model,
        args=args,
        source_dir=rice_dir,
        project_root=dataset_root,
        save_name=rice_save_name,
        label_dir_name=rice_label_dir_name
    )


if __name__ == "__main__":
    main()
