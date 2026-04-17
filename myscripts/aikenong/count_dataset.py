import os
from pathlib import Path
from typing import Iterable, Optional

try:
    from openpyxl import load_workbook
except ImportError:  # pragma: no cover
    load_workbook = None

# 要统计的根目录
ROOT_DIR = Path("/public/home/qwang/CommonAlgorithm/aikenong_dataset")

# 认为是图片的扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
ERROR_INFO_FILE = "error_info.xlsx"
ERROR_INFO_ROOT = Path("/public/home/zncui/pycharm_project/MaskDINO")
ERROR_PATH_COLUMN = "文件夹路径"
ERROR_PATH_PREFIX = "/media/ubuntu/农场数据采集硬盘1"


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def normalize_error_path(raw_path: str) -> Optional[Path]:
    path_text = raw_path.strip().replace("\\", "/")
    if not path_text:
        return None

    if path_text == ERROR_PATH_PREFIX:
        return None
    if path_text.startswith(ERROR_PATH_PREFIX + "/"):
        path_text = path_text[len(ERROR_PATH_PREFIX) + 1 :]

    return Path(path_text.lstrip("/"))


def load_skip_dirs_from_excel(root: Path) -> set[Path]:
    excel_path = ERROR_INFO_ROOT / ERROR_INFO_FILE
    if not excel_path.exists():
        print(f"未找到 {excel_path}，不启用错误目录过滤。")
        return set()

    if load_workbook is None:
        print("未安装 openpyxl，无法读取 error_info.xlsx。请先安装: pip install openpyxl")
        return set()

    try:
        wb = load_workbook(excel_path, read_only=True, data_only=True)
    except Exception as exc:
        print(f"读取 {excel_path} 失败: {exc}")
        return set()

    ws = wb.active
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), None)
    if not header:
        print(f"{ERROR_INFO_FILE} 为空，不启用错误目录过滤。")
        wb.close()
        return set()

    title_to_idx = {
        str(col).strip(): idx for idx, col in enumerate(header) if col is not None and str(col).strip()
    }
    if ERROR_PATH_COLUMN not in title_to_idx:
        print(f"{ERROR_INFO_FILE} 缺少列: {ERROR_PATH_COLUMN}，不启用错误目录过滤。")
        wb.close()
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

        candidate = (root / rel_path).resolve(strict=False)
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            ignored_rows += 1
            continue

        if candidate.exists() and candidate.is_dir():
            skip_dirs.add(candidate)

    wb.close()
    print(f"已加载错误目录 {len(skip_dirs)} 个（忽略无效行 {ignored_rows} 行）。")
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


def iter_images(subdir: Path, root: Path, skip_dirs: set[Path]) -> Iterable[Path]:
    for current_dir, dirs, files in os.walk(subdir, topdown=True):
        current_path = Path(current_dir).resolve(strict=False)

        # 剪枝：跳过 error_info.xlsx 标记的目录及其子树
        if is_under_skipped_dir(current_path, root, skip_dirs):
            dirs[:] = []
            continue

        dirs[:] = [
            d
            for d in dirs
            if not is_under_skipped_dir(current_path / d, root, skip_dirs)
        ]

        for file_name in files:
            file_path = current_path / file_name
            if not is_image_file(file_path):
                continue
            yield file_path


def count_images_in_subdirs(root: Path):
    if not root.exists():
        print(f"目录不存在: {root}")
        return

    if not root.is_dir():
        print(f"路径不是目录: {root}")
        return

    print(f"统计目录: {root}\n")
    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if not subdirs:
        print("该目录下没有子目录。")
        return

    skip_dirs = load_skip_dirs_from_excel(root)

    total_images = 0

    for subdir in subdirs:
        image_count = 0

        for _ in iter_images(subdir, root, skip_dirs):
            image_count += 1

        total_images += image_count

        print(f"{subdir.name}: {image_count} 张图片")

    print("\n---------------------------")
    print(f"所有子目录图片总数: {total_images} 张")


if __name__ == "__main__":
    count_images_in_subdirs(ROOT_DIR)