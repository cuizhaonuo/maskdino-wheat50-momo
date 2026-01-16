
import os
from PIL import Image

root_dir = "/public/home/zncui/datasets/nannong-maize-rice/whole"

jpg_count_per_dir = {}
unreadable_images = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    jpg_files = [f for f in filenames if f.lower().endswith(".jpg")]

    if jpg_files:
        jpg_count_per_dir[dirpath] = len(jpg_files)

    for jpg in jpg_files:
        img_path = os.path.join(dirpath, jpg)
        try:
            with Image.open(img_path) as img:
                img.verify()  # 只检查是否损坏，不真正加载像素
        except Exception as e:
            unreadable_images.append((img_path, str(e)))

# ===== 输出结果 =====
print("📁 每个目录下的 JPG 文件数量：")
for d, c in jpg_count_per_dir.items():
    print(f"{d}: {c}")

print("\n❌ 无法读取的 JPG 文件：")
if unreadable_images:
    for path, err in unreadable_images:
        print(f"{path} -> {err}")
else:
    print("所有 JPG 文件均可正常读取 ✅")
