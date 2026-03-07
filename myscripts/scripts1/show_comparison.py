from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

root = Path('/public/home/zncui/datasets/wheat50/total_images')
select_dir = root / 'select_images'
source_dir = root / 'results'
vis_dir = root / 'vis_results_infered_poly'

save_path = root / "comparison.png"

# 获取 select_dir 中的图片
select_images = sorted(list(select_dir.glob('*.jpg')))

pairs = []
for sel in select_images:
    src = source_dir / sel.name
    vis = vis_dir / sel.name

    if src.exists() and vis.exists():
        pairs.append((src, sel, vis))

n = len(pairs)

fig, axes = plt.subplots(3, n, figsize=(3*n, 9))

for i, (src, sel, vis) in enumerate(pairs):

    img_src = Image.open(src)
    img_sel = Image.open(sel)
    img_vis = Image.open(vis)

    # 第一行 source
    axes[0, i].imshow(img_src)
    # axes[0, i].set_title(src.name, fontsize=8)
    axes[0, i].axis('off')

    # 第二行 select
    axes[1, i].imshow(img_sel)
    axes[1, i].axis('off')

    # 第三行 vis
    axes[2, i].imshow(img_vis)
    axes[2, i].axis('off')

# 行标题
axes[0,0].set_ylabel("results", fontsize=12)
axes[1,0].set_ylabel("select", fontsize=12)
axes[2,0].set_ylabel("vis", fontsize=12)

plt.tight_layout()

# 保存图片
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

print(f"Saved to: {save_path}")