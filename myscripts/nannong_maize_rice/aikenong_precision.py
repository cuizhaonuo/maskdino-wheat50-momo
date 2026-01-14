import json
import pandas as pd
from pycocotools.coco import COCO
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

swinl_maize_rice_search2_root = Path('/public/home/zncui/pycharm_project/MaskDINO/myscripts/swinl_maize_rice_val100_test500_search')
csv_path = swinl_maize_rice_search2_root / 'test_pred_vs_gt.csv'
df = pd.read_csv(csv_path)
df['error'] = (df['pred_counts'] - df['ins_counts']).abs() / df['ins_counts']
df = df.loc[df['ins_counts'] > 0]

yolo_val100_test500_root = Path('/public/home/zncui/datasets/maize_rice_dataset/yolo_val100_test500')
yolo_labels_test_path = yolo_val100_test500_root / 'labels' / 'test' # 这里的 test 包含了所有图片的标签
data = []
for p in yolo_labels_test_path.iterdir():
    if p.suffix == '.txt':
        with open(str(p), 'r') as f:
            lines = f.readlines()
        cats = []
        for line in lines:
            parts = line.strip().split()
            cat_id = int(parts[0])
            cats.append(cat_id)
        unique, counts = np.unique(cats, return_counts=True)
        pattern = unique[np.argmax(counts)]
        data.append({
            'filename': p.stem + '.jpg',
            'pattern': 'maize' if pattern == 0 else 'rice'
        })
df_map = pd.DataFrame(data)
df = pd.merge(df, df_map, on='filename', how='left')

# for row in df.iterrows():
#     filename = row['filename']
#     if filename.startwith('DJI'):
#         row['pattern'] = 'rice'
#     else:
#         row['pattern'] = 'maize'


pattern_groups = df.groupby('pattern')
for pattern, group in pattern_groups:
    print(f'Pattern: {pattern}')
    print(f'Mean Absolute Error: {group["error"].mean():.4f}')
    print(group[group["error"] == group["error"].max()])

# 绘制直方图
fig, axes = plt.subplots(1, len(pattern_groups), figsize=(5 * len(pattern_groups), 4), sharey=True)
for group_id, (pattern, group) in enumerate(pattern_groups):
    axes[group_id].hist(group['error'], bins=30, color='skyblue', edgecolor='black')
    axes[group_id].set_title(f'Pattern: {pattern}\nMean AE: {group["error"].mean():.4f}')
    axes[group_id].set_xlabel('Absolute Error')
    axes[group_id].set_ylabel('Frequency')

import os
# 将每个 group 各自保存一个 csv 文件
output_dir = '/public/home/zncui/datasets/nannong-maize-rice/maskdino_data/results/ours_pattern_errors'
os.makedirs(output_dir, exist_ok=True)
for pattern, group in pattern_groups:
    output_path = os.path.join(output_dir, f'{pattern}_errors.csv')
    group.to_csv(output_path, index=False)
# 将 df_merged 保存为一个新的 csv 文件
merged_output_path = '/public/home/zncui/datasets/nannong-maize-rice/maskdino_data/results/ours_maize_pattern_merged_errors.csv'
df.to_csv(merged_output_path, index=False)

# 绘制散点图到一张图中
# fig, ax = plt.subplots(figsize=(8, 6))
# colors = {'maize': 'blue', 'rice': 'green'}
# for pattern, group in pattern_groups:
#     ax.scatter(group['ins_counts'], group['pred_counts'], label=f'Pattern: {pattern}', alpha=0.6, color=colors.get(pattern, 'gray'))
# min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
# max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
# ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x Line')
# ax.set_xlabel('GT')
# ax.set_ylabel('Pred')
# ax.legend()


plt.tight_layout()
plt.show()
