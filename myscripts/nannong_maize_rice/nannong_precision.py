import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
from matplotlib import pyplot as plt

nannong_data_root = Path('/public/home/zncui/datasets/nannong-maize-rice')
key = [
    '0724','0727', '0730', 'whole/train', 'whole/val'
]

maize_result_root = '/public/home/zncui/datasets/nannong-maize-rice/maskdino_data/results/maskdino_maize_results.csv'
df = pd.read_csv(maize_result_root)
df['error'] = (df['pred_counts'] - df['ins_counts']).abs() / df['ins_counts']
df = df.loc[df['ins_counts'] > 0]

paths = [nannong_data_root / k for k in key]
data = []
for path in paths:
    for p in path.iterdir():
        if p.suffix == '.JPG':
            data.append({
                'filename': p.name,
                'pattern': p.parent.name
            })
df_map = pd.DataFrame(data)
df_merged = pd.merge(df, df_map, on='filename', how='left')
pattern_groups = df_merged.groupby('pattern')

# 将每个 group 各自保存一个 csv 文件
output_dir = '/public/home/zncui/datasets/nannong-maize-rice/maskdino_data/results/pattern_errors'
os.makedirs(output_dir, exist_ok=True)
for pattern, group in pattern_groups:
    output_path = os.path.join(output_dir, f'{pattern}_errors.csv')
    group.to_csv(output_path, index=False)
# 将 df_merged 保存为一个新的 csv 文件
merged_output_path = '/public/home/zncui/datasets/nannong-maize-rice/maskdino_data/results/maize_pattern_merged_errors.csv'
df_merged.to_csv(merged_output_path, index=False)

# fig, axes = plt.subplots(1, len(pattern_groups), figsize=(5 * len(pattern_groups), 4), sharey=True)
# for group_id, (pattern, group) in enumerate(pattern_groups):
#     print(f'Pattern: {pattern}')
#     print(f'Mean Absolute Error: {group["error"].mean():.4f}')
#     print(group[group["error"] == group["error"].max()])
#     axes[group_id].hist(group['error'], bins=30, color='skyblue', edgecolor='black')
#     # 柱状图
#     # axes[group_id].bar(len(group), group["error"])
#     axes[group_id].set_title(f'Pattern: {pattern}\nMean AE: {group["error"].mean():.4f}')
#     axes[group_id].set_xlabel('Absolute Error')
#     axes[group_id].set_ylabel('Frequency')
# plt.tight_layout()
# plt.show()







