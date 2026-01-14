import os
from matplotlib import pyplot as plt
import pandas as pd

maize100_matched_src_paths_csv = '/public/home/zncui/datasets/maize_rice_dataset/maize100/maize100_matched_src_paths.csv'
df = pd.read_csv(maize100_matched_src_paths_csv)
matched_data = []
for pattern, group in df.groupby('tgt_filename'):
    path = set()
    for idx, row in group.iterrows():
        src_path = row['src_path']
        if '2025' in src_path:
            continue
        path.add(src_path)
    print('pattern: ', pattern)
    print('src_path: ', list(path)[0])
    matched_data.append({
        'filename': pattern,
        'src_path': list(path)[0]
    })
matched_df = pd.DataFrame(matched_data)

maize100_results_csv = '/public/home/zncui/datasets/maize_rice_dataset/maize100/maize100_results.csv'
maize100_df = pd.DataFrame(matched_df)

merged_df = pd.merge(maize100_df, pd.read_csv(maize100_results_csv), on='filename', how='left')
merged_df.to_csv('/public/home/zncui/datasets/maize_rice_dataset/maize100/maize100_results_with_srcpath.csv', index=False)

src_path_grouped_df = merged_df.groupby('src_path')
rows = 2
cols = len(src_path_grouped_df)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
for (pattern, group), ax in zip(src_path_grouped_df, axes):
    place = pattern.split('/')[-2]
    ax[0].hist(group['error'], bins=30, color='skyblue', edgecolor='black')
    ax[0].set_title(f'{place}\n Mean AE: {group["error"].mean():.4f}\n counts: {len(group)}')
    ax[0].set_xlabel('Absolute Error')
    ax[0].set_ylabel('Frequency')
    ax[1].scatter(group['ins_counts'], group['pred_counts'], color='blue', alpha=0.6)
    ax[1].plot([0, group['ins_counts'].max()], [0, group['ins_counts'].max()], 'r--')
    ax[1].set_title(f'{place} Pred vs GT')
    ax[1].set_xlabel('Ground Truth Counts')
    ax[1].set_ylabel('Predicted Counts')
plt.tight_layout()
plt.show()


