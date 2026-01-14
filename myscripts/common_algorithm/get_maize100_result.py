import os
import pandas as pd
import re
import glob
from matplotlib import pyplot as plt
from tqdm import tqdm

maize100_root = '/public/home/zncui/datasets/maize_rice_dataset/maize100'
swinl_maize_rice_val100_test500_search_root = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/swinl_maize_rice_val100_test500_search'
results_csv_path = os.path.join(swinl_maize_rice_val100_test500_search_root, 'test_pred_vs_gt.csv')


df = pd.read_csv(results_csv_path)
df['error'] = (df['pred_counts'] - df['ins_counts']).abs() / df['ins_counts']
df = df.loc[df['ins_counts'] > 0]

data = []
for p in os.listdir(maize100_root):
    if p.endswith('.jpg'):
        data.append({
            'filename': p
        })
df_map = pd.DataFrame(data)

df_merged = pd.merge(df_map, df, on='filename', how='left')
output_path = os.path.join(maize100_root, 'maize100_results.csv')
df_merged.to_csv(output_path, index=False)

mask = df_merged['filename'].str.startswith('DJI')
df_a = df_merged[mask]
df_b = df_merged[~mask]

rows = 3
cols = 2
groups = [('maize100_a', df_a), ('maize100_b', df_b), ('maize100_all', df_merged)]
group_color_mapping = {
    'maize100_a': 'skyblue',
    'maize100_b': 'lightgreen',
    'maize100_all': 'salmon'
}
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
for idx, (pattern, group) in enumerate(groups):
    color = group_color_mapping.get(pattern)
    axes[idx][0].hist(group['error'], bins=30, color=color, edgecolor='black')
    axes[idx][0].set_title(f'{pattern}\n Mean AE: {group["error"].mean():.4f}')
    axes[idx][0].set_xlabel('Absolute Error')
    axes[idx][0].set_ylabel('Frequency')

    axes[idx][1].scatter(group['ins_counts'], group['pred_counts'], color=color, alpha=0.6)
    axes[idx][1].plot([0, group['ins_counts'].max()], [0, group['ins_counts'].max()], 'r--')
    axes[idx][1].set_title(f'{pattern} Pred vs GT')
    axes[idx][1].set_xlabel('Ground Truth Counts')
    axes[idx][1].set_ylabel('Predicted Counts')

plt.tight_layout()
# plt.show()

# ========
data = []
paths = ['/public/home/mryan/lab_share/nfs_field/raw_data/DJI/AQ600/zhaoxian2024/summer_CAAS',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/FC6360/tongliao2024/maize',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/M3M/hebei2025/2025hebei-maize',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/P1/xindu2024',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/P1/xindu2025/maize',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/P1/zhaoxian2024/maize',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/P1/zhaoxian2025/maize',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/M3M/wuhan2025/2025wuhan-rice',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/M3M/pujiang2024',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/M3M/yaan2024',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/P1/zibo2024/rice',
         '/public/home/mryan/lab_share/nfs_field/raw_data/DJI/P1/xindu2025/rice'
         ]

for path in paths:
    m = re.search(r"(DJI/.+)$", path)
    src_path = m.group(1)
    file_names = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.jpg'):
                file_names.append(f)
    print('len(file_names):', len(file_names))
    if len(file_names) > 1:
        for p in file_names:
            # if 'DJI_20240819145524_0100_hd5' in p:
            #     print(p)
            # if 'DJI_3790' in p:
            #     print(p)
            data.append({
                'src_path': src_path,
                'filename': p
            })
print('len(data):', len(data))

# for idx, row in df_merged.iterrows():
#     filename = row['filename']
#     m = re.search(r'(-\d+)\.', filename)
#     info = m.group(1)
#     df_merged.loc[idx, 'filename'] = filename.replace(info, '', 1)

df_merged['filename_clean'] = None  # 先创建新列
for idx, row in df_merged.iterrows():
    filename = row['filename']
    clean_name = re.sub(r'^\d+\.\d+-|-\d+(?=\.)', '', filename)
    df_merged.loc[idx, 'filename_clean'] = clean_name

maize_rice_total_df = pd.DataFrame(data)
maize100_df = df_merged
count = 0
matched_data = []
for i, src_row in tqdm(maize_rice_total_df.iterrows(), total=len(maize_rice_total_df)):
    src_filename = src_row['filename']
    src_path = src_row['src_path']
    for j, tgt_row in maize100_df.iterrows():
        tgt_filename = tgt_row['filename']
        tgt_filename_clean = tgt_row['filename_clean'].replace('.jpg', '')
        # parts = tgt_filename.split('-')
        # tgt_filename = ''.join(parts[1:-1])
        if tgt_filename_clean in src_filename:
            print('Matched:', tgt_filename_clean, tgt_filename, src_filename, src_path)
            matched_data.append({
                'tgt_filename_clean': tgt_filename_clean,
                'tgt_filename': tgt_filename,
                'src_filename': src_filename,
                'src_path': src_path
            })
            count += 1
print('Total matched count:', count)

matched_df = pd.DataFrame(matched_data)
matched_df.to_csv(os.path.join(maize100_root, 'maize100_matched_src_paths.csv'), index=False)

# tgt_df = pd.merge(maize100_df, maize_rice_total_df, on='filename', how='left')
# print(1)
# output_path = os.path.join(maize100_root, 'maize100_results_with_srcpath.csv')
# tgt_df.to_csv(output_path, index=False)

# if __name__ == "__main__":
