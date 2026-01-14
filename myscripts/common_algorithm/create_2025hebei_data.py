import os
import shutil
import pandas as pd
from pathlib import Path

# hebei2025_maize_root = '/public/lab_share/nfs_field/raw_data/DJI/M3M/hebei2025/2025hebei-maize'
# yolo_hebei2025_maize_test_root = '/public/home/zncui/datasets/maize_rice_dataset/yolo_hebei2025_maize/images/test'
# hebei2025_maize_csv_path = '/public/home/zncui/datasets/maize_rice_dataset/hebei2025_maize_dir_counts.csv'

hebei2025_maize_root = '/public/lab_share/nfs_field/raw_data/DJI/M3M/wuhan2025/2025wuhan-rice'
yolo_hebei2025_maize_test_root = '/public/home/zncui/datasets/maize_rice_dataset/yolo_hebei2025_rice/images/test'
os.makedirs(yolo_hebei2025_maize_test_root, exist_ok=True)
hebei2025_maize_csv_path = '/public/home/zncui/datasets/maize_rice_dataset/hebei2025_rice_dir_counts.csv'

data = []
for root, dirs, files in os.walk(hebei2025_maize_root):
    for file in files:
        if file.endswith('.JPG'):
            parts = root.split(os.sep)
            data.append({
                'parent1': parts[-2],
                'parent2': parts[-1],
                'filename': file
            })
df = pd.DataFrame(data)

# 统计每个目录的数量
df.groupby(['parent1', 'parent2']).size().reset_index(name='counts') \
  .to_csv(hebei2025_maize_csv_path, index=False)

df1 = df.groupby(['parent1', 'parent2'])
for (parent1, parent2), group in df1:
    # 从 group 中随机抽取 10 张图片
    sampled = group.sample(15, random_state=42)
    for idx, row in sampled.iterrows():
        src_path = os.path.join(hebei2025_maize_root, row['parent1'], row['parent2'], row['filename'])
        tgt_filename = f'{parent1}-{parent2}-{row["filename"]}'
        tgt_path = os.path.join(yolo_hebei2025_maize_test_root, tgt_filename)
        shutil.copyfile(src_path, tgt_path)







