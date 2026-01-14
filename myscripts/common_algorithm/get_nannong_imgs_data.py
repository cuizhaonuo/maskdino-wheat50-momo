import os
import pandas as pd

nannong_imgs_root = '/public/home/zncui/datasets/nannong-maize-rice/nannong_imgs/'
split = ['maize', 'rice']
maskdino_results_root = '/public/home/zncui/datasets/nannong-maize-rice/maskdino_data/results/'

for sp in split:
    sp_root = os.path.join(nannong_imgs_root, sp)
    img_list = os.listdir(sp_root)
    img_list = [img_name for img_name in img_list if img_name.lower().endswith('.jpg')]
    data = []
    for img_name in img_list:
        data.append({
            'filename': img_name
        })
    df = pd.DataFrame(data)
    csv_path = os.path.join(maskdino_results_root, f'maskdino_{sp}_results.csv')
    data_df = pd.read_csv(csv_path)
    merged_df = pd.merge(df, data_df, on='filename', how='left')
    out_csv_path = os.path.join(nannong_imgs_root, f'maskdino_{sp}_results.csv')
    merged_df.to_csv(out_csv_path, index=False)
    print(f'Saved merged data for {sp} to {out_csv_path}')