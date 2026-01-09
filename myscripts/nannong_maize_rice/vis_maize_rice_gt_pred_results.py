import pandas as pd
from matplotlib import pyplot as plt

nannong_rice_results_path = '/public/home/zncui/datasets/nannong-maize-rice/results/nannong_rice_results.csv'
maskdino_rice_results_path = '/public/home/zncui/datasets/nannong-maize-rice/results/maskdino_rice_results.csv'

maskdino_maize_results_path = '/public/home/zncui/pycharm_project/MaskDINO/myscripts/nannong_maize_0.3s_0.6iou_infer_test2/test_pred_vs_gt.csv'

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

nannong_rice_df = pd.read_csv(nannong_rice_results_path)
nannong_rice_gt_counts = nannong_rice_df['ins_counts']
nannong_rice_pred_counts = nannong_rice_df['pred_counts']
axes[0].scatter(nannong_rice_gt_counts, nannong_rice_pred_counts, color='blue', alpha=0.6)
nannong_min = min(nannong_rice_gt_counts.min(), nannong_rice_pred_counts.min())
nannong_max = max(nannong_rice_gt_counts.max(), nannong_rice_pred_counts.max())
axes[0].plot([nannong_min, nannong_max], [nannong_min, nannong_max], color='red', linestyle='--', linewidth=1)
axes[0].set_xlabel('Ground Truth Counts')
axes[0].set_ylabel('Predicted Counts')

maskdino_rice_df = pd.read_csv(maskdino_rice_results_path)
maskdino_rice_gt_counts = maskdino_rice_df['ins_counts']
maskdino_rice_pred_counts = maskdino_rice_df['pred_counts']
axes[1].scatter(maskdino_rice_gt_counts, maskdino_rice_pred_counts, color='green', alpha=0.6)
maskdino_min = min(maskdino_rice_gt_counts.min(), maskdino_rice_pred_counts.min())
maskdino_max = max(maskdino_rice_gt_counts.max(), maskdino_rice_pred_counts.max())
axes[1].plot([maskdino_min, maskdino_max], [maskdino_min, maskdino_max], color='red', linestyle='--', linewidth=1)
axes[1].set_xlabel('Ground Truth Counts')
axes[1].set_ylabel('Predicted Counts')

maskdino_maize_df = pd.read_csv(maskdino_maize_results_path)
maskdino_maize_gt_counts = maskdino_maize_df['ins_counts']
maskdino_maize_pred_counts = maskdino_maize_df['pred_counts']
axes[2].scatter(maskdino_maize_gt_counts, maskdino_maize_pred_counts, color='orange', alpha=0.6)
maskdino_maize_min = min(maskdino_maize_gt_counts.min(), maskdino_maize_pred_counts.min())
maskdino_maize_max = max(maskdino_maize_gt_counts.max(), maskdino_maize_pred_counts.max())
axes[2].plot([maskdino_maize_min, maskdino_maize_max], [maskdino_maize_min, maskdino_maize_max], color='red', linestyle='--', linewidth=1)
axes[2].set_xlabel('Ground Truth Counts')
axes[2].set_ylabel('Predicted Counts')

output_path = '/public/home/zncui/datasets/nannong-maize-rice/results/maize_rice_gt_pred_comparison.png'
plt.savefig(output_path)
plt.close()