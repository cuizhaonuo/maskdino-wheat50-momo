import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/attn_argmax_indices.csv"
save_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/attn_key_frequency.jpg"

df = pd.read_csv(csv_path)
data = df.iloc[:, 1:].values.flatten()

keys, counts = np.unique(data, return_counts=True)
order = np.argsort(keys)
keys = keys[order]
counts = counts[order]

plt.figure(figsize=(12, 4))
plt.bar(keys, counts, alpha=0.6)
plt.plot(keys, counts)
plt.xlabel("Key / Query ID")
plt.ylabel("Selected Times (Argmax Count)")
plt.title("Attention Sink Detection: Key Index Selection Frequency")
plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.close()

print(f"✅ Frequency figure saved to: {save_path}")
