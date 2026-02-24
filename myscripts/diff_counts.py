import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("./baseline-xxx-search-0.3-0.4-1000-debug/test_iou-IOU_s-0.3000_n-0.4000_m-plain_ms-keep_max/test_pred_vs_gt.csv")

# 计算每行的绝对差值
df["abs_diff"] = (df["ins_counts"] - df["pred_counts"]).abs()

# 计算总绝对差值
total_abs_diff = df["abs_diff"].sum()

# 计算 ins_counts 总数
total_ins = df["ins_counts"].sum()

# 计算百分比
percentage = (total_abs_diff / total_ins) * 100

print("总绝对差值:", total_abs_diff)
print("ins_counts 总数:", total_ins)
print(f"百分比: {percentage:.2f}%")
