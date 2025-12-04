
csv_path = "/public/home/zncui/pycharm_project/MaskDINO/myscripts/query_match_count.csv"

# 读取文件然后计算每一列的和
import pandas as pd
df = pd.read_csv(csv_path, header=None)
col_sums = df.sum()
for i, col_sum in enumerate(col_sums):
    if col_sum > 0:
        print(f"Column {i}: {col_sum}")

# 将 colsums 绘制成柱状图
import matplotlib.pyplot as plt
plt.bar(range(len(col_sums)), col_sums, width=3)
plt.xlabel("Column Index")
plt.ylabel("Sum")
plt.title("Column Sums")
plt.savefig("query_match_count_bar.png")
plt.show()