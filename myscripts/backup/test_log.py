import sys
import time

for i in range(5):
    print(f"标准输出 stdout: {i}")  # 正常输出
    print(f"标准错误 stderr: {i}", file=sys.stderr)  # 错误输出
    time.sleep(1)
