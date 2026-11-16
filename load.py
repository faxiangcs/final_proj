import os

from datasets import load_dataset

# 使用环境变量 HF_TOKEN，不要把 token 写进代码
dataset = load_dataset("c-i-ber/Nova", token=os.environ.get("HF_TOKEN"))
print(dataset)

