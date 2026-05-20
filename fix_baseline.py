import json
import re
import os
import glob

# 指定 infer_result 所在的文件夹路径
infer_dir = '/mnt/workspace/final_proj/result/llava-v1.6-vicuna-7b-hf/infer_result'
output_file = '/mnt/workspace/final_proj/result/llava-v1.6-vicuna-7b-hf/base.jsonl'

print("🔍 正在扫描结果文件夹...")

# 自动寻找文件夹下的所有 jsonl 文件
jsonl_files = glob.glob(os.path.join(infer_dir, '*.jsonl'))

if not jsonl_files:
    print(f"❌ 错误：在 {infer_dir} 下没有找到任何 .jsonl 文件！")
    exit(1)

# 如果有多个文件，自动选择最新生成的那一个
input_file = max(jsonl_files, key=os.path.getctime)
print(f"🎯 自动锁定最新结果文件: {input_file}")
print("⏳ 正在修复 Baseline 结果的格式漂移...")

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line)
        
        # 兼容不同的键名
        key = 'response' if 'response' in data else 'predict'
        
        if key in data:
            text = data[key]
            
            # 使用正则表达式，强行替换为标准中文标题
            text = re.sub(r'(?:1\.\s*)?Abnormal location:?\s*', '【异常定位】：', text, flags=re.IGNORECASE)
            text = re.sub(r'(?:2\.\s*)?Image Description:?\s*', '\n【图像描述】：', text, flags=re.IGNORECASE)
            text = re.sub(r'(?:3\.\s*)?Diagnosis(?: Deduction)?:?\s*', '\n【诊断推理】：', text, flags=re.IGNORECASE)
            
            data[key] = text
            
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"✅ 格式修复完成！清洗后的数据已保存至: {output_file}")