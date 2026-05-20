import json
import re

# 👇 替换为你需要处理的文件路径
input_file = '/mnt/workspace/final_proj/output/nova_sft_merged/grpo-2/checkpoint-362/infer_result/20260311-223128.jsonl'
output_file = '/mnt/workspace/final_proj/output/nova_sft_merged/grpo-2/checkpoint-362/infer_result/result.jsonl'

def extract_and_format_diagnosis(text):
    """提取包含中文套话的诊断推理，并转换为纯净的英文无序列表"""
    # 1. 找到【诊断推理】：后面的所有内容
    match = re.search(r'【诊断推理】：\s*(.*)', text, flags=re.DOTALL)
    
    if match:
        diag_content = match.group(1).strip()
        diseases = []
        
        # 2. 提取“鉴别诊断”部分
        diff_match = re.search(r'(?:鉴别诊断包括|考虑鉴别诊断包括|可能是)\s*(.*?)(?:。|结合特征|最终诊断)', diag_content)
        if diff_match:
            # 按分号切割并清理空格
            diffs = re.split(r'[;；]', diff_match.group(1))
            for d in diffs:
                clean_d = d.strip()
                if clean_d and clean_d not in diseases:
                    diseases.append(clean_d)
                    
        # 3. 提取“最终诊断”部分
        final_match = re.search(r'(?:最终诊断为：|最终诊断是)\s*(.*?)(?:。|$)', diag_content)
        if final_match:
            finals = re.split(r'[;；]', final_match.group(1))
            for f in finals:
                clean_f = f.strip()
                # 如果最终诊断已经在列表里了，就不重复添加；如果不在，追加到最后
                if clean_f and clean_f not in diseases:
                    diseases.append(clean_f)
        
        # 4. 保底机制：如果正则没匹配到中文套话（比如模型突然输出了纯英文）
        if not diseases:
            # 暴力把中文套话替换成分号，然后切割
            clean_str = re.sub(r'考虑鉴别诊断包括|。结合特征，最终诊断为：|。', ';', diag_content)
            raw_items = re.split(r'[;；]', clean_str)
            for item in raw_items:
                clean_item = item.strip()
                if clean_item and clean_item not in diseases:
                    diseases.append(clean_item)

        # 5. 重新拼装为目标格式
        if diseases:
            bullet_points = [f"* {d}" for d in diseases]
            new_diag_text = "3. 诊断推理:\n" + "\n".join(bullet_points) + "\n"
            text = text[:match.start()] + new_diag_text
            
    return text

print("⏳ 正在剔除中文模板，进行深度格式清洗...")

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        data = json.loads(line)
        
        key = 'response' if 'response' in data else 'predict'
        
        if key in data and isinstance(data[key], str):
            data[key] = extract_and_format_diagnosis(data[key])
            
        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"✅ 深度清洗完成！干净的数据已生成: {output_file}")