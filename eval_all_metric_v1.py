import json
import re
import numpy as np
from collections import Counter

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    sentence_bleu = None

# ================= 工具函数：提取与清洗 =================

def clean_text(text):
    """暴力清洗：去掉所有标点符号，转换为纯小写，防止因为标点算错"""
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split()).lower()

def extract_bboxes(text):
    """提取文本中所有的坐标框 [x1, y1, x2, y2]"""
    pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    matches = re.findall(pattern, text)
    return [list(map(float, m)) for m in matches]

def extract_predictions(pred_text):
    """
    针对固定格式的精准提取：
    格式："考虑鉴别诊断包括 A; B; C。结合特征，最终诊断为：D。"
    """
    top5_list = []
    final_diag = ""

    # 1. 提取最终诊断 (Top-1)
    final_match = re.search(r'最终诊断为：(.*?)(?:。|$)', pred_text)
    if final_match:
        final_diag = final_match.group(1).strip()

    # 2. 提取鉴别诊断列表 (Top-5)
    diff_match = re.search(r'考虑鉴别诊断包括\s*(.*?)(?:。结合特征|$)', pred_text)
    if diff_match:
        diff_str = diff_match.group(1)
        # 支持中英文分号切分
        raw_items = re.split(r'[;；]', diff_str)
        for item in raw_items:
            clean_item = item.strip()
            # 过滤空字符串，并去重
            if clean_item and clean_item not in top5_list:
                top5_list.append(clean_item)

    top5_list = top5_list[:5]

    # 兜底：如果只提取到了列表但没提取到最终诊断，默认取列表第一个
    if not final_diag and top5_list:
        final_diag = top5_list[0]

    return top5_list, final_diag

def extract_caption(text):
    """提取图像描述内容"""
    if "【图像描述】：" in text and "【诊断推理】：" in text:
        return text.split("【图像描述】：")[1].split("【诊断推理】：")[0].strip()
    match = re.search(r'2\. 图像描述：\n(.*?)\n3\.', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text

# ================= 指标计算函数 =================

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def text_f1_score(pred_text, label_text):
    """计算文本的 Precision, Recall, F1"""
    pred_tokens = clean_text(pred_text).split()
    label_tokens = clean_text(label_text).split()
    
    if not pred_tokens or not label_tokens:
        return 0.0, 0.0, 0.0
        
    common = Counter(pred_tokens) & Counter(label_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(label_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

# ================= 主评估流程 =================

def evaluate(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    metrics = {
        'iou_30': [], 'iou_50': [], 'iou_50_95': [],
        'bleu': [], 'precision': [], 'recall': [], 'f1': [],
        'top1_hits': 0, 'top5_hits': 0, 'valid_boxes': 0, 'valid_diag': 0
    }
    smooth = SmoothingFunction().method1 if sentence_bleu else None

    for item in data:
        pred = item.get('response', item.get('predict', ''))
        label = item['labels']

        # --- 1. 定位指标 (mAP) ---
        p_boxes = extract_bboxes(pred)
        l_boxes = extract_bboxes(label)
        
        if l_boxes:
            metrics['valid_boxes'] += 1
            if p_boxes:
                max_iou = max([calculate_iou(p_boxes[0], lb) for lb in l_boxes])
                metrics['iou_30'].append(1 if max_iou >= 0.3 else 0)
                metrics['iou_50'].append(1 if max_iou >= 0.5 else 0)
                thresholds = np.arange(0.5, 1.0, 0.05)
                metrics['iou_50_95'].append(np.mean([max_iou >= t for t in thresholds]))
            else:
                metrics['iou_30'].append(0); metrics['iou_50'].append(0); metrics['iou_50_95'].append(0)

        # --- 2. 描述指标 ---
        p_cap = extract_caption(pred)
        l_cap = extract_caption(label)
        
        if l_cap and sentence_bleu:
            b_score = sentence_bleu([l_cap.split()], p_cap.split(), smoothing_function=smooth)
            metrics['bleu'].append(b_score)
            p, r, f1 = text_f1_score(p_cap, l_cap)
            metrics['precision'].append(p); metrics['recall'].append(r); metrics['f1'].append(f1)

        # --- 3. 诊断指标计算 ---
        # 【修改点1】：使用正确定义的提取函数
        p_list, p_final = extract_predictions(pred)
        l_list, l_final = extract_predictions(label)
        
        # 【修改点2】：标签兜底逻辑。如果真实标签没有使用固定格式（例如只是单纯的病名），则直接取标签原文
        if not l_final:
            l_final = label.replace("【诊断推理】：", "").strip()

        # 统一清洗：去掉标点转小写
        p_final_clean = clean_text(p_final)
        l_final_clean = clean_text(l_final)
        p_list_clean = [clean_text(i) for i in p_list]

        if l_final_clean:
            metrics['valid_diag'] += 1 # 记录有效的诊断样本数
            is_top1 = False
            
            # --- Top-1 命中判定 ---
            if p_final_clean and (p_final_clean in l_final_clean or l_final_clean in p_final_clean):
                is_top1 = True
            
            if is_top1:
                metrics['top1_hits'] += 1
                metrics['top5_hits'] += 1
            else:
                # --- Top-5 命中判定 ---
                for p_item in p_list_clean:
                    if p_item and (p_item in l_final_clean or l_final_clean in p_item):
                        metrics['top5_hits'] += 1
                        break

    # --- 打印最终报表 ---
    total_diag = metrics['valid_diag'] if metrics['valid_diag'] > 0 else 1
    total = len(data)

    print(f"\n ====== {file_path} 评估报表 ====== 📊")
    print(f"总样本数: {total} | 有效诊断标签数: {metrics['valid_diag']}")
    
    print("\n📍 任务 1：异常定位 (Anomaly Localization)")
    print(f"  - mAP@30:      {np.mean(metrics['iou_30']) if metrics['iou_30'] else 0:.4f}")
    print(f"  - mAP@50:      {np.mean(metrics['iou_50']) if metrics['iou_50'] else 0:.4f}")
    print(f"  - mAP@[50:95]: {np.mean(metrics['iou_50_95']) if metrics['iou_50_95'] else 0:.4f}  <-- COCO标准")

    print("\n📝 任务 2：图像描述 (Image Captioning)")
    print(f"  - Precision:   {np.mean(metrics['precision']) if metrics['precision'] else 0:.4f}")
    print(f"  - Recall:      {np.mean(metrics['recall']) if metrics['recall'] else 0:.4f}")
    print(f"  - F1 Score:    {np.mean(metrics['f1']) if metrics['f1'] else 0:.4f}")
    if sentence_bleu:
        print(f"  - BLEU:        {np.mean(metrics['bleu']) if metrics['bleu'] else 0:.4f}")

    print("\n🩺 任务 3：诊断推理 (Diagnostic Reasoning)")
    print(f"  - Top-1 Acc:   {metrics['top1_hits']/total_diag:.4f}  ({metrics['top1_hits']}/{total_diag})")
    print(f"  - Top-5 Acc:   {metrics['top5_hits']/total_diag:.4f}  ({metrics['top5_hits']}/{total_diag})")
    print("========================================================\n")

if __name__ == "__main__":
    evaluate('/Users/yfx/vscode/project/final_proj/result/Qwen2-VL-7B/infer_result/answer.jsonl')