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
    """暴力清洗：去掉所有标点符号，转换为纯小写"""
    if not text: 
        return ""
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split()).lower()

def extract_bboxes(text):
    """提取文本中所有的坐标框 [x1, y1, x2, y2]"""
    pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    matches = re.findall(pattern, text)
    return [list(map(float, m)) for m in matches]

def extract_caption(text):
    """提取图像描述内容"""
    if "【图像描述】：" in text and "【诊断推理】：" in text:
        return text.split("【图像描述】：")[1].split("【诊断推理】：")[0].strip()
    match = re.search(r'2\. 图像描述：\n(.*?)\n3\.', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text 

def extract_diagnoses_universal(text):
    """
    🌟 终极双模提取器：同时兼容“预测(Markdown列表/纯英文)”和“标签(SFT中文套话)”
    """
    # 截取诊断推理区块
    reasoning_part = text
    if "诊断推理：" in text:
        reasoning_part = text.split("诊断推理：")[-1]
    elif "诊断推理:" in text:
        reasoning_part = text.split("诊断推理:")[-1]

    top5_list = []
    final_diag = ""

    # ================= 模式 1：尝试提取 SFT 原始标签（中文套话格式） =================
    final_match = re.search(r'(?:最终的?临床?诊断(?:是|为：?)|最终诊断为：?)\s*([a-zA-Z\s\-\']+)', reasoning_part)
    diff_match = re.search(r'(?:鉴别诊断(?:包括)?|考虑鉴别诊断(?:包括)?|可能是)\s*([a-zA-Z\s\-\'\;；\,]+)(?:。|结合|最终)', reasoning_part)
    
    if final_match or diff_match:
        if final_match:
            final_diag = final_match.group(1).strip()
        if diff_match:
            raw_diffs = re.split(r'[;；,]', diff_match.group(1))
            top5_list = [d.strip() for d in raw_diffs if d.strip()]
        
        if final_diag and not top5_list: top5_list = [final_diag]
        if top5_list and not final_diag: final_diag = top5_list[0]
        
        return top5_list[:5], final_diag

    # ================= 模式 2：尝试提取 预测结果（Markdown 星号格式） =================
    raw_items = re.findall(r'(?:^|\n)\s*\*\s*([^\n]+)', reasoning_part)
    if raw_items:
        for item in raw_items:
            clean_item = item.strip()
            if clean_item and clean_item not in top5_list:
                top5_list.append(clean_item)
        top5_list = top5_list[:5]
        final_diag = top5_list[0] if top5_list else ""
        return top5_list, final_diag

    # ================= 模式 3：兜底机制（纯英文乱码或无格式文本） =================
    eng_match = re.search(r'(?:most likely|diagnosis is|diagnosed as)\s+([a-zA-Z\s\-\']+)(?:\.|$|\n)', reasoning_part, flags=re.IGNORECASE)
    if eng_match:
        final_diag = eng_match.group(1).strip()
        top5_list = [final_diag]
    else:
        # 暴力剔除坐标和括号，强行作为诊断结果
        clean_t = re.sub(r'\[.*?\]', '', reasoning_part)
        clean_t = re.sub(r'【.*?】', '', clean_t)
        final_diag = " ".join(clean_t.split())
        top5_list = [final_diag]
        
    return top5_list[:5], final_diag

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

    print("\n" + "="*70)
    print("🚀 开始诊断推理验证 (双模混合解析：支持 Label 和 Predict 异构对齐)")
    print("="*70 + "\n")

    for idx, item in enumerate(data):
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
        # 用统一提取器同时处理预测和标签
        p_list, p_final = extract_diagnoses_universal(pred)
        l_list, l_final = extract_diagnoses_universal(label)

        # 统一清洗：去掉标点转小写
        p_final_clean = clean_text(p_final)
        l_final_clean = clean_text(l_final)
        p_list_clean = [clean_text(i) for i in p_list]

        if l_final_clean:
            metrics['valid_diag'] += 1 
            is_top1 = False
            is_top5 = False
            
            # --- 命中判定核心逻辑 ---
            # 判断 Top-1 (互相包含即算命中)
            if p_final_clean and (p_final_clean in l_final_clean or l_final_clean in p_final_clean):
                is_top1 = True
                is_top5 = True
                metrics['top1_hits'] += 1
                metrics['top5_hits'] += 1
            else:
                # 判断 Top-5 (只要真实诊断出现在预测候选列表中即算命中)
                for p_item in p_list_clean:
                    if p_item and (p_item in l_final_clean or l_final_clean in p_item):
                        is_top5 = True
                        metrics['top5_hits'] += 1
                        break
            
            # --- 可视化 Debug ---
            status = "✅ [Top-1 命中]" if is_top1 else "⚠️ [Top-5 命中]" if is_top5 else "❌ [ 未 命 中 ]"
            
            display_label = (l_final_clean[:22] + '..') if len(l_final_clean) > 22 else l_final_clean
            display_pred = (p_final_clean[:22] + '..') if len(p_final_clean) > 22 else p_final_clean
            
            print(f"{status} 标签: {display_label:<24} | 预测Top1: {display_pred:<24} | 候选列表: {p_list_clean}")

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
    # 注意把这里的路径替换为你实际跑出来的文件路径
    evaluate('/Users/yfx/vscode/project/final_proj/result/Qwen2-VL-7B/infer_result/answer.jsonl')