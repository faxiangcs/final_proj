import re
from collections import Counter
from typing import List
from swift.plugin import ORM, orms

# ================= 提取工具库 =================
def extract_bboxes(text):
    pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    matches = re.findall(pattern, text)
    return [list(map(float, m)) for m in matches]

def extract_diagnosis(text):
    diff_diag, final_diag = "", ""
    reasoning_text = text
    if "【诊断推理】：" in text:
        reasoning_text = text.split("【诊断推理】：")[-1]
        
    final_match = re.search(r'最终(?:的)?(?:临床)?诊断(?:是|为|：|:)\s*(.*?)(?:。|\n|$)', reasoning_text)
    if final_match:
        final_diag = final_match.group(1).strip()
    elif "最终诊断" in reasoning_text:
        parts = re.split(r'最终(?:的)?(?:临床)?诊断(?:是|为|：|:)\s*', reasoning_text)
        if len(parts) > 1:
            final_diag = parts[-1].split('。')[0].split('\n')[0].strip()
            
    diff_match = re.search(r'(?:鉴别诊断(?:包括)?|可能是|考虑)+\s*(.*?)(?:。?\s*结合特征|。?\s*最终|\n|$)', reasoning_text)
    if diff_match:
        diff_diag = diff_match.group(1).strip()

    # 终极剥壳去前缀
    def strip_nonsense(diag_str):
        if not diag_str: return ""
        old_str = ""
        while diag_str != old_str:
            old_str = diag_str
            diag_str = re.sub(r'^(考虑|鉴别诊断(?:包括)?|可能是|the possibility of\s*|a\s+|an\s+)', '', diag_str, flags=re.IGNORECASE).strip()
        return diag_str

    return strip_nonsense(diff_diag), strip_nonsense(final_diag)

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def calculate_diagnosis_overlap(pred_text, true_diag):
    if not true_diag or not pred_text: return 0.0
    pred_text, true_diag = pred_text.lower(), true_diag.lower()
    if true_diag in pred_text: return 1.0
        
    true_tokens = set(re.findall(r'\b[a-z0-9]+\b', true_diag))
    pred_tokens = set(re.findall(r'\b[a-z0-9]+\b', pred_text))
    stop_words = {'of', 'with', 'the', 'and', 'in', 'type', 'pattern', 'nos', 'disease', 'syndrome', 'a', 'an'}
    true_keywords = true_tokens - stop_words
    if not true_keywords: true_keywords = true_tokens
        
    overlap = true_keywords.intersection(pred_tokens)
    return len(overlap) / len(true_keywords) if true_keywords else 0.0

def align_solutions(completions, kwargs):
    solutions = kwargs.get('solution')
    if not solutions: return [""] * len(completions)
    if len(solutions) == len(completions): return solutions
    num_gens = len(completions) // len(solutions)
    aligned = []
    for s in solutions: aligned.extend([s] * num_gens)
    return aligned

# ================= 核心奖励函数（极简双塔架构） =================

# 1. 异常定位奖励 (保持原样，负责视觉空间理解)
class LocalizationReward(ORM):
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        solutions = align_solutions(completions, kwargs)
        for comp, label in zip(completions, solutions):
            c_text = comp if isinstance(comp, str) else comp[0]['content']
            l_text = label if isinstance(label, str) else label[0]['content'] if isinstance(label, list) else str(label)
            
            p_boxes = extract_bboxes(c_text)
            l_boxes = extract_bboxes(l_text)
            
            if not l_boxes:
                rewards.append(0.5 if not p_boxes and "未发现" in c_text else 0.0)
                continue
            if not p_boxes:
                rewards.append(0.0)
                continue
                
            max_iou = max([calculate_iou(pb, lb) for pb in p_boxes for lb in l_boxes], default=0)
            if max_iou > 0.5: rewards.append(2.0)
            elif max_iou > 0.3: rewards.append(1.0)
            else: rewards.append(0.0)
        return rewards

# 2. 诊断复合奖励 (格式纪律 + 诊断准确度)
class DiagnosisReward(ORM):
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        solutions = align_solutions(completions, kwargs)
        
        for comp, label in zip(completions, solutions):
            c_text = comp if isinstance(comp, str) else comp[0]['content']
            l_text = label if isinstance(label, str) else label[0]['content'] if isinstance(label, list) else str(label)
            
            score = 0.0
            
            # --- 第一关：铁腕格式审查 ---
            has_diag = "【诊断推理】：" in c_text
            if has_diag:
                diag_text = c_text.split("【诊断推理】：")[-1].strip()
                pattern = r'^\s*考虑鉴别诊断包括.+。结合特征，最终诊断为：.+'
                if re.search(pattern, diag_text, re.DOTALL):
                    score += 0.5  # 格式完美，拿到底薪
                else:
                    score -= 0.5  # 格式乱写，直接倒扣
            else:
                score -= 0.5 # 连【诊断推理】都没有，扣分！
                
            # --- 第二关：诊断重合度审查 ---
            p_diff, p_final = extract_diagnosis(c_text)
            _, l_final = extract_diagnosis(l_text)
            
            # 如果真实标签没有诊断结论，只给格式分就结束
            if not l_final:
                rewards.append(score)
                continue
                
            final_hit_ratio = calculate_diagnosis_overlap(p_final, l_final)
            diff_hit_ratio = calculate_diagnosis_overlap(p_diff, l_final)
            
            # 阶梯式奖金（叠加在格式底薪之上）
            if final_hit_ratio == 1.0:
                score += 3.0
            elif final_hit_ratio >= 0.5:
                score += 2.0
            elif final_hit_ratio > 0:
                score += 1.5
            elif diff_hit_ratio == 1.0:
                score += 2.0
            elif diff_hit_ratio >= 0.5:
                score += 1.0
                
            rewards.append(score)
                
        return rewards

# ================= 注册挂载点 (仅保留两个) =================
orms['localization_reward'] = LocalizationReward
orms['diagnosis_reward'] = DiagnosisReward