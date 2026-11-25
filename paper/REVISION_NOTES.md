# 论文修订说明

本次按 `IOTA_Final_Project.pdf` 的 Path 2（将 RL 应用于自选问题）组织论文。成稿为英文 IEEE 双栏 **5 页，含参考文献**。沿用已有实验和选择规则，没有重新训练、调整测试集或选择新的最优 checkpoint。

## 主线与证据

论文围绕一个问题展开：**连续 IoU 奖励能否减少 GRPO 中无法提供任务优势的同分组，以及这种信号变化能否带来定位改善？**

1. **问题机制。** 原始定位奖励使用 IoU≥0.5 的硬匹配。低于阈值但质量不同的框可能得到相同分数；若总奖励全部相同，GRPO 的组内标准化优势为零。正文明确该条件，避免把“定位分相同”直接等同于“总优势为零”。
2. **干预与对照。** Original→Shaped + Dx 仅替换定位奖励，诊断奖励不变；Shaped + Dx→Shaped only 移除诊断分量。九次运行共享 SFT-201、冻结的 SFT KL reference、数据和训练预算。第二个比较同时改变奖励组合，不能单独证明“诊断任务造成负迁移”。
3. **最稳定的实验证据。** 日志中的组内总奖励同分率从 14.54% 降至 0.19%；三个种子分别处于 14.10%–15.04% 和 0.00%–0.38%。Shaped only 为 0.88%。这些是各策略训练过程中实际采样到的组，不是对同一批固定 rollout 重打分。
4. **实际任务验证。** 验证集选出的 Shaped only 在测试集三个种子上均高于 SFT，均值 0.4009→0.4189；差值的 95% 区间为 [−0.0566,+0.0908]。因此正文只将其表述为幅度有限的定位点估计改善，不写成显著或普遍优于 SFT。
5. **结论范围。** 同分组大幅减少是训练诊断上的稳定观察；测试集定位差异仍有较大不确定性。诊断和描述分数未改善，仅在讨论中用一段交代，不据此宣称整体报告质量或临床诊断能力提升。

## 关键数字与原始来源

以下路径均相对于仓库根目录。

| 证据或数字 | 对应文件 |
|---|---|
| 三组奖励、三种子、266 steps、共享初始化和选择规则 | `experiments/grpo_refinement_20261119/plan.json` |
| 同分组均值及种子范围 | `output/grpo_refinement_20261119/{reference,dense,loc_only}_seed{42,43,44}/v*/logging.jsonl`，字段 `frac_reward_zero_std`；每次运行取 266 步均值，再对种子取均值 |
| 原始与连续定位奖励的定义 | `medical_reward.py`、`src/refinement_rewards.py` |
| 验证集轨迹、checkpoint 候选与选择 | `experiments/grpo_refinement_20261119/selection.json` |
| 验证集确定 Shaped only，均值 0.4936 | `experiments/grpo_refinement_20261119/configuration_selection.json` |
| 所有配置的测试均值、种子标准差、95% 区间 | `experiments/grpo_refinement_20261119/aggregate_results.json` |
| 各运行 IoU、校准、诊断、描述指标 | `experiments/grpo_refinement_20261119/test_results.json` |
| SFT 原始定位 0.4009、IoU 0.4028、文本指标 | `result/qwen_sft_test_checkpoint-201/report.json` |
| SFT 校准后的定位 0.4069 | `aggregate_results.json` 中 `baseline_calibrated`；校准尺度由验证集确定 |
| 病例切分、标签及图像检查 | `data/nova_sft/audit_report.json` |
| 冻结 reference 的检查 | 各正式运行的 `reference_audit.json` 和实验目录下的 `training_done.json` |

论文图 1 的 (a) 是奖励公式的解析示意，(b) 是实测训练日志，(c) 是锁定候选 checkpoint 的验证集结果。三者在图注中明确区分，避免将示意图误作实测证据。阴影表示三个种子的最小值至最大值，不是置信区间。

原始代码把 `TP/(TP+FP+FN)` 命名为 `localization_f1`，但其数学定义是 box-set Jaccard。正文按实际公式说明，并与评测用的 `2TP/(2TP+FP+FN)` 区分；未改动训练或评测实现。

## 内容取舍与作业要求

- **摘要、引言与相关工作：** 聚焦奖励设计和定位，给出明确研究问题与贡献，不再以笼统的“结构化报告整体提升”为目标。
- **方法：** 补全 observation/state/action/transition/termination/reward，说明为什么用 RL、为什么选择 GRPO，并如实交代完整报告也可视为 contextual bandit 的动作。
- **实验与结果：** 增加奖励对照表和三联证据图；主表保留全部三组、全部测试种子和不确定性。共同训练终点和尺度校准作为简短补充检验。
- **较弱实验：** 删去早期单种子、弱 SFT 起点及其他 reference 设置的历史叙述；诊断和描述退化压缩为一段，保留对结论范围必要的信息。
- **讨论与结论：** 保留测试集曾在早期开发中查看、单一 SFT 初始化、未提供等算力继续 SFT 对照等关键边界，避免由现有结果推导未验证的因果结论。
- **复现与格式：** 更新 `REPRODUCE.md`；通过 `make_figure.py` 从原始日志和保存的统计生成图表，编译 `main.pdf`，检查页数、引用及版面。

修订前的 TeX、PDF、绘图脚本和复现说明保存在 `paper/review/before_evidence_revision_20261121_154237/`。该目录仅供本地对照，不属于提交论文。
