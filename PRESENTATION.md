# 10-minute presentation: English script

**Project:** Reward Shaping for Informative GRPO Updates in Brain MRI Localization

**Presenter:** Yan Faxiang · 50051798 · HKUST(GZ)

**Source:** final `paper/main.tex` and the three-configuration, three-seed experiment

**中文逐字稿：** [PRESENTATION_ZH.md](PRESENTATION_ZH.md)

**检查报告：** [SUBMISSION_AUDIT.md](SUBMISSION_AUDIT.md)

The assignment asks for a 10-minute main presentation, followed by answers to three questions. The script below targets 10 minutes with slide changes and short pauses; rehearse with a timer. Answers follow the main talk and are not part of its 10-minute allocation. The assignment does not specify a presentation language: choose this script or the Chinese version, not both in sequence.

This manuscript is not a recorded video. The release includes the main method, configuration and aggregate evidence. Full training reproduction requires independently obtained NOVA data, model weights and GPUs.

## Slide and timing plan

| Slide | Time | Slide title | What to show |
|---|---|---|---|
| 1 | 0:00–0:40 | Research question | Project title; reward discrimination → localization |
| 2 | 0:40–1:35 | What is MRI localization? | A schematic image, gold/predicted boxes; explain IoU |
| 3 | 1:35–2:40 | RL formulation | Observation → token policy → complete report → verifier → update |
| 4 | 2:40–3:55 | Why GRPO loses a signal | Advantage equation and four tied rewards |
| 5 | 3:55–5:05 | Change the reward | Hard/shaped formulas; Fig. 1a; three-arm reward table |
| 6 | 5:05–6:20 | Controlled experiment | Case split; common initialization/reference/budget; validation selection |
| 7 | 6:20–7:35 | Training evidence | Fig. 1b enlarged; 14.54%, 0.19%, 0.88% |
| 8 | 7:35–8:35 | Localization evidence | Simplified results table with test means, differences and intervals |
| 9 | 8:35–9:20 | Checks and limitations | Mean IoU, box-scale check; one split, shared SFT, historical test access |
| 10 | 9:20–10:00 | Takeaways | Mechanism → intervention → training signal → task evaluation |
| Backup | After 10:00 | Three required questions | The three answers below |

Keep slides visual and short. The following paragraphs are speaker notes, not slide text. Use the supplied graph with readable labels; the full paper table is too dense to show without enlargement.

## Slide 1 — Research question · 0:00–0:40

<!-- SPEECH_START -->
Good afternoon. My project asks a specific reinforcement learning question: when a reward gives the same score to outputs of different quality, can changing that reward give GRPO a more useful learning signal?

I study this in brain MRI localization, where a language model generates boxes around abnormalities. The main finding is a large reduction in groups with tied rewards. The downstream localization improvement is smaller and uncertain. I will show the mechanism, the controlled experiment, and the evidence supporting each conclusion.
<!-- SPEECH_END -->

**讲者提示：** 开场直接分清“学习信号改善”和“任务性能提升”，不要声称全面提升医学报告质量。

## Slide 2 — Problem and metric · 0:40–1:35

<!-- SPEECH_START -->
A structured MRI report contains lesion coordinates, an image description, and a proposed diagnosis. I focus on the coordinates. The input is one MRI slice, clinical history, and a reporting instruction. The output includes up to three predicted boxes.

The geometric measure is intersection over union, or IoU: the overlap area divided by the union area. A predicted box counts as a match when its IoU with an expert box reaches zero point five. Predictions and expert boxes are matched one to one.

Supervised fine-tuning learns the reference coordinate tokens. But different strings can describe similarly useful boxes. This motivates directly optimizing geometric feedback on generated reports.
<!-- SPEECH_END -->

**讲者提示：** 指向预测框与参考框的交集和并集，让不熟悉医学影像的听众理解任务即可。

## Slide 3 — RL formulation · 1:35–2:40

<!-- SPEECH_START -->
Here is the reinforcement learning formulation. The observation contains the image, history, and instruction. At each step, the state also includes the generated prefix. The action is the next token, and the transition appends it to that prefix.

An episode ends at the end-of-sequence token or after five hundred and twelve generated tokens. Intermediate rewards are zero. A deterministic verifier scores the completed report using training annotations. The discount is one, and evaluation never supplies those annotations to the policy.

The image is static, so the complete report can also be viewed as one contextual-bandit action. This is sequence generation with terminal feedback, rather than an interactive clinical control task. RL is useful because it can optimize this non-differentiable, report-level score without differentiating through the box parser.
<!-- SPEECH_END -->

**讲者提示：** 用 contextual bandit 的准确表述回答潜在质疑；不要把“生成 token”包装成未实现的临床决策流程。

## Slide 4 — GRPO and tied rewards · 2:40–3:55

<!-- SPEECH_START -->
I use Group Relative Policy Optimization, or GRPO. For the same input, the policy samples four complete reports. Each report receives a terminal reward. Its advantage is the reward minus the group mean, divided by the group standard deviation plus a small stabilizer.

This avoids learning a separate value critic. A clipped policy objective improves the rewarded completions, while a KL penalty limits drift from a frozen SFT reference.

Now consider four reports whose boxes have IoUs of zero point one, zero point two, zero point three, and zero point four five. All fail the hard matching threshold. If their diagnosis rewards are also equal, their total rewards tie, and every task advantage becomes zero.

The group can still produce a KL update, but no task-reward update. The learning problem is that the reward cannot distinguish these partially successful proposals.
<!-- SPEECH_END -->

**讲者提示：** 先指出总奖励同分的条件，再解释优势为零；不能只因定位分同分就断言总优势为零。

## Slide 5 — Reward intervention · 3:55–5:05

<!-- SPEECH_START -->
The original localization reward is true positives divided by true positives plus false positives plus false negatives. This is a box-set Jaccard score. The evaluation metric is F1, so I distinguish their definitions.

I retain half of this hard reward and add half of a continuous overlap term. That term sums positive, one-to-one matched IoUs, doubles the sum, and divides by the number of predicted and reference boxes. Unmatched boxes remain in the denominator.

In the earlier example, the shaped scores become zero point zero five, zero point one, zero point one five, and zero point two two five. They now give different task advantages.

I compare Original, Shaped plus Diagnosis, and Shaped only. The first contrast changes only localization scoring. The second also removes the diagnosis component and its format bonus.
<!-- SPEECH_END -->

**讲者提示：** 连续项不是逐 token 中间奖励，也不是本文首次发明 IoU 奖励。图 1a 是解析示意，不能当作实测训练曲线。

## Slide 6 — Controlled experiment · 5:05–6:20

<!-- SPEECH_START -->
The data are a filtered NOVA subset with two hundred and thirty-five cases and seven hundred and seventy-nine images. Cases, rather than individual slices, are split between training, validation, and test. The test set has thirty-six cases and one hundred and eleven images.

All nine GRPO runs start from the same Qwen SFT checkpoint after two hundred and one updates. They share the same frozen reference, prompt, data, and training budget. Each reward configuration uses seeds forty-two, forty-three, and forty-four and runs for two hundred and sixty-six updates. Only the reward configuration changes in the controlled contrast.

Checkpoints are selected by validation localization F1. The configuration is selected by the mean validation score across seeds. Choices are locked before new test inference. I also report final checkpoints at the common training endpoint, and bootstrap differences by resampling seeds and whole cases.
<!-- SPEECH_END -->

**讲者提示：** PPT 可写 lr 5e−6、G=4、β=.001、LoRA r16；无需把全部参数逐字读出。

## Slide 7 — Training evidence · 6:20–7:35

<!-- SPEECH_START -->
This is the clearest result. The graph measures the fraction of sampled groups with zero total-reward standard deviation, averaged over training steps and seeds.

Keeping diagnosis reward fixed, replacing the hard localization score reduces this fraction from fourteen point five four percent to zero point one nine percent. The pattern holds in every seed. Original ranges from about fourteen to fifteen percent, while Shaped plus Diagnosis stays below zero point four percent. Shaped only averages zero point eight eight percent.

The formula explains how partial overlap can remove ties, and the logs show that ties actually become much less common during training.

These are separately evolving policies, so the rates reflect both the reward and the reports each policy samples. This is not rescoring identical fixed rollouts. More nonzero advantages also do not guarantee better localization. That requires the next evaluation.
<!-- SPEECH_END -->

**讲者提示：** 这是主要结果，可指图停顿。圆点是各种子，不能说成置信区间。

## Slide 8 — Localization evidence · 7:35–8:35

<!-- SPEECH_START -->
Validation selects Shaped only. Its mean test F1 is zero point four one eight nine, compared with zero point four zero zero nine for SFT. The gain is one point eight F1 percentage points. All three selected seeds are above the shared SFT baseline.

However, the paired seed-and-case bootstrap interval runs from minus zero point zero five six six to plus zero point zero nine zero eight. It includes zero, so I cannot claim a statistically established localization improvement.

For completeness, Original averages zero point four zero zero four, and Shaped plus Diagnosis averages zero point four zero eight nine. The smallest tie rate does not produce the highest localization score. Reward discrimination and task performance are distinct outcomes.
<!-- SPEECH_END -->

**讲者提示：** 只重点读 SFT 和 Shaped only；其余配置放表格保持完整。+0.0180 是 1.80 个百分点，不是 18%。

## Slide 9 — Checks and limitations · 8:35–9:20

<!-- SPEECH_START -->
Two checks support the geometric interpretation. Mean IoU rises from about zero point four zero three to zero point four four four. Validation-fitted box scaling improves SFT slightly, but leaves Shaped only ahead by one point two F1 points, still with an interval including zero.

The scope remains limited: one split, thirty-six test cases, and one shared SFT initialization. This test split was inspected during earlier development. There is no compute-matched continued-SFT control. Diagnosis and description scores do not improve, so I do not claim better overall reporting.
<!-- SPEECH_END -->

**讲者提示：** 不展开所有负结果，只保留影响结论可信度的边界。

## Slide 10 — Takeaways · 9:20–10:00

<!-- SPEECH_START -->
My contribution is a controlled study connecting reward design to the GRPO learning signal and then to localization behavior.

Continuous IoU shaping sharply reduces tied-reward groups. The localization point estimate is positive, but its generalization benefit remains uncertain. Existing visual RL work already uses IoU; the contribution here is the controlled comparison and the separation of optimization evidence from task evidence.

The practical lesson is to evaluate both: does the reward distinguish sampled outputs, and do those distinctions improve the task? Thank you.
<!-- SPEECH_END -->

## After the main talk — three required questions

These are the assignment's three questions. Keep them as three separate answers after the main talk, in the recording or during live questions. Budget roughly two additional minutes in total; actual timing depends on delivery.

### 1. Why use RL?

**Answer:**

SFT optimizes reference-token likelihood, while our target is geometric overlap of the generated boxes. The score depends on parsed, completed reports and is not differentiable through the generated text. RL can optimize the expected terminal score with annotation-based feedback. The sequential decisions are token choices conditioned on the prefix; at the report level, this is also a contextual-bandit problem. I use RL to investigate this objective mismatch, not because the experiment proves RL is universally better than SFT.

### 2. Why did you choose this method?

**Answer:**

GRPO compares several sampled reports for the same input, so it directly uses verifiable rewards without a separate learned value critic. This is convenient for a large vision-language policy and lets me inspect whether groups supply nonzero advantages. I chose continuous IoU shaping because a hard threshold loses distinctions between partially correct boxes. The two components retain threshold incentives and add graded overlap feedback. I did not compare PPO or DPO, so this is a design justification, not a claim that GRPO is the best algorithm.

### 3. What is your contribution?

**Answer:**

I formulated MRI report localization as sequence generation with verifiable terminal rewards and implemented the reward comparison. The empirical contribution is three reward configurations across three seeds, with common initialization, reference and budget, plus validation-based selection and case-aware uncertainty. The strongest observation is the reduction of tied groups from 14.54 to 0.19 percent with diagnosis fixed. Test F1 rises by 1.80 points for the selected localization-only policy, but the interval includes zero. I claim clearer optimization feedback, rather than proven clinical or generalization gains.

## Extra questions for rehearsal

**Is reward shaping the same as dense intermediate reward?** No. There is still one terminal reward. “Continuous” refers to the geometric score assigned to a completed report.

**Does this prove that diagnosis hurts localization?** No. Removing diagnosis changes the reward mixture and its format component. This experiment does not isolate the mechanism behind the observed ranking.

**Why use NOVA if it is evaluation-only?** This is an explicitly disclosed internal adaptation study. It departs from NOVA's benchmark protocol, so I do not compare these numbers with its official zero-shot leaderboard.

**Do all-positive seeds prove significance?** No. They share a test split and an SFT initialization. The seed-and-case interval includes zero and historical test access remains a limitation.

**Is this potential-based shaping with an optimal-policy guarantee?** No. It modifies the terminal objective; I make no policy-invariance claim.

## Before recording/submission

1. Rehearse the main script to approximately 10 minutes, leaving a brief pause when pointing to each graph. Keep the three answers after the talk.
2. Show current results only. Do not reintroduce the old weak-start checkpoint-100 result or LLaVA collapse as the main evidence.
3. Use the current submission branch and its verified commands; the legacy branch preserves earlier tools. Distinguish CPU verification from a fresh GPU training rerun.
4. Record the actual video unless exempted by selection for the December 1 live presentation. A script or URL is not the same as the required video file.
5. Rebuild the final ZIP from the final version and explicitly include the video when required. The old ZIP contains an earlier paper.
