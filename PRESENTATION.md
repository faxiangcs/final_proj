# Presentation video outline

**Project:** GRPO with Verifiable Clinical Rewards for Structured Brain MRI Reporting  
**Author:** Yan Faxiang · 50051798 · HKUST(GZ)  
**Target length:** 7 minutes (hard limit: 10 minutes)  
**Video link:** `<add the final YouTube / OneDrive / Bilibili link here before submitting>`

This file is a recording plan, not the video itself. The assignment requires a presentation video (20 points); record it with slides or a screen-share of the report and code.

---

## 1. Opening (0:00–0:40)

**Say:** “This is my IOTA 5201 Path 2 project. I apply GRPO with verifiable clinical rewards to structured brain MRI reporting, and I compare it against supervised fine-tuning.”

**Show:** title slide with the report title, name, student ID, and HKUST(GZ).

## 2. Problem and why RL (0:40–1:50)

**Say:**

- A report has three parts: abnormal localization boxes, an image description, and a differential/final diagnosis.
- The three parts are coupled: a wrong box or description can change the diagnosis. The model generates the report token by token, so this is a sequential decision process.
- SFT learns reference tokens but does not directly optimize the terminal localization or diagnosis score. Those scores are non-differentiable.
- RL can optimize the finished report directly. NOVA is a hard open-world benchmark: 281 cases, 906 slices, 36 test cases in my split, and 93.7% of test images have a final diagnosis unseen in fine-tuning.

**Show:** one NOVA example (image + history + boxes) and the three-section output format.  
**Do not show:** patient-identifiable text; use a de-identified or redacted example.

## 3. MDP and rewards (1:50–3:10)

**Say:**

- Observation `o = (I, H, P)`; state `s_t = (o, y_<t)`; action is the next token; deterministic transition; terminal reward; `γ = 1`.
- Localization reward: one-to-one greedy box matching at IoU 0.5; `R_loc = TP / (TP + FP + FN)` after coordinate normalization, clipping, NMS, and a three-box cap.
- Diagnosis reward: `0.95 × (0.7 × s_final + 0.3 × s_diff) + 0.05 × s_format`.
- Total reward is the sum, range [0, 2]. No caption reward.
- This is a token-level MDP with a static input, equivalent to a contextual bandit over complete reports. RL is justified because the reward is terminal and non-differentiable; SFT remains a strong alternative.

**Show:** the MDP diagram and the two reward equations. Point out that the localization reward is a Jaccard score, not F1, and that evaluation uses F1.

## 4. Experimental setup (3:10–3:50)

**Say:**

- Two model families: Qwen3.5-9B and LLaVA-1.5-7B; LoRA `r=16`, `α=32`; vision tower frozen, aligner trained.
- SFT: lr `5e-5`, 3 epochs, effective batch 8, 201 optimizer steps for Qwen.
- GRPO: from an SFT checkpoint; lr `5e-6`, 1 epoch, `G=4`, temperature 0.9, `β=0.001`, 266 steps, about 1.3 hours on four RTX 3090s.
- Baselines: center prior, zero-shot, SFT-50 (weak), SFT-201 (strong), and GRPO from each.
- Metrics: localization F1@0.3/0.5/0.75, mean IoU, custom AP proxy; caption token F1/BLEU-4; diagnosis keyword Top-1/Top-5. Paired case-cluster bootstrap with 4000 resamples.

**Show:** the training pipeline diagram and the main results table (or a simplified version).

## 5. Results (3:50–5:20)

**Say:**

- Qwen SFT improves F1@0.5 from 0.068 zero-shot to 0.146 at SFT-50 and 0.401 at SFT-201.
- Weak-start GRPO ck100: 0.146 → 0.216, +0.071, CI [+0.015, +0.125]. Mean IoU 0.300 → 0.344; hit@0.5 17 → 27 of 111.
- **Important caveat:** ck100 was selected after inspecting test results. The final ck266 is 0.191, CI [−0.028, +0.113].
- Strong-start GRPO: 0.390 vs. SFT-201 0.401, Δ −0.011, CI [−0.088, +0.078]. No detected gain.
- Diagnosis and caption do not improve reliably. LLaVA SFT collapses to seven distinct boxes; GRPO does not fix it.

**Show:** the two-panel figure: (a) test F1@0.5 over GRPO steps for weak and strong starts; (b) zero-variance group fractions.

## 6. Why the signal fades (5:20–6:20)

**Say:**

- Zero-variance groups: 23.7% weak, 17.7% strong, 53.8% LLaVA. Those groups give no reward-advantage gradient.
- The hard IoU 0.5 threshold makes the reward sparse: a box a few pixels off gets the same zero as a box on the wrong hemisphere.
- Box calibration: SFT-50 boxes are systematically small (area ratio 0.72). A validation-fitted scale of 1.25 raises test F1@0.5 from 0.146 to 0.242. For SFT-201 the gain is only +0.006. Calibration is post-hoc, not a model improvement.
- Diagnosis is dominated by the clinical-history prior: a text-only Qwen reaches Top-1 0.171, while the image-conditioned zero-shot model reaches 0.108/0.189. This is a small, separate check, not proof that images are useless.
- Data v2/v3 improve validation but not test; validation gains do not transfer.

**Show:** zero-variance bar panel and the calibration numbers.

## 7. Lessons and conclusion (6:20–7:00)

**Say:**

- GRPO can refine an under-trained SFT policy, but it is not a substitute for stronger SFT or better data.
- Compare RL against a sufficiently trained SFT baseline and a simple calibration baseline.
- Monitor reward variance and box diversity; lock checkpoint and metric rules before a new test.
- Limitations: one split, one seed, 36 test cases; retrospective checkpoint selection; lexical caption/diagnosis proxies; no radiologist review.
- The contribution is an implemented RL formulation and an honest empirical analysis of its limits, not a claim that GRPO generally helps.

**Show:** conclusion slide with the three take-away bullets and the GitHub link.

---

## Recording checklist

- [ ] 1080p screen recording; readable font (at least 18 pt on slides).
- [ ] Audio: quiet room, microphone check, no clipping.
- [ ] Show your face in a small webcam window (optional but recommended).
- [ ] Keep the video under 10 minutes; 7 minutes is the target.
- [ ] Do not show NOVA clinical text, patient identifiers, API keys, or local absolute paths.
- [ ] Upload to a link the instructor can open without requesting access; put the link at the top of this file and in the README.
- [ ] Verify the link in an incognito window before submitting.

## Likely questions

1. **Why GRPO rather than PPO or DPO?** GRPO removes the learned critic, which matters with a 24 GB GPU budget, and consumes the programmatic reward directly. PPO with a shared critic and DPO with reward-derived pairs are reasonable alternatives; I did not test them.
2. **Why is the localization reward not F1?** It is a box-set Jaccard score over matched counts; evaluation uses F1. The mapping is `F1 = 2R / (1 + R)` for identical matched counts. This is stated in the paper.
3. **Is the +0.071 gain real?** It is a nominal 95% CI under case resampling, but ck100 was chosen after test inspection and only one seed was run. I report it as exploratory, not confirmatory.
4. **Why does strong SFT not benefit from GRPO?** The reward signal is sparse and the policy already produces low-variance groups; validation gains do not transfer to test. The paper lists several plausible causes but does not isolate one.
5. **Why fine-tune on NOVA if it is evaluation-only?** It is a course research departure, explicitly noted as a limitation. I do not compare absolute numbers to the NOVA leaderboard, and I do not redistribute the data.
