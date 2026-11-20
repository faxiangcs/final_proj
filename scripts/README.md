# Nova 训练命令

> 说明：本文件是详细实验记录。公开仓库不包含 `data/`、`output/`、`result/`、
> 模型权重或预测文件；下文这些路径指本机实验产物。示例中的 `$HOME/...` 请按
> 实际路径替换；原始运行使用 `/data/yanfaxiang/...`。

所有脚本都以仓库根目录为基准，默认读取 `data/nova_sft/`，不再引用根目录的
`demo_*.jsonl` / `grpo_nova_*.jsonl`（这些是旧格式，已弃用）。

## 环境

本机已安装并验证的训练环境：

```bash
export PATH=$HOME/venvs/ms-swift/bin:$PATH
```

- 位置：`$HOME/venvs/ms-swift`（conda prefix，Python 3.11.18）
- `torch 2.6.0+cu124`、`transformers 5.20.2`、`ms-swift 4.6.1`
- 额外：`qwen_vl_utils 0.0.14`、`msgspec`
- 重建：`bash scripts/setup_swift_env.sh`

注意：

- `transformers` 必须是 5.x，4.57 没有 `qwen3_5`；
- ms-swift 4.x 用 `--tuner_type lora`，旧版的 `--train_type` 已被移除；
- ms-swift 4.x 的多模态数据必须把图片放在 row 级 `images` 字段
  （`{"messages": [...], "images": ["/path/img.png"]}`）。message 内写
  `image` 会被 `RowPreprocessor` 静默丢弃，训练会退化成纯文本；
  `src/build_dataset.py` 已按新格式输出，审计 A02b 会检查；
- Qwen3.5 的 `causal_conv1d` 需要 nvcc，本机未安装，会走参考实现；
  实测 GRPO 影响可控（32 token × 4 条约 5.3s/step）；
- 没有安装 `flash-linear-attention`：本机 Triton 3.2 下反而慢约 8 倍；
- 已验证 1-step：LLaVA SFT（loss 1.66，13.8G）、Qwen3.5-9B SFT（loss 2.06，18.4G）、
  LLaVA GRPO（14.1G）、Qwen3.5-9B GRPO（18.5G，5.3s/step）。

推荐统一入口：

```bash
bash scripts/train.sh sft
bash scripts/train.sh grpo
```

## 数据位置

```text
data/nova_sft/
├── train.jsonl          # SFT 训练
├── val.jsonl            # SFT 验证
├── test.jsonl           # 推理/最终测试
├── grpo_train.jsonl     # GRPO 训练（messages + solution）
├── grpo_val.jsonl       # GRPO 验证
├── grpo_test.jsonl      # GRPO 测试
├── audit.jsonl          # 逐样本结构化 target
├── manifest.json        # 版本、split、过滤统计
├── audit_report.json    # A01–A16 + A05b 审计结果
└── README.md            # 数据说明
```

图片仍来自 `Nova/images/`，数据里记录的是绝对路径。训练机路径不同时，用
`--image_root` 重新生成，不要手工改 JSONL。

## SFT

```bash
bash scripts/train.sh sft
# 等价于：bash scripts/run_sft.sh
```

默认：

- `model=llava-hf/llava-1.5-7b-hf`；本机若存在
  `$HOME/models/llava-1.5-7b-hf` 会自动改用本地权重（已做 SHA256 校验）
- `max_length=2048`
- LoRA `r=8, alpha=32`
- `lr=1e-4`，1 epoch
- `freeze_vit=true`，`freeze_aligner=true`
- 每 50 step eval / save

常用覆盖：

```bash
# 降低学习率、跑 2 epoch（方案 6.2 建议）
SFT_LR=5e-5 SFT_EPOCHS=2 bash scripts/train.sh sft

# 增大 LoRA、解冻 aligner（显存允许时）
LORA_RANK=16 FREEZE_ALIGNER=false bash scripts/train.sh sft

# 四卡 DDP 全量 SFT（本机 4×3090 的实际跑法）
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
OUTPUT_DIR=$PWD/output/nova_sft_full \
SFT_LR=5e-5 SFT_EPOCHS=3 LORA_RANK=16 FREEZE_ALIGNER=false \
SFT_BATCH_SIZE=1 SFT_GRAD_ACCUM=2 EVAL_STEPS=25 SAVE_STEPS=25 \
  bash scripts/train.sh sft

# 只打印命令，不执行
DRY_RUN=1 bash scripts/train.sh sft
```

`NPROC_PER_NODE` 由 ms-swift 转成 `torch.distributed.run --nproc_per_node`；
有效 batch = `SFT_BATCH_SIZE × GPU 数 × SFT_GRAD_ACCUM`。全量 train 533 条，
上面配置约 200 optimizer steps / 3 epoch。

## GRPO

默认使用合并后的 SFT 模型 `nova_sft_merged/`。注意：这个目录是上一轮的旧合并模型；
新 SFT 跑完后，请用 `MODEL=...` 指向新模型，或按下面第 2 种方式直接接 adapter。

```bash
bash scripts/train.sh grpo
# 等价于：bash scripts/run_grpo.sh
```

默认：

- `max_length=2048`
- `max_completion_length=512`
- `num_generations=4`，`temperature=0.9`，`beta=0.001`
- `reward_funcs=localization_reward diagnosis_reward`
- `external_plugins=medical_reward.py`
- `use_vllm=false`
- `lr=2e-5`，1 epoch

如果刚训完 SFT，有两种接法：

```bash
# 1) 先 merge LoRA，再跑 GRPO
swift merge-lora \
  --adapters output/nova_sft/checkpoint-50 \
  --output_dir nova_sft_merged_new
MODEL="$PWD/nova_sft_merged_new" bash scripts/train.sh grpo

# 2) 直接用 base model + adapter
MODEL=$HOME/models/llava-1.5-7b-hf \
ADAPTERS=output/nova_sft/checkpoint-50 \
bash scripts/train.sh grpo
```

常用覆盖：

```bash
NUM_GENERATIONS=8 TEMPERATURE=1.0 bash scripts/train.sh grpo
TOP_P=0.95 GRPO_LR=5e-6 bash scripts/train.sh grpo
REWARD_WEIGHTS="1 1" bash scripts/train.sh grpo
GRPO_MAX_STEPS=1 bash scripts/train.sh grpo   # smoke test
GRPO_EVAL_STRATEGY=steps bash scripts/train.sh grpo
DRY_RUN=1 bash scripts/train.sh grpo
```

Qwen3.5-9B 接 SFT adapter 的 GRPO（4×3090，已验证 18.5G 显存）：

```bash
NPROC_PER_NODE=4 \
MODEL=$HOME/models/Qwen3.5-9B \
ADAPTERS=output/qwen_sft_full/v0-20261117-192623/checkpoint-50 \
LORA_RANK=16 LORA_ALPHA=32 FREEZE_ALIGNER=false FREEZE_VIT=true \
GRPO_LR=5e-6 GRPO_GRAD_ACCUM=2 NUM_GENERATIONS=4 \
TEMPERATURE=0.9 TOP_P=0.95 ENABLE_THINKING=false \
LOGGING_STEPS=1 SAVE_STEPS=25 EVAL_STRATEGY=no \
OUTPUT_DIR=$PWD/output/qwen_grpo_v1 \
  bash scripts/run_grpo.sh
```

注意 `LORA_RANK`/`LORA_ALPHA` 必须与 SFT adapter 一致（Qwen SFT 用的是 r=16/α=32），
否则不能接着训练同一个 adapter。

GRPO 跑完后并行评测多个 checkpoint：

```bash
NAME_PREFIX=qwen_grpo_v1 bash scripts/eval_grpo_checkpoints.sh \
  output/qwen_grpo_v1/v0-*/checkpoint-50 \
  output/qwen_grpo_v1/v0-*/checkpoint-100 \
  output/qwen_grpo_v1/v0-*/checkpoint-175 \
  output/qwen_grpo_v1/v0-*/checkpoint-266
```

每个 checkpoint 占一张 GPU（`CUDA_VISIBLE_DEVICES=0..N-1`），推理设置与 SFT
baseline 一致（pt、greedy、512 tokens），产出 `result/<name>/{pred.jsonl,report.json}`。

> `medical_reward.py` 已重写，只注册 `localization_reward` 和
> `diagnosis_reward` 两个函数；格式分折进 `diagnosis_reward`（0.05），
> `description_reward` 仍未实现，不要写进 `REWARD_FUNCS`。

> 从 ck201 起步的复跑（2026-11-19，`OUTPUT_DIR=output/qwen_grpo_ck201`）：
> 266 步、1h18m。ck266 的 val raw F1@0.5 **0.4647**（SFT ck201 0.3746，
> Δ+0.090 CI 显著），val 最优 scale **1.00**（不再需要放大框）；但 test
> 0.3904 vs SFT ck201 0.4069（CI 跨 0），统计打平。GRPO 在强 SFT 起点上
> 没有额外 test 收益，起点选择比 GRPO 本身更关键。

## 推理

```bash
# 合并模型
bash inference.sh

# LoRA adapter
MODEL=$HOME/models/llava-1.5-7b-hf \
ADAPTERS=output/nova_sft/checkpoint-50 \
bash inference.sh
```

默认读取 `data/nova_sft/test.jsonl`，`max_new_tokens=512`。

## 评测

```bash
# swift infer 输出自带 labels，可直接评
python3 src/eval_nova.py --pred infer_result/answer.jsonl --out eval_report.json

# baseline 只有预测、gold 在测试集里时
python3 src/eval_nova.py \
  --pred result/qwen35_baseline.jsonl \
  --gold data/nova_sft/test.jsonl \
  --coord auto \
  --out result/qwen35_baseline_report.json
```

默认 `--coord auto` 按整个文件的坐标峰值判断 `[0,1]` / `[0,100]` / 像素 /
`[0,1000]`；480 与 1000 的边界有歧义时显式传 `--coord pixel|norm1|norm100|norm1000`。
LLaVA-NeXT 的 `[0,1]` 和 Qwen 的 `[0,1000]` 都会被换算回 480 像素空间再做 IoU。

输出包含：多框一对一匹配的 P/R/F1@0.3/0.5/0.75、mAP@[.50:.95]、mean IoU、
caption token F1 / BLEU-4、诊断 Top-1/Top-5（strict + lenient + keyword）、
case 级 Top-1、以及解析质量统计（空预测、无框、无 final、坐标裁剪/丢弃数）。

诊断的 `keyword` 口径只要求预测出现 gold 诊断的关键实体词（如 `pilocytic`、
`percheron`、`craniopharyngioma`），不要求复述 gold 整句。别名表默认读取
`src/diagnosis_aliases.json`，可用 `--aliases path.json` 覆盖；没有别名的 gold
标签回落到 lenient 匹配。strict/lenient 仍然保留，用作严格口径对照。

`eval_all_metric.py` 保留为兼容入口：

```bash
python3 eval_all_metric.py infer_result/answer.jsonl --gold data/nova_sft/test.jsonl
```

统一入口：

```bash
PRED=result/qwen35_baseline/pred.jsonl \
GOLD=data/nova_sft/test.jsonl \
OUT=result/qwen35_baseline/report.json \
bash scripts/run_eval.sh
```

## Box 尺寸校准（不重训）

SFT 输出存在系统性的框尺寸偏置（ck50 的预测/金标面积比中位数约 0.72）。
主模型应取**训练满 3 epoch 的 final checkpoint（ck201）**：val 上 ck75/ck175/ck201
统计打平（35 case 不足以区分），test 上 ck201 raw F1@0.5 0.4009 / calibrated
0.4069，明显高于 ck50 的 0.1456。先在各 checkpoint 的 val 预测上拟合一个全局
scale，再应用到 test：

```bash
# 1) val 推理（每个 checkpoint 一张卡）
export PATH=$HOME/venvs/ms-swift/bin:$PATH
MODEL=$HOME/models/Qwen3.5-9B \
TEST_FILE=$PWD/data/nova_sft/val.jsonl \
NAME_PREFIX=qwen_sft_val \
bash scripts/eval_grpo_checkpoints.sh \
  output/qwen_sft_full/v0-*/checkpoint-50 \
  output/qwen_sft_full/v0-*/checkpoint-75 \
  output/qwen_sft_full/v0-*/checkpoint-100 \
  output/qwen_sft_full/v0-*/checkpoint-125

# 2) 在 val 上拟合 scale，输出 raw / calibrated 曲线
python3 scripts/fit_box_scale.py \
  --pred result/qwen_sft_val_checkpoint-50/pred.jsonl \
  --gold data/nova_sft/val.jsonl \
  --out result/qwen_sft_val_checkpoint-50/box_scale.json

# 3) 用选出的 scale 评 test（raw 和 calibrated 必须同时报；1.25 是 ck50 的 val 拟合值）
python3 src/eval_nova.py \
  --pred result/qwen_sft_ck50/pred.jsonl \
  --gold data/nova_sft/test.jsonl \
  --box_scale 1.25 \
  --out result/qwen_sft_ck50/report_calibrated.json

# 4) 多个 checkpoint 的 val 报告排序
python3 scripts/compare_checkpoints.py result/qwen_sft_val_checkpoint-*/report.json

# 5) 偏差诊断（面积比 / log-area 斜率）和配对 bootstrap
python3 scripts/box_bias_report.py \
  --pred result/qwen_sft_ck50/pred.jsonl \
  --gold data/nova_sft/test.jsonl \
  --out result/qwen_sft_ck50/box_bias.json
python3 scripts/calibration_ci.py \
  --pred result/qwen_sft_ck50/pred.jsonl \
  --gold data/nova_sft/test.jsonl \
  --scale 1.25 \
  --out result/qwen_sft_ck50/calibration_ci.json
```

`compare_checkpoints.py` 按 F1@0.5 排序；`box_bias_report.py` 里
`log_area_slope < 1` 表示大框被系统性压小、小框被放大（回归到训练集均值）；
`calibration_ci.py` 以 case 为单位做 cluster bootstrap，给出校准前后的配对
置信区间，报告里应同时给出 raw / calibrated / delta / 95% CI。

注意：每个模型要单独拟合，不要把 SFT 的 scale 套到 GRPO 上。实测 val 拟合：
SFT ck50 1.25、ck75 1.15、ck100/ck125/ck150/ck175/ck201 1.20；GRPO ck100 是
1.05（它的框整体偏大，test 上调到 1.15 虽然更高，但属于在 test 上调参，
不能写进报告）。ck50 校准收益最大（+0.096，CI 不含 0），ck201 只有 +0.006，
说明训练越久框越准、校准的边际收益越低。校准是后处理，不是模型能力提升，
报告中要分开写。

## SFT 侧修复（data v2，需重训）

针对 SFT 的三个数据侧问题：

- 框坐标按两位小数渲染，一个框 31 token；整数只需 19 token，且没有 `.00` 噪声；
- 定位监督只占 label 的不到三分之一，模型容易把框回归到训练集平均尺寸；
- 45 条 train panel caption 全部属于 10 个“只有 panel 图”的 case；删掉会同时
  丢掉这 10 个病种，所以 v2 **保留**它们（`--drop_train_panel_captions` 只对
  mixed case 生效，在本数据集上是 no-op）；
- 少数 final diagnosis 是整句描述（如 `Regarding the whole file, we believe it
  is falx EMH`），会让 exact/lenient 匹配永远失败。

对应改动（`data/nova_sft_v2`，**test 集仍是同一批 36 case / 111 张**，可直接和旧 run 对比）：

```bash
python3 src/build_dataset.py \
  --nova_root Nova --image_root Nova/images \
  --out_dir data/nova_sft_v2 --seed 42 \
  --integer_boxes --aux_boxes_ratio 0.25 \
  --clean_final
python3 src/audit_data.py --data_dir data/nova_sft_v2

# 与旧 Qwen run 相同的 4 卡配置，只改数据（lr/epoch/LoRA 保持不变，便于归因）
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
MODEL=$HOME/models/Qwen3.5-9B \
DATA_DIR=$PWD/data/nova_sft_v2 \
OUTPUT_DIR=$PWD/output/qwen_sft_v2 \
MAX_LENGTH=1024 \
SFT_LR=5e-5 SFT_EPOCHS=3 LORA_RANK=16 FREEZE_ALIGNER=false \
SFT_BATCH_SIZE=1 SFT_GRAD_ACCUM=2 EVAL_STEPS=25 SAVE_STEPS=25 \
  bash scripts/train.sh sft
```

> 本次实际运行因 GPU 1 被其他用户占用，使用 `CUDA_VISIBLE_DEVICES=0,2,3`
> 和 `NPROC_PER_NODE=3`，`SFT_GRAD_ACCUM=3`（有效 batch 9，4 卡方案是 8），
> 其余超参不变。3 卡约 3–4 小时。

- `--drop_train_panel_captions` 只处理同一个 case 里既有 panel 又有普通图的
  情况；本数据集没有这种 case，所以命令里没有启用。`--drop_panel_captions`
  是全局删除，会改变 split，不要拿它做对比实验。
- `--clean_final` 只改 final 字段；test 里有 2 张图的 gold 被清理，重训模型和
  旧模型都应在 `data/nova_sft_v2/test.jsonl` 上重新评一遍再对比。
- `train_aux_boxes.jsonl` 只进 SFT；`run_sft.sh` 检测到文件存在会自动把
  `train.jsonl` 和它一起传给 `--dataset`。GRPO 继续用 `grpo_train.jsonl`。
- 审计新增 A17 专门校验辅助定位样本（只有 `【异常定位】`，无 caption/诊断）。
- 若显存/时间允许，可把 `SFT_LR` 降到 `3e-5` 作为更保守的变体，但不要和
  数据改动同时改，否则无法归因。
- v2 每 epoch 666 行（533 full + 133 aux），3 epoch 约 250 optimizer steps
  （有效 batch 8），比旧 run 的 200 步多 25%；按旧 run 速度约 2.5 小时。

重训后仍按上面的流程在 val 上拟合 scale；checkpoint 默认取 final（val 只有
35 个 case，无法可靠区分晚期 checkpoint，不要用 val 早停），再评 test。

> 实测结果（2026-11-18）：v2 ck249（aux 0.25，`output/qwen_sft_v2/v1-20261118-185243/`）
> 的 val raw F1@0.5 **0.4815**（旧 ck201 0.3746，Δ+0.107 显著），但 test
> calibrated **0.3919** vs ck201 **0.4069**（Δ−0.015，CI 跨 0），统计打平。
> 小框面积比 1.470 → 1.072、斜率 0.578 → 0.665，但大框仍 0.532。
> v3 ck219（aux 0.10，`output/qwen_sft_v3/v0-20261118-194052/`）test calibrated
> **0.3949**，定位仍与 ck201 打平；诊断 T5 keyword **0.2342** 为所有模型最高，
> 但只是 111 图上的小样本差异。**两版数据修复都没有兑现 test 定位增益，
> 主模型仍是 ck201**；详见 `result/box_calibration.md` 第 6.1 节。

## Qwen3.5-9B baseline

服务需先以多模态方式启动（**不能带 `--language-model-only`**）。本机已验证的
启动方式（vLLM 0.30 / CUDA 12.9）：

```bash
VENV=$VLLM_VENV
CUDA129=$CUDA129

CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
CUDA_HOME="$CUDA129" PATH="$VENV/bin:$CUDA129/bin:$PATH" \
"$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model $HOME/models/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --max-model-len 4096 --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 --enforce-eager \
  --limit-mm-per-prompt '{"image": 1}' \
  --reasoning-parser qwen3
```

注意：

- `PATH` 必须包含 venv 的 `bin`（vLLM JIT 需要 `ninja`），并优先使用 CUDA 12.9
  的 `nvcc`；系统 `/usr/local/cuda` 是 11.6，会让 flashinfer 编译失败；
- `VLLM_USE_FLASHINFER_SAMPLER=0` 规避 flashinfer `TopKMaskLogits` 的
  `device kernel image is invalid`；
- vLLM 0.30 的 `--limit-mm-per-prompt` 要传 JSON；
- 本机 GPU 1–3 可能被其他用户占用，单卡 TP=1 + 4096 上下文即可跑 9B。

然后：

```bash
# 图像 + 文本（Qwen3.5 默认 thinking 会吃满 512 token、content 为空，
# baseline 需要完整三段式时用 NO_THINKING=1，并给足输出预算）
NO_THINKING=1 MAX_TOKENS=1024 bash scripts/run_baseline_qwen.sh image

# 纯文本对照（同一 prompt，不传图）
NO_THINKING=1 MAX_TOKENS=1024 bash scripts/run_baseline_qwen.sh text

# 格式引导版：system prompt 强制三段式、正文英文、0–480 像素坐标，诊断才可比
OUT=$PWD/result/qwen35_baseline_fmt/pred.jsonl \
NO_THINKING=1 MAX_TOKENS=512 FORMAT_HINT=1 \
  bash scripts/run_baseline_qwen.sh image

# 断点续跑（会话中断后从 --out 已有 index 之后继续）
OUT=$PWD/result/qwen35_baseline_fmt/pred.jsonl \
NO_THINKING=1 MAX_TOKENS=512 FORMAT_HINT=1 RESUME=1 \
  bash scripts/run_baseline_qwen.sh image

# 评测
python3 src/eval_nova.py \
  --pred result/qwen35_baseline/pred.jsonl \
  --gold data/nova_sft/test.jsonl \
  --out result/qwen35_baseline/report.json
```

`src/infer_qwen_baseline.py` 只用标准库；输出与 `swift infer` 的
`response` / `labels` / `messages` 字段对齐。`--text_only` 的定位指标没有意义，
只能和图像模型比诊断字段。纯文本无图时输出近常量先验框，`eval_nova` 的定位
分数再高也不可比。

四种运行的对比与结论见 `result/baseline_comparison.md`。

## 监控面板

标准库只读面板。推荐用 SSH 隧道：面板会显示训练日志和最近预测，可能含临床
文本，不建议直接暴露到公网。

服务器上启动（默认只绑定 `127.0.0.1:8777`）：

```bash
bash scripts/start_monitor.sh start
bash scripts/start_monitor.sh status
```

你电脑上建立隧道：

```bash
ssh -N -L 8777:127.0.0.1:8777 yanfaxiang@10.120.16.27
# 浏览器打开 http://127.0.0.1:8777/
```

如果确实需要局域网直接访问：

```bash
MONITOR_HOST=0.0.0.0 bash scripts/start_monitor.sh restart
# 浏览器打开 http://<server-ip>:8777/；需要在防火墙/安全组放行 8777
```

常用命令：

```bash
bash scripts/start_monitor.sh restart
bash scripts/start_monitor.sh stop
curl --noproxy '*' http://127.0.0.1:8777/healthz
```

如果浏览器配置了 HTTP 代理，把 `127.0.0.1` 和服务器 IP 加入 no_proxy；否则
浏览器可能把请求发给代理而打不开。SSH 隧道方式不受代理影响。

显示：4 条 Qwen baseline（自由格式 / 格式引导 × 图像 / 纯文本）的进度与
失败/空输出、平均延迟与 tok/s、最近输出、vLLM 健康状态、4 张 GPU 的
显存/利用率、`output/**/logging.jsonl` 里最近的训练 loss/eval、以及日志尾部。
每 5 秒自动刷新。

`python3 scripts/monitor_dashboard.py --once` 打印一份 JSON 快照，便于脚本检查。

## 数据审计

训练前先跑：

```bash
python3 src/audit_data.py --data_dir data/nova_sft
```

审计失败时退出码为 1，不要启动训练。正式训练前建议加 tokenizer 做精确 token 预算：

```bash
python3 src/audit_data.py \
  --data_dir data/nova_sft \
  --tokenizer /path/to/llava-1.5-7b-hf
```

## 重新生成数据

```bash
python3 src/build_dataset.py \
  --nova_root Nova \
  --image_root Nova/images \
  --out_dir data/nova_sft \
  --seed 42
```

训练机路径不同时：

```bash
python3 src/build_dataset.py \
  --nova_root /mnt/workspace/final_proj/Nova \
  --image_root /mnt/workspace/final_proj/Nova/images \
  --out_dir /mnt/workspace/final_proj/data/nova_sft \
  --seed 42
```

> `data/nova_sft/` 被 `.gitignore` 的 `/data/` 忽略，不会自动上传 GitHub。
> 跨机器使用请 `scp` / `rsync`，或在训练机上重新生成。
