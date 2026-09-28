# LitBank 事件与实体抽取 → 出行链研究

本仓库以 Qwen3-1.7B 为基础模型，在 6 GB 显存的 RTX 3060 Laptop GPU 上使用 4-bit QLoRA。当前第一阶段使用 LitBank 英文文学文本，训练模型输出可回溯到原文的事件触发词与实体 JSON；后续阶段才研究人物、地点、交通工具之间的出行链结构及强化学习。

## 当前能力与数据边界

- LitBank 的 `EVENT` 层标注确实发生的事件触发词；实体层提供 `PER`、`FAC`、`GPE`、`LOC`、`VEH`、`ORG`。本项目使用前五类实体，暂不使用 `ORG`。
- 两层标注没有逐事件的实体论元、出发地或目的地关系。输出按**文本块**分别列出事件与实体，不把同块实体误当作某个事件的参与者。
- 数据来自项目根目录的 `litbank/events/brat` 和 `litbank/entities/brat`；本次数据版本的 Git 提交为 `3e50db0ffc033d7ccbb94f4d88f6b99210328ed8`。源数据保留原样；`litbank/`、生成的数据、模型和预测结果不纳入本仓库的 Git 跟踪。
- 原有中文 `trip_chain` Schema、合成样例和训练入口仍保留，用于后续出行链阶段；它们不是 LitBank 金标。

单块输出示意：

```json
{
  "evidence": "24_o_pioneers:c0001",
  "EVENT": ["walked"],
  "PER": ["His sister"],
  "FAC": [], "GPE": [], "LOC": [], "VEH": []
}
```

`evidence` 是稳定的文本块编号。`data/processed/litbank_mentions/*_gold.jsonl` 保存块编号、作品编号、原文以及每个金标的块内字符偏移，用于核对与回映射；模型只输出提及文本，不生成容易出错的数字偏移。相同提及在一块内标注多次时，数组中保留相应的重复次数。上例只说明格式，不代表特定作品的实际金标。

## 数据准备

在 WSL2 Ubuntu 24.04 中运行；环境路径及安装顺序见 [环境说明](docs/environment.md)：

```bash
cd /mnt/c/Users/24171/Desktop/Qwen
source /home/gollie/.venvs/qwen3-trip-chain/bin/activate
export PYTHONPATH=src
python -m trip_chain.data.litbank \
  --litbank-dir litbank \
  --output-dir data/processed/litbank_mentions \
  --tokenizer /home/gollie/models/Qwen3-1.7B-modelscope
```

按**作品**随机划分 80/10/10 部训练、验证、测试作品（种子 42），然后按原文行和安全空白切块；默认不重叠。每个标注完整归属一个块，不能跨块截断。转换器按实际 Qwen tokenizer 同时统计提示词、输入和金标输出，为对话边界预留 48 token，保证预算不超过 832 token。超长行在不切断标注的位置继续拆分。当前生成 827/105/100 条训练、验证、测试样本；完整对话模板下实测最长分别为 718/785/623 token，没有样本超过上限。详细长度分布见 `data/processed/litbank_mentions/stats.json`。

### 切块、证据编号与重叠

当前 `safe_units` 优先沿 LitBank 原文的**换行**划分，换行通常对应句子，但代码没有调用句子识别器，因此不保证每个单元恰好是一句。单行超过 256 个模型 token 时，选择不会穿过任何 BRAT 标注的空白位置拆开。`convert_document` 再把相邻单元装入一个块：正文最多 256 token，提示词＋正文＋完整金标输出＋48 token 预留必须不超过 832。不能容纳的单元留给下一块；单个单元仍超预算时直接报错，不截断标注。`evidence` 用作品名加连续块号，例如 `24_o_pioneers:c0001`；金标文件另存该块在作品中的 `start/end`。

**本次训练与测试没有重叠。** 每个原文位置只进入一个块，每个事件或实体标注也只由该块负责输出。重叠的潜在价值是为块首提供前文，例如第 3 句说明人物是 Mary，第 4 句只写 “She walked home”。若把第 3 句附在后一个块，模型能用它理解 “She”，但后一个块仍只能输出第 4 句起的标注：

```text
c001：负责第 1–3 句；输出第 1–3 句标注
c002：只读第 3 句作为上下文；负责第 4–6 句；只输出第 4–6 句标注
```

这需要在数据结构中区分“只读上下文”与“负责抽取的正文”，让重叠句的标注始终归原块，并在合并结果时按原文位置去重。上下文同样占用 256 个输入 token 和 832 个总 token 的预算，加入前文就要缩小该块负责抽取的正文。**现有代码尚未实现重叠或上下文归属规则**；只有在单独分析块首错误、确认缺少前文确实造成漏抽时，才适合实现并与本次无重叠基线比较。

### 本次样本长度实测

下表是 Qwen tokenizer 统计的训练集 token 数；“提示词”包括系统指令与块编号等包装文本，不含正文和聊天特殊 token。“总量上界”另外预留 48 token 给聊天模板。它们是长度预算，不是显存用量。

| 部分 | 中位数 | 最大值 |
|---|---:|---:|
| 提示词 | 134 | 149 |
| 输入正文 | 239 | 256 |
| 金标 JSON 输出 | 121 | 319 |
| 含预留的总量上界 | 541 | 747 |

验证集输出最长为 383 token，含预留总量上界最高为 814。用完整 Qwen 对话模板复核时，训练、验证、测试样本的实际最大长度分别为 718、785、623，均低于训练上限 832。

## 训练参数与选择依据

配置分别位于 `configs/model/qwen3_1_7b.yaml` 和 `configs/training/litbank_6gb.yaml`。这里的“标定”是依据 **6 GB 显存上限、真实样本长度与一次显存试跑**确定可运行的起点；本轮没有做学习率、LoRA 秩或训练轮数的网格搜索，不能把这些数值解释为最优参数。

| 参数 | 本次设置 | 选择依据与作用 |
|---|---|---|
| 基础模型 | Qwen3-1.7B | 沿用仓库既有模型；训练命令使用 WSL 中的本地权重目录。 |
| 量化 | 4-bit NF4、double quant | 压缩冻结的基础模型权重，以适配 6 GB 显存；只训练 LoRA 适配器。 |
| 计算精度 | BF16；允许 TF32 | 本机支持 BF16，训练入口会在不支持时退回 FP16；TF32 用于受支持的矩阵运算。 |
| LoRA | `r=16`、`alpha=32`、`dropout=0.05`，覆盖注意力和 MLP 投影 | 沿用项目已有可运行的 QLoRA 起点；本轮未比较其他秩、缩放或模块组合。 |
| 批量与梯度累积 | 单卡 batch 1、累积 16 | 单步只放 1 条样本以控制峰值显存；约每 16 条更新一次参数，827 条训练样本形成 52 个优化步骤。 |
| 梯度检查点与优化器 | 开启 gradient checkpointing；`paged_adamw_8bit` | 用额外计算换取更低的激活与优化器状态显存。 |
| 长度 | 正文最多 256 token；训练 `max_length=832` | 含预留的真实样本最长 814，完整模板实测最长 785；832 留下空间且避免截断答案。 |
| 学习率与调度 | `2e-4`、cosine、前 3% 步数预热 | 一轮训练的起始配置；约 52 个优化步骤，预热比例约等于前 2 步。未做学习率搜索。 |
| 轮数与检查点 | 1 轮；每 50 步验证和保存，最多保留 2 个检查点 | 先取得可复现基线；训练结束再验证并保存最终 adapter。单轮结果不能证明已充分收敛。 |
| 损失 | `completion_only_loss=True` | 只对助手 JSON 答案计算训练损失；输入正文和提示词只提供条件。 |
| 生成 | 贪心解码，`max_new_tokens=384` | 输出确定，便于复现；测试金标输出最长 226 token，但模型仍可能重复生成直到上限。 |

在更保守的 **1024-token、batch 1 单步**试跑中，峰值分配显存为 4566.35 MiB、峰值预留 5052 MiB；这只是单步承载检查。完整 832-token 配置训练的实际峰值分配为 4005.4 MiB。显存数字不是抽取准确率，也不保证其他后台 GPU 负载下仍有相同余量。

## 微调与评估

单步 1024 token 显存试跑：

```bash
python tmp/benchmarks/benchmark_qlora_length.py \
  --length 1024 --model /home/gollie/models/Qwen3-1.7B-modelscope
```

LitBank 一轮 QLoRA 微调：

```bash
python -m trip_chain.training.train_qlora \
  --model-name-or-path /home/gollie/models/Qwen3-1.7B-modelscope \
  --training-config configs/training/litbank_6gb.yaml \
  --train-file data/processed/litbank_mentions/train.jsonl \
  --validation-file data/processed/litbank_mentions/validation.jsonl
```

在锁定的测试作品上贪心生成并评估：

```bash
python -m trip_chain.evaluation.litbank \
  --gold data/processed/litbank_mentions/test_gold.jsonl \
  --adapter outputs/checkpoints/qwen3-1.7b-litbank-mentions-qlora
```

评估器要求合法 JSON、正确的 `evidence`、固定字段，而且每个提及文本必须在对应原文块中出现；逐标签按“提及文本及其重复次数”计算精确率、召回率、F1，并报告整体微平均、合法输出比例和整块完全正确比例。另报诊断性提及 F1，它会忽略块编号和文本落地错误，用来分辨格式失败与识别失败；正式的**原始模型结果**以严格指标为准。预测和指标分别写入 `outputs/predictions/litbank_mentions_test.jsonl` 与 `outputs/metrics/litbank_mentions_test.json`。

同时提供确定性的文本落地修正：从输入块复制 `evidence`，过滤不在原文中的提及及超过原文出现次数的重复提及；若 JSON 在输出上限处截断，只恢复其中已经完整生成的字符串。修正后的结果和指标保存在 `outputs/predictions/litbank_mentions_test_grounded.jsonl`、`outputs/metrics/litbank_mentions_test_grounded.json`，需与原始结果分开报告。模型输出没有字符偏移，因此上述指标不衡量同词多次出现时的精确位置。

## 仓库目录用途

| 目录 | 用途与当前状态 |
|---|---|
| `litbank/` | 本地 LitBank 源数据仓库；本轮读取 `events/brat/` 和 `entities/brat/`，不修改原始标注，也不跟踪到本项目 Git。 |
| `configs/model/`、`configs/training/` | 基础模型与训练配置；`litbank_6gb.yaml` 是本轮实际参数。 |
| `configs/data/`、`configs/inference/` | 原出行链的数据路径与推理配置；LitBank 转换和评估当前由各自 CLI 参数控制。 |
| `data/raw/` | 预留其他原始数据；本轮 LitBank 实际放在根目录 `litbank/`。 |
| `data/interim/` | 预留清洗或人工标注的中间数据。 |
| `data/processed/` | 可直接训练和评估的数据；`litbank_mentions/` 是本轮紧凑 JSON 样本及金标，`litbank/` 保存早期数字偏移试验的数据，根层 JSONL 是原出行链合成样例。 |
| `data/samples/` | 原出行链格式的少量合成示例，不是 LitBank 金标。 |
| `src/trip_chain/data/` | BRAT 解析、切块、拆分与数据格式转换；也保留原出行链样例准备代码。 |
| `src/trip_chain/training/` | 通用 QLoRA 训练入口。 |
| `src/trip_chain/evaluation/` | LitBank 生成与计分，以及原出行链指标。 |
| `src/trip_chain/inference/`、`src/trip_chain/utils/` | 为后续独立推理接口和共用工具预留，目前没有正式实现。 |
| `schemas/` | 原出行链 JSON Schema；本轮 LitBank 输出由评估代码验证。 |
| `requirements/` | 环境安装清单及 Windows/WSL 锁定版本。 |
| `models/base/`、`models/adapters/`、`models/merged/` | 预留统一管理基础权重、适配器与合并权重；本轮实际基础权重在 WSL 本地模型目录，adapter 在 `outputs/checkpoints/`。 |
| `outputs/checkpoints/` | 本轮 LoRA adapter 与训练中途检查点；`outputs/logs/` 为运行日志，`outputs/predictions/` 为逐块预测，`outputs/metrics/` 为训练与评估指标。 |
| `tests/` | 数据 Schema、切块样例与计分逻辑的自动化测试。 |
| `tmp/benchmarks/` | 显存试跑脚本，属于实验辅助代码。 |
| `docs/` | 环境、原出行链标注规范与历史任务记录。 |
| `notebooks/` | 预留探索分析；`output/pdf/` 是历史输出占位目录，本轮 LitBank 流程不使用。 |
| `.venv/`、`.pytest_cache/`、`.ruff_cache/` | 本地环境与工具缓存，可重新生成，不属于项目源代码。 |

`outputs/` 和 `data/processed/` 的实验产物默认不提交到 Git。若在另一台机器复现，需要先取得 LitBank 数据与 Qwen 基础模型，再按上面的命令重新生成样本、训练 adapter 和计算指标。

### 关键代码定义与调用顺序

| 定义 | 所在文件 | 实际作用 |
|---|---|---|
| `LABELS`、`SYSTEM_PROMPT`、`Span` | `data/litbank.py` | 固定六个输出键、任务指令和带原文字符偏移的内部标注对象。`Span` 的偏移留在金标，不要求模型生成。 |
| `read_spans` | `data/litbank.py` | 读取 BRAT `.ann` 文本标注，核对 `.txt[start:end]` 与标注词面一致；跳过注释，遇到不匹配或不支持的断续跨度就报错。 |
| `safe_units` | `data/litbank.py` | 优先取原文行；超长行在不穿过标注的空白处拆开。 |
| `sample`、`token_counts` | `data/litbank.py` | 为块生成 `evidence`、六类标签数组、内部偏移映射及 prompt/completion；分别计数提示、正文、输出，并为对话模板预留 48 token。 |
| `convert_document`、`main` | `data/litbank.py` | 把安全单元按双重长度预算打包；先按作品分组 80/10/10，再写训练 JSONL、独立金标 JSONL 和 `stats.json`。 |
| `train` | `training/train_qlora.py` | 加载 4-bit 基础模型、挂接 LoRA，使用 `SFTTrainer` 仅对 completion 训练，记录耗时、损失和峰值显存。 |
| `generate` | `evaluation/litbank.py` | 按训练提示词对测试块贪心生成，把每条原始预测立即写盘。 |
| `valid_mentions`、`score` | `evaluation/litbank.py` | 检查字段、块编号、提及是否出现在原文及重复次数；按类别累计 TP/FP/FN，再计算 Precision/Recall/F1。 |
| `ground_prediction` | `evaluation/litbank.py` | 对原始预测做确定性落地：补入已知块编号、过滤不存在或超额的提及，并从截断 JSON 中恢复已经完整生成的字符串；它不读取测试答案。 |

核心链路是 `read_spans → safe_units → sample/convert_document → train → generate → valid_mentions/score`；`ground_prediction` 是与原始模型评估分开报告的后处理支路。原出行链的 `data/prepare.py`、`evaluation/metrics.py` 和 `schemas/trip-chain.schema.json` 服务另一项任务，不能拿来校验 LitBank 标签。

运行代码检查：

```bash
ruff check src tests
pytest -q
```

## 实验结果

1024 token、batch size 1 的单步 QLoRA 显存试跑已通过：峰值分配 4566.35 MiB，峰值预留 5052 MiB。首次尝试让模型生成字符偏移时，一轮训练损失为 0.5612、验证损失为 0.4247，但三条测试探针的严格抽取 F1 为 0，显示数字偏移和过长输出是明显问题；该尝试已改为上述紧凑格式。

紧凑格式完成了 **1 轮、827 条训练样本**的 QLoRA 微调，耗时约 12 分 36 秒，训练损失 0.6631，最终验证损失 0.3505，训练峰值分配显存 4005.4 MiB。锁定的 10 部测试作品共 100 个文本块、1947 个事件或实体提及，全部完成生成与评估：

| 标签 | 测试金标数 | 原始输出 F1 | 文本落地修正后 F1 |
|---|---:|---:|---:|
| EVENT | 831 | 0.1205 | 0.4830 |
| PER | 719 | 0.1911 | 0.6051 |
| FAC | 199 | 0.0841 | 0.3415 |
| GPE | 52 | 0 | 0.1493 |
| LOC | 126 | 0 | 0.1860 |
| VEH | 20 | 0 | 0.0909 |
| **微平均** | **1947** | **0.1327** | **0.4919** |

原始预测的 JSON 语法可解析率为 69%，同时满足字段、块编号和原文落地要求的比例为 15%，整块完全正确率为 0%。若忽略格式与落地错误，只比较可解析输出中的提及文本，诊断性 F1 为 0.4080。修正后的合法输出率为 100%，这是确定性后处理保证的，**不代表模型本身有 100% 的格式准确率**；修正后精确率 0.5082、召回率 0.4766、微平均 F1 0.4919，整块完全正确率仍为 0%。地点、设施及车辆类别的召回率尤其低，当前结果只适合作为事件与实体抽取基线，尚不足以直接用来可靠生成出行链。`VEH` 测试金标仅 20 个，单类估计不稳定。

### 指标如何计算、为何有差距

- 每个标签按“**类别＋精确提及文本＋重复次数**”匹配金标。匹配上的数量是 TP，多报是 FP，漏报是 FN；`Precision=TP/(TP+FP)`，`Recall=TP/(TP+FN)`，`F1=2PR/(P+R)`。微平均先汇总六类的 TP/FP/FN，再算一组 P/R/F1；它会更受 `EVENT` 和 `PER` 等高频类别影响。
- `json_parse_rate=69%` 只要求能解析 JSON；`valid_json_rate=15%` 还要求固定字段、正确块编号，以及每个词面在输入中出现且重复次数不超过原文。**原始严格计分把不合规的整个文本块视为空预测**，所以虽然原始严格精确率有 0.6092，召回率只有 0.0745，F1 只有 0.1327。常见失败是重复输出词面、输出原文中没有的词，或生成到 384-token 上限时 JSON 尚未闭合。
- `diagnostic_mention_micro` 的 F1 为 0.4080：它只看能解析的输出中的提及文本，不检查 `evidence` 和原文落地，用于判断“词有没有抽到”。它不能代替原始严格分数。
- 确定性修正从**输入块**取编号与原文，裁掉不在原文或超过出现次数的提及，并从截断输出恢复已完整生成的字符串；因此得到 100% 合规输出与 0.4919 的微平均 F1。修正会丢掉错误提及，也不能补出模型从未生成的金标；其精确率 0.5082、召回率 0.4766，说明误报和漏报都仍很多。原始与修正后整块完全正确率均为 0%。
- 事件 F1 0.4830、人物 F1 0.6051 是修正后的最好两类；`FAC/GPE/LOC/VEH` 分别只有 0.3415/0.1493/0.1860/0.0909。测试集中 GPE 52 条、VEH 20 条，单类分数对少量样本较敏感。地点类召回低，尚不能把输出当作可靠的出发地或目的地信息。
- `train_loss=0.6631` 与 `eval_loss=0.3505` 是在给定正确前文的情况下预测训练答案 token 的损失，不等于自由生成时的 JSON 合规率或抽取 F1。测试按完整答案自行生成，重复续写和错误标签会真实计入结果。

当前金标保留字符偏移，但模型输出只含词面。评估器用 `text.count` 检查词面出现次数，**不能判定同一个词在块内多次出现时究竟指哪一处**；也没有事件到人物、地点的金标关联。因此这些指标衡量的是块级触发词与实体提及抽取，不衡量事件角色、起终点或出行链顺序。

训练、原始预测和修正后指标分别见 `outputs/metrics/litbank_mentions_train_metrics.json`、`outputs/metrics/litbank_mentions_test.json`、`outputs/metrics/litbank_mentions_test_grounded.json`；原始输出可在 `outputs/predictions/litbank_mentions_test.jsonl` 逐条检查。

LitBank 数据集及标签定义见 [官方仓库](https://github.com/dbamman/litbank)。
