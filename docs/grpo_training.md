# Qwen3-1.7B 直接 GRPO 强化学习

本入口从 Qwen3-1.7B 基础模型创建新的 LoRA 参数，直接进入 GRPO，**跳过 SFT**，不加载原先的 LitBank 适配器。数据使用 GLiNER2 的人物、活动节点、出行边三层标注。策略目标、组内优势、概率比率裁剪和 KL 全部交给 **TRL 0.29.1 的 GRPOTrainer**，本项目只负责数据适配和六项任务奖励。

## 运行方法

在 WSL 的现有训练环境中运行。将下面的 `<项目目录>` 替换为 `C:\Users\24171\Desktop\研\model\Qwen` 对应的 WSL 路径 `/mnt/c/Users/24171/Desktop/研/model/Qwen`。

```bash
cd /mnt/c/Users/24171/Desktop/研/model/Qwen
source /home/gollie/.venvs/qwen3-trip-chain/bin/activate
```

现有环境已安装 TRL 0.29.1。需要重新安装依赖时，先安装项目 CUDA PyTorch 清单，再安装 GRPO 清单：

```bash
python -m pip install -r requirements/torch-cu128.txt
python -m pip install -r requirements/grpo.txt
```

建议依次执行：

```bash
# 只校验标注并生成划分，不需要 GPU、模型或分词器。
python train_grpo.py --prepare-only

# 用本地分词器核对每条完整输入和金标输出的 token 预算，不训练。
python train_grpo.py --dry-run

# 选取完整的小样本，执行一个优化步骤并保存 smoke 适配器。
python train_grpo.py --smoke-test

# 按默认配置直接做 GRPO，不经过 SFT。
python train_grpo.py
```

可从任意目录用入口的绝对路径启动。所有配置中的相对路径都相对于 Qwen 项目目录解析。模型默认路径是现有 WSL 权重 `/home/gollie/models/Qwen3-1.7B-modelscope`，可覆盖：

```bash
python train_grpo.py --model-name-or-path /your/base-model/path
```

显式加入 20 条合成样本：

```bash
python train_grpo.py --include-synthetic --output-dir outputs/checkpoints/chain-grpo-with-synthetic
```

默认只使用人工已审核数据。自定义输入文件用 `--annotations file1.jsonl file2.jsonl`，合成数据仍需要 `--include-synthetic`。默认合成文件也会随该开关加入；要完全替换它，可在配置中将 `data.synthetic_file` 设为 `null`。

恢复本脚本保存的完整 GRPO 检查点：

```bash
python train_grpo.py --resume-from-checkpoint outputs/checkpoints/qwen3-1.7b-chain-grpo/checkpoint-10
```

默认小数据运行可能不到 10 步，只保存最终 adapter，不产生中途检查点。需要检查点时，在 YAML 中减小 `save_steps`。恢复训练需要检查点中的优化器、调度器、随机状态及 `trainer_state.json`，还需要检查点所在运行目录的 `run_manifest.json`。仅有最终 adapter 不能完整恢复优化器状态。恢复时保持模型、数据划分、LoRA、奖励及主要算法参数一致，使用原运行的配置与 `--include-synthetic` 开关；可通过 `--max-steps` 增加总步数。恢复检查点会读取该次 GRPO 的适配器，不会引入 SFT。

## 输入与输出

默认读取 `../../GLiNER2/annotation/人工已审核/approved_annotations.jsonl`，严格要求 `status: human_approved`。显式允许的合成数据要求 `synthetic: true`（或相应来源标记）及 `status: synthetic_labeled`。机器预标注和待复核记录不会自动进入训练。

输入每行沿用现有 `chunk_id`、`text`、`chains`。`chains` 中保留 `person`、`nodes`、`edges`，数据副本保留原始 `evidence: {text,start,end}`。字符区间使用 Unicode 字符索引，左闭右开，必须与原文一致。活动名称可沿用已审核的概括性表达，奖励不会要求所有语义字段都等于原文子串。

策略只接收系统抽取规则和原始 `text`，金标通过独立 `gold_json` 列传给奖励函数，不拼入 prompt。Qwen 聊天模板设为 `enable_thinking=False`。

模型生成核心字段，省略证据段落和字符偏移以减少输出长度：

```json
{"chains":[{"person":{"person_name":"Marta","identity_occupation":null,"age":null},"nodes":[{"activity_name":"wash cloth","activity_place":"river","activity_start_time":null,"activity_duration":null,"sequence":1}],"edges":[]}]}
```

缺失语义字段必须是 JSON `null`。无人物时输出 `{"chains":[]}`。只有人物属性时保留该人物，`nodes` 和 `edges` 为空。

内部活动节点从 1 连续编号，N 个节点的内部边为 1→2、2→3 等。**兼容现有金标中的边界出行**：原文只描述前往第一个活动地点、起点未知时，可保留 `from_sequence:null,to_sequence:1`；末端明确出行但无下一个活动节点时允许 `from_sequence:N,to_sequence:null`。不会为未知端点补造节点。若模型额外生成 evidence，则仍须严格核对其原文区间。

重复原文通过文本 SHA-256 去重，重复文本的标注冲突会报错。同来源章节、同名人物关联的章节整体分组，再按固定种子划分训练/验证，降低重叠切块和同名人物跨集合泄漏。验证比例按组数计算，并在可行时保留至少一组含节点的样本到两侧。这只能避免同名字符串泄漏，不能识别别名或所有跨章节指代。暂不额外划出测试集。

不截断原文或金标。预检发现超过输入/输出预算时直接报错，应调整预算，或另行提供完整重新标注的短样本。

## 六项奖励

六项权重在 `configs/training/grpo_6gb.yaml` 中分别设置，默认全为 **1.0**，按表中顺序相加，再由 TRL 按同一 prompt 的候选组进行优势标准化。

| 奖励 | 计算口径 | 范围 |
|---|---|---|
| node | 人物存在、非空人物属性、节点存在、活动与地点的事实级 F1 | [0,1] |
| edge | 带人物及节点归属的边存在、出行方式的事实级 F1 | [0,1] |
| time | 节点开始时间、活动持续时间、边出行耗时的事实级 F1 | [0,1] |
| chain | 人物/节点覆盖及节点先后关系 F1，乘以预测链结构合法比例 | [0,1] |
| format | 严格 JSON 和字段类型合规；若额外给出证据，也检查其区间 | 0 或 1 |
| unsupported | 错误或无依据的预测事实占全部预测事实的比例，取负值 | [-1,0] |

人物先做一对一最大权重匹配，再在人物内部一对一匹配节点。不同明确姓名不能互相匹配。匿名人物可由身份和活动内容对齐。节点由活动、地点、时间等已标属性对齐。事实比较只忽略大小写和连续空白差异，不用另一个大模型判断同义。边使用对齐后的实际节点身份计算，因此倒序、错误连边、跨人物挪用和重复预测会扣分。

非空事实按多重集匹配，重复项会产生额外误报。`null` 本身不累计成正确事实，避免大量空值抬高奖励；预测把金标 `null` 填成非空会产生误报和 unsupported 负分。全空链正确时 node、chain、format 分别为 1，其他项为 0。双方均没有边或时间事实时，该项为 0，不额外奖励“没有这类内容”。正例输出空链不能得到 node/edge/time/chain 奖励。

无效 JSON、重复 JSON 键、NaN、缺失字段或字段类型错误：前五项全为 0，unsupported 为 -1。不会修复、截取或从 Markdown 中提取答案再评分。顺序/连接语义错误由 chain/edge 项扣分，JSON 类型正确不代表链正确。

## TRL 算法与显存设置

按 PPT 的序列长度归一化目标显式设置 `loss_type: grpo`，使用 `scale_rewards: group`、`multi_objective_aggregation: sum_then_normalize`、token 级概率比率、`epsilon: 0.2`。旧策略缓存和优势计算均由 TRL 处理。

KL 使用 TRL 的采样估计：令 `d = logp_ref - logp_policy`，逐 token 计算 `exp(d) - d - 1`。默认 `beta: 0.04`，保留 KL 项；`use_bias_correction_kl: false`、`sync_ref_model: false`。beta 独立于六项奖励权重。LoRA 从零更新开始，基础权重冻结，TRL 在关闭适配器时使用同一基础模型作为参考策略，避免额外放置第二份完整模型。

基础权重以 4-bit NF4 加载，训练 LoRA r=8，梯度检查点开启，单卡训练微批量为 1。每个 prompt 默认生成 2 个候选；`generation_batch_size: 2` 指两个候选而不是两个不同 prompt。累计 8 个微批次更新。默认学习率 5e-6，一轮；这些是可修改的起点，不是已证明的最优参数。

训练默认只写验证划分，不在训练中自动评估，以节省显存和生成时间。需要验证奖励时可将 `eval_strategy` 改为 `steps` 并配置 `eval_steps`。默认验证微批量 2、每个 prompt 两个候选，验证显存需另测；验证奖励不是独立的完整抽取准确率报告。

`--smoke-test` 只用最小的完整正例及一个完整负例，候选长度上限设为不超过 512 token，累计两个微批次，运行一步。它检验加载、生成、奖励、反传和保存路径，**不代表默认 1536-token 长度和全部样本的显存已经验证，也不能证明模型效果提升**。

直接从基础模型强化学习时，多个候选可能都无法输出合法结构，得到相同奖励。这时组内优势为零，几乎没有任务学习信号。脚本会记录 `frac_reward_zero_std` 并在接近 1 时提示检查，不会自动改为 SFT。也不会用答案替换候选来制造奖励。

TRL 0.29.1 已移除 `max_prompt_length` 参数，因此本项目在分词预检时自行执行这一限制，再把有效配置传入 GRPOConfig。适配器路径、日志路径均由配置或命令行显式控制。不会上传模型、数据或日志。

## 输出文件

默认目录：`outputs/checkpoints/qwen3-1.7b-chain-grpo/`。smoke 目录另加 `-smoke`。

- `data/train_gold.jsonl`、`validation_gold.jsonl`：分组后的完整金标副本。
- `data/data_summary.json`、`token_lengths.json`：来源、正负例数量及逐条长度。
- `resolved_config.json`：解析后的运行参数。
- `run_manifest.json`：基础模型、依赖版本、KL 口径、运行模式和恢复一致性信息。
- `reward_trace.jsonl`：每次奖励计算的候选原始输出及六项奖励，便于排查格式失败、乱填或零优势。评估调用也会追加；可设置 `save_reward_trace: false` 关闭。
- `adapter_model.safetensors`、`adapter_config.json` 和 tokenizer：最终 GRPO 适配器及分词器。
- `train_results.json`、`trainer_state.json`：训练指标和日志历史。
- `checkpoint-*`：达到保存步数时的完整恢复检查点。

已有最终 adapter 或检查点时，新训练会要求使用另一输出目录或恢复检查点，避免覆盖。只读预检若发现已有运行清单，会把检查产物放到 `preflight/` 子目录。

## 检查

```bash
python -m pytest -q
python -m ruff check src tests train_grpo.py
```

当前人工样本数量很少。单步通过只证明流程能执行；训练奖励上升、格式合规和实际抽取质量需要分别判断。

2026-10-19 验证记录：33 项自动化测试及代码检查通过；人工数据及显式加入合成数据的转换通过；真实分词器长度预检通过。在 RTX 3060 Laptop 6GB 上完成真实 Qwen3-1.7B 单步 GRPO，512-token 候选上限下耗时约 52.6 秒，峰值分配显存 4108.83 MiB。保存后核对 196 个 LoRA B 矩阵均出现非零参数，确认不是仅加载模型或空跑。结果保存在 `outputs/checkpoints/grpo-smoke-verified/`。默认 1536-token 上限及整轮训练未执行，本记录不代表抽取效果评估。

算法与接口依据：[TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)、[TRL 0.29.1 GRPOConfig](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_config.py)、[TRL 0.29.1 GRPOTrainer](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_trainer.py)。
