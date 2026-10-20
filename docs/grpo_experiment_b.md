# 实验 B：六候选串行生成，观察训练奖励

入口：`train_grpo_experiment_b.py`。配置：`configs/training/grpo_experiment_b.yaml`。跳过 SFT，从基座创建新 LoRA。全部数据用于训练，不划验证集，不执行验证或测试评估。

## 与历史 A 的关系

| 项目 | 历史 A | B 正式配置 |
|---|---:|---:|
| 数据 | 14 条人工审核 + 20 条合成标注 | 同一份 34 条冻结快照 |
| 每题候选数 | 2 | 6 |
| 同时生成数 | 2 | 1，可改为 2 |
| 训练微批 | 1 | 1 |
| 梯度累积 | 4 | 12 |
| 每次更新对应原文数 | 2 | 2 |
| 一轮优化步数／候选数 | 17／68 | 17／204 |
| 原文／输出上限 | 3072／1024 token | 相同 |
| 六项奖励权重／KL beta | 全为 1／0.04 | 相同 |

同样保持基座、LoRA、seed=42、采样温度、top-p、学习率和调度器。B 的计算成本更高；固定更新规模不等于固定耗时。历史 A 数据和结果不会改写。

## 运行

在已配置的 WSL Python 环境中：

```bash
cd /mnt/c/Users/24171/Desktop/研/model/Qwen
source /home/gollie/.venvs/qwen3-trip-chain/bin/activate

# CPU 校验全部 34 条数据。
python train_grpo_experiment_b.py --prepare-only

# 真实分词器预算检查，不加载训练模型。
python train_grpo_experiment_b.py --dry-run

# GPU 短程试跑，仍为六候选，每次一条；保留正式 1024-token 上限。
# 使用完整小正例与负例，共 12 个候选、1 个优化步。
python train_grpo_experiment_b.py --smoke-test --max-completion-length 1024

# 正式一轮：全部 34 条，训练结束自动绘制 reward 对照。
python train_grpo_experiment_b.py
```

默认创建带北京时间戳的新目录：`outputs/experiments/grpo_b_g6_serial_full_.../`；试跑使用 `..._smoke_.../`。可用 `--output-dir` 指定独立路径。`--max-steps 2` 可运行全数据采样顺序中的前两步；它不是完整一轮。

`--smoke-test` 明确使用小样本，并将 warmup 设为 0，使单步可以更新参数。试跑曲线标记 SMOKE / SUBSET，不能与 A 推断效果差异。未显式指定输出长度时 smoke 上限为 512；上方命令明确设为 1024。

恢复完整 B 检查点：

```bash
python train_grpo_experiment_b.py --resume-from-checkpoint outputs/experiments/<B运行目录>/checkpoint-2
```

使用原运行的配置，保留微批设置。最终 adapter 不能代替完整恢复检查点，也不能作为本脚本基座路径。

## 串行生成的边界

`rollout.micro_batch_size: 1` 是项目新增选项，不传给 GRPOConfig；逻辑 `generation_batch_size` 仍为 6。仅在 TRL 普通 Transformers 文本生成调用内部切批，原序合并完整 token IDs，再交给 TRL 统一评分和反传。

不在候选之间更新权重，不重置种子，不去重，不补采合法答案，不修复 JSON。收齐六个候选后才计算组内均值和标准差。六项奖励、KL、概率比率、裁剪、掩码、损失归一化均保留原 TRL 路径。现有字符串 `"null"`、同义表达等奖励问题不在本实验中修改。

此选项仅支持 TRL 0.29.1 的普通 Transformers 文本生成，不混用 vLLM、paged generation 或多模态输入。它主要减少同时活动的响应 KV cache；已生成 token 仍需保留用于训练，不能保证总显存与 G 完全无关。

## 日志与曲线

- `resolved_config.json`、`run_manifest.json`：实际配置、依赖、数据签名与微批设置。
- `data/train_gold.jsonl`：本次数据副本。
- `rollout_microbatches.jsonl`：实际活动序列数、逻辑批量、切片顺序、耗时和生成长度；串行时 `active_sequences` 应始终为 1。
- `reward_trace.jsonl`：每个实际候选与六项奖励。
- `progress.jsonl`：逐优化步 reward、分项、零奖励方差组比例、KL、梯度等。
- `figures/reward_curve.png`：B 自身原始 reward 和尾随 3 步均值。
- `figures/reward_comparison.png`：历史 A 和 B 的 reward、零奖励方差组比例对照。
- `figures/baseline_a_rewards.csv`、`experiment_b_rewards.csv`：实际日志数值。
- `figures/comparison_summary.json`：前后 reward 均值及数据是否一致等。
- 最终 LoRA、训练指标、状态及达到保存间隔时的恢复检查点。

训练中或训练后重绘，无需加载模型：

```bash
python plot_grpo_reward_comparison.py --run-dir outputs/experiments/<B运行目录>
```

曲线不补造或插值；移动平均使用当前及之前最多两步。当前默认一轮，后续可另存配置增加轮数观察趋势。不同步使用不同原文，reward 波动也受样本难度影响。

## 自动检查

```bash
python -m pytest tests/test_grpo.py tests/test_grpo_microbatch.py -q
python -m ruff check src/trip_chain/training tests/test_grpo_microbatch.py train_grpo_experiment_b.py plot_grpo_reward_comparison.py
```

检查包含六候选不会被 smoke 改回二候选；切批保留数量、顺序和随机数推进；正式配置只改变候选数、逻辑生成批和梯度累积；数据签名与 A 一致；绘图来自真实日志。真实 GPU 试跑证据另存于运行目录。

## 2026-10-20 实测记录

30 项相关自动检查和代码检查通过。全部 34 条预算预检通过，最长 prompt 为 1225 token，最长金标输出为 808 token。

真实 GPU 试跑目录：`outputs/experiments/grpo_b_g6_serial_smoke_20261020_215203/`。保留 1024-token 上限，使用 ch03_007 正例和 ch08_007 负例；每条各 6 个候选，共 12 次串行调用，活动序列数始终为 1，逻辑组大小始终为 6。完成 1 个优化步，耗时 813.73 秒，CUDA 峰值分配显存 5403.26MiB，梯度范数 0.1328125，196/196 个 LoRA B 矩阵更新。单步平均 reward 为 -0.465705，奖励日志和对照图已导出，核验见该目录的 `smoke_verification.json`。

这证明加载、六候选串行生成、完整组评分、反传、保存和绘图可执行。它只覆盖两条完整小样本，不能证明所有长样本均不溢出，也不能从单步判断 reward 上升；全量正式 B 尚未启动。
