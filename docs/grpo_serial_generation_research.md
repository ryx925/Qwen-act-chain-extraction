# GRPO 分批／串行生成候选：资料核查与质量影响判断

核查日期：2026-10-20。适用对象：本项目 Qwen3-1.7B、4-bit QLoRA、TRL 0.29.1、单张 RTX 3060 Laptop 6GB。

## 判断

可以采用。固定模型权重、输入、采样参数和候选总数时，在理想数值计算下，将同一题的 G 个独立采样分为若干批执行，不改变目标采样分布。GRPO 要求候选来自同一采样策略，并使用整个候选组计算优势，没有要求 G 个候选必须同时驻留显存。这是从算法定义推导出的结论，不是本机质量对照实验的结果。[DeepSeekMath，第 4.1 节](https://arxiv.org/html/2402.03300v3#S4.SS1)

实际输出不能保证逐字相同，最终训练质量也不能仅凭理论保证。批大小变化会改变随机数的消耗顺序，并可能改变浮点计算结果；数值差异可能在自回归生成中放大。应该验证总体质量分布，而不是要求同一个 seed 下每条文本完全一致。[PyTorch 数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#batched-computations-or-slice-computations)、[随机数与可复现性说明](https://docs.pytorch.org/docs/main/notes/randomness.html)

## 直接相关资料与证据边界

| 资料 | 支持的判断 | 不能据此断言的内容 |
|---|---|---|
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/html/2402.03300v3)，2024，第 4.1 节、Algorithm 1 | 同一旧策略采样 G 个候选，随后评分、计算组内优势，再更新策略 | 不是本项目串行／并行的质量实测 |
| [Infinite Sampling: Efficient and Stable Grouped RL Training for Large Language Models](https://arxiv.org/html/2506.22950v1)，2025，预印本，第 3.1、3.4、5 节 | 用微采样组控制活动序列数；收齐候选后进行全组奖励汇总；有 Qwen3-1.7B 实验 | 其主要指标是解码峰值显存、解码步数与平均长度；不能把长度保持当成抽取准确率或最终 RL 性能不变的证明 |
| [verl 官方 GRPO 文档](https://verl.readthedocs.io/en/latest/algo/grpo.html#configuration) | 将微批大小与算法批次区分，前者用于控制单次处理规模，设计上不应改变算法／收敛行为 | 这是 verl 的设计说明，不是我们修改 TRL 后自动正确的保证 |
| [TRL 0.29.1 GRPO 文档](https://huggingface.co/docs/trl/v0.29.1/en/grpo_trainer)、[对应源码](https://github.com/huggingface/trl/blob/v0.29.1/trl/trainer/grpo_trainer.py) | 奖励按 num_generations 分组计算均值和标准差；候选顺序、输入元数据和评分必须一一对应 | generation_batch_size 不等同于一个可任意降到 1 的独立生成微批开关 |
| [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html#batched-computations-or-slice-computations) | 批量计算与逐条计算不保证浮点结果逐位相同 | 不应把“同一 seed 的输出不同”直接等同于质量下降 |
| [Thinking Machines Lab: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)，2025，研究团队技术报告 | 批大小相关的数值变化会改变生成；其 RL 实验提示采样与训练概率不一致可能影响稳定性 | 其模型、任务和后端与本项目不同，不是本项目必然失稳的证据 |
| [Transformers Text generation](https://huggingface.co/docs/transformers/llm_tutorial#padding-side) | decoder-only 批量生成应正确处理左填充和 attention mask | 改变批次划分时不能忽略输入、位置和停止条件的一致性 |

Infinite Sampling 的具体结果：在 A100-80GB 上，对 Qwen3-1.7B、G=32，论文报告 g=1 时解码峰值从 21.55GB 降至 10.64GB。该数字支持“控制并发序列能节省显存”，不能直接套算本项目 4-bit 模型的峰值或节省比例。论文的动态调度分支还报告了输出明显变短的现象，因此本项目首先采用简单、固定候选数、完整生成的分批方式，不引入依输出长短筛选候选的优化。其奖励／KL 公式也不作为本项目替代实现；继续以当前 TRL 定义为准。[论文实验与方法](https://arxiv.org/html/2506.22950v1)

## 保持算法含义所需的实现条件

1. **采样期间权重不更新。** 同一题的 G 个候选收集齐之前不执行优化器更新，也不切换模型或适配器。简单实现中，先完成当前生成批次，再进入训练更新。
2. **按完整候选组计算优势。** 例如 G=4、每次生成 1 个，必须将 4 个奖励一起计算均值和标准差；不能分为四个大小为 1 的 GRPO 组，也不能按两批各算一次优势。当前保留 `sum_then_normalize` 和 `scale_rewards: group`。
3. **候选与元数据保持对齐。** 固定 sample_id、candidate_index 和原始顺序。切批后合并 token IDs、奖励、旧策略／参考策略概率时不得串题或改变排列。
4. **采样设置不变。** 保持 temperature=0.8、top_p=0.95、top_k=0、最大输出 1024 token、EOS、聊天模板、原文和 JSON 字段要求；保持相同量化精度和后端。不为减少显存而默默截断原文或缩短输出上限。
5. **随机数正确推进。** 固定一次运行的种子，但不在每个候选前重置同一个种子。自然产生相同候选是允许的，不进行强制去重，也不只保留合法或高 reward 的候选，否则会改变采样分布。
6. **缓存相互隔离。** 每个候选的生成历史与响应 KV cache 独立，不能把前一个候选的回答作为下一个的上下文。若共享提示词缓存，必须保证不会被某个候选的后续写入污染。
7. **保留 TRL 的损失与概率处理。** 生成结束后继续使用 TRL 的分组、裁剪、KL、长度掩码和损失归一化。不能将旧策略概率与 KL 参考策略概率混为一谈。需要关注不同生成微批导致的采样／训练概率差异。

## 对本项目的具体判断

上次 G=2 时，训练进程 CUDA 峰值分配显存为 5423.56MiB；当前训练微批已经为 1，而生成阶段默认一次处理生成批次中的序列。最直接的改法是保持逻辑生成批次和候选总数，只在生成调用内部切为 1～2 条执行，再交还完整结果给 TRL。仅将 `per_device_train_batch_size` 设成 1 并不能自动实现这一点。

从算法及资料看，没有理由认为“串行”本身必然降低单条答案的期望质量。但这也不保证其与并行的实际质量完全相同，更不保证 G 从 2 增至 4 就必然提升训练效果。后者同时改变了组内优势估计和计算量，需要单独评估。

当前已知问题是 15/68 个输出未满足严格纯 JSON 解析要求，其中 10 个未完整输出、5 个带代码围栏；53 个通过现有字段校验的输出中仍有 40 个出现字符串 `"null"`。分批生成本身不会自动修复这些问题，增大 G 也不会修复奖励规则对同义时间和年龄表达的处理。

## 应如何验证质量没有明显退化

第一步只验证执行方式，不改变 G：从同一份冻结权重出发，比较 G=2 时的“并行 2 条”和“串行 1 条 × 2 次”。使用相同文本、相同采样参数、多个随机种子；汇总分布，不以逐字相同为通过条件。此项目前尚未执行。

第二步再考察候选数：固定 G=4，比较“串行 1 条 × 4 次”和“分批 2 条 × 2 次”。只有这些方式检查通过后，再比较 G=2 与 G=4 的训练效果，避免将候选数和执行方式同时改变而无法归因。

建议记录：

- 严格 JSON 解析率；真实 `null` 与字符串 `"null"` 的使用；字段与链结构约束通过率。
- 节点、边、时间及人物归属的匹配情况，必要时人工复核同义表达；不只看总 reward。
- 组内不同候选比例、奖励标准差、零优势组比例。
- 生成长度和截断率、采样／训练概率差异及训练稳定性。
- 生成和训练各自的峰值显存、系统总显存占用与耗时。

最终训练质量应在 SFT 和 RL 都未用于参数训练的独立数据上比较，并保持原文采样、每次更新的原文数量、优化步数和其他超参数可比。若只固定运行时长，串行方式可能因吞吐下降而少做更新，这属于计算预算变化造成的差异，需要与生成方式本身的影响分开。

本次仅完成资料核查和方案判断，未修改训练实现，也未开始新训练。
