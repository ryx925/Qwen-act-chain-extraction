# 阶段任务清单

- [x] 1. 配置 WSL2 CUDA/PyTorch/bitsandbytes/TRL/PEFT 环境并完成 NF4 自检。
- [x] 2. 固化 Linux 精确依赖锁文件并记录环境位置。
- [x] 3. 建立出行链 JSON Schema、字段缺失规则、多段行程关系与标注规范草案。
- [x] 4. 建立 12 条合成标注样例，并按 `source_id` 生成 8/2/2 分组拆分。
- [x] 5. 实现 JSONL 校验、无泄漏拆分和对话式 prompt/completion 数据转换。
- [x] 6. 完成 Qwen3-1.7B 4-bit QLoRA 单步冒烟训练并记录峰值显存（2945.87 MiB）。
- [x] 7. 实现字段级 Precision/Recall/F1、整链准确率、缺失和多余预测 ID 报告。

任务 1–7 已完成。单步训练损失为 1.9332、耗时 1.5516 秒；这里只验证训练链路与显存可行性。样例为合成数据，评估自测为 gold replay，仅用于验证指标实现，均不能代表正式模型效果。
