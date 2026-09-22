# 环境状态

更新时间：2026-09-22

## 已确认硬件

- GPU：NVIDIA GeForce RTX 3060 Laptop GPU
- 显存：6144 MiB
- NVIDIA 驱动：596.08
- Python：3.12.3（64 位）
- 训练虚拟环境：`/home/gollie/.venvs/qwen3-trip-chain`（WSL ext4）
- Windows 轻量环境：项目内 `.venv`

## 技术路线

6GB 显存无法稳妥承担 Qwen3-1.7B 的全参数训练，因此默认使用 4-bit NF4 QLoRA、梯度检查点、batch size 1 和梯度累积。推理同样默认 4-bit 加载。首次接入真实数据后，仍需根据文本长度做一次显存基准测试。

PyTorch 固定为 `2.11.0+cu128`。CUDA 运行时由 wheel 自带，不要求本机另装 CUDA Toolkit；本机驱动能够覆盖该运行时。Qwen3 的 Transformers 最低要求是 4.51，当前引导环境安装为 5.17.0。

## 当前安装状态

- 已完成：PyTorch CUDA、Transformers、Datasets、Accelerate、PEFT、TRL、bitsandbytes 和数据/测试依赖。
- 已校验：CUDA 可用、bf16 可用、bitsandbytes NF4 CUDA 运算通过、`pip check` 无冲突。
- 精确依赖版本：`requirements/lock-wsl-cu128.txt`。
- 模型快照：`/home/gollie/models/Qwen3-1.7B-modelscope`（WSL ext4）。
- Qwen3-1.7B 已完成 4-bit NF4 QLoRA 单步冒烟训练：峰值显存 2945.87 MiB，耗时 1.5516 秒，训练损失 1.9332。

从零重建训练环境时执行：

```powershell
python3 -m venv /home/gollie/.venvs/qwen3-trip-chain
/home/gollie/.venvs/qwen3-trip-chain/bin/python -m pip install -r requirements/torch-cu128.txt
/home/gollie/.venvs/qwen3-trip-chain/bin/python -m pip install -r requirements/base.txt
```

TensorBoard 暂放在 `requirements/optional.txt`，因为当前包源没有适配此 Python 环境的 `grpcio` wheel；这不阻塞训练，可先使用 Trainer 的 JSON 日志。

模型快照通过 ModelScope 下载，以规避当前网络环境访问 Hugging Face 权重文件不稳定的问题。配置仍保留官方模型 ID，训练时可用 `--model-name-or-path` 切换至本地快照。
