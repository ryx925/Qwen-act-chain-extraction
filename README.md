# Qwen3-1.7B 纯文本出行链提取

本项目用于完成纯文本出行链提取的完整流程：数据准备、Qwen3-1.7B 参数高效微调、评估、推理与结果归档。

## 当前阶段

- 已建立项目目录结构。
- 已完成 WSL2 CUDA 训练环境和 Windows 轻量环境配置。
- 默认训练路线为 4-bit QLoRA，以适配本机 RTX 3060 Laptop 6GB 显存。
- 已完成数据 Schema、合成样例、无泄漏拆分、训练入口和结构化评估指标。
- Qwen3-1.7B 单步 QLoRA 冒烟训练通过，峰值显存 2945.87 MiB。

## 环境

- Windows 11 / PowerShell
- Python 3.12
- NVIDIA RTX 3060 Laptop（6GB）
- PyTorch CUDA 12.8 wheel
- 模型：`Qwen/Qwen3-1.7B`

训练环境使用 WSL2 Ubuntu 24.04，并位于 WSL 的 ext4 文件系统中：
`/home/gollie/.venvs/qwen3-trip-chain`。Windows `.venv` 保留用于轻量数据检查。

激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

进入 WSL 训练环境：

```powershell
wsl -d Ubuntu-24.04
cd /mnt/c/Users/24171/Desktop/Qwen
source /home/gollie/.venvs/qwen3-trip-chain/bin/activate
```

重新安装环境时：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements\bootstrap.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\common.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements\base.txt
```

当前引导环境的实际版本记录在 `requirements/lock-bootstrap.txt`，完整 WSL GPU 环境记录在
`requirements/lock-wsl-cu128.txt`。具体状态见 `docs/environment.md`。

`common.txt` 可先独立安装；`torch-cu128.txt` 是体积约 2.75GB 的 GPU 核心包，必须在
`base.txt` 之前安装，防止依赖解析器退回 CPU 版 PyTorch。

## 数据准备与测试

```bash
export PYTHONPATH=src
python -m trip_chain.data.prepare \
  --input data/samples/annotated_examples.jsonl \
  --output-dir data/processed
pytest -q
```

## QLoRA 冒烟训练

```bash
python -m trip_chain.training.train_qlora \
  --smoke-test \
  --model-name-or-path /home/gollie/models/Qwen3-1.7B-modelscope
```

省略 `--model-name-or-path` 时使用配置中的 `Qwen/Qwen3-1.7B`。正式训练时移除
`--smoke-test`。本机冒烟结果记录在 `outputs/metrics/smoke_train_metrics.json`。

## 结构化评估

```bash
export PYTHONPATH=src
python -m trip_chain.evaluation.metrics \
  --gold data/processed/test_gold.jsonl \
  --predictions outputs/predictions/test_predictions.jsonl
```

## 目录约定

```text
configs/                 模型、数据、训练、推理配置
data/
  raw/                   原始数据，只读保留
  interim/               清洗和标注过程数据
  processed/             可直接训练/评估的数据
  samples/               小样本与格式示例
docs/                    标签体系、数据规范与实验说明
models/
  base/                  可选的本地基础模型
  adapters/              LoRA/QLoRA 适配器
  merged/                合并后的模型（如需要）
notebooks/               探索与人工检查
outputs/
  checkpoints/           训练检查点
  predictions/           推理结果
  metrics/               评估指标
  logs/                  运行日志
src/trip_chain/
  data/                  数据读取、清洗、转换
  training/              训练逻辑
  inference/             推理与结构化解析
  evaluation/            指标与错误分析
  utils/                 通用组件
tests/                   自动化测试
```

`data/`、`models/`、`outputs/` 中的大文件默认不提交到版本库，只保留目录占位文件。
