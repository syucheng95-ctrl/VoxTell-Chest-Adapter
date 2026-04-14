# AGENT.md

本文件面向进入本项目协作的智能体或开发者，目的是让后续工作尽快进入状态，避免重复排障和误操作。

## 1. 项目目标

当前项目围绕 `VoxTell` 在 `ReXGroundingCT` 数据上的实验展开，主要包括三类工作：

1. 跑通 `VoxTell` 在胸部 CT finding grounding / segmentation 上的推理流程。
2. 对比 `raw prompt`、`structured prompt`、`hybrid-by-category` 三种文本输入策略。
3. 在现有推理代码基础上，探索 `LoRA` 形式的最小可行微调方案。

## 2. 当前结论

截至目前，已经确认：

- `VoxTell` 推理链路可以正常跑通。
- `structured prompt` 会明显改变模型行为，但在 full-val 上不能直接替代 `raw prompt`。
- `hybrid-by-category` 在 `mean Dice` 上优于纯 `raw` 和纯 `structured`，但 `micro Dice` 仍未超过纯 `raw`。
- 当前更值得继续推进的方向，不是继续纯 prompt engineering，而是做最小可行的 `LoRA` 微调。

## 3. 关键环境

- 操作系统：Windows
- Python 环境：`C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe`
- 项目根目录：`C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\题目2-demo`

不要默认使用系统 `python`。之前已经确认，正确环境是 `pytorch` 这个 conda 环境。

## 4. 目录说明

### 核心目录

- `VoxTell/`
  - 上游模型源码
- `models/`
  - 本地模型权重和拆分权重
- `datasets/`
  - 本地数据集、镜像 metadata、subset 数据
- `outputs/`
  - 推理输出和评估结果

### 文档目录

- `docs/plans/`
  - 实施方案文档
- `docs/reports/`
  - 实验报告
- `docs/notes/`
  - 阅读笔记、结构图、过程性笔记
- `docs/references/`
  - 论文、原始 PDF、参考资料

### 脚本目录

- `scripts/experiments/`
  - 跑实验和推理的脚本
- `scripts/analysis/`
  - 指标评估、分析、对比脚本
- `scripts/utils/`
  - 工具脚本
- `scripts/docx/`
  - Word 报告生成脚本

### 状态目录

- `status/`
  - 当前进度记录和交接说明

## 5. 重要文件入口

- `status/CURRENT_STATUS_VOXTELL.txt`
  - 当前阶段状态说明
- `docs/reports/VoxTell_full_val_report.docx`
  - 完整 public val 实验汇报
- `docs/plans/VoxTell_LoRA_minimal_plan.docx`
  - 当前最小可行 LoRA 方案
- `outputs/voxtell_val_metrics.json`
  - full-val 原始评估结果
- `outputs/voxtell_val_hybrid_metrics.json`
  - hybrid-by-category 评估结果

## 6. 数据与来源

### 当前本地可用数据

- `datasets/ReXGroundingCT_subset/`
  - 已补齐完整 public `val`
  - 包含 `images_flat/` 和 `segmentations/`

### 关于数据来源

请注意：

- 官方 `rajpurkarlab/ReXGroundingCT` 当前访问到的内容并不完整，不要假设它直接包含全部图像。
- 本项目当前可用图像和标注，是从镜像仓库路线补齐的。
- 不要随意重下大体量数据，先核对本地已有内容。

## 7. 运行相关注意事项

### Hugging Face 缓存

之前已经遇到过默认缓存目录权限问题。运行时优先考虑把缓存显式指到项目内：

- `HF_HOME`
- `HF_HUB_CACHE`

项目里已有 `hf_cache/`，这是运行缓存，不是脏文件，不要随手删。

### 内存问题

之前出现过 Windows 虚拟内存压力导致的失败，包括：

- `numpy.core._exceptions._ArrayMemoryError`
- Windows 弹出“内存不能为 read/write”一类提示

这不等于环境坏了，通常是系统级内存压力过大。遇到这种问题时优先检查：

- 是否有残留 `python.exe`
- `Acrobat.exe`、`ollama.exe` 等高占用进程是否还在
- 当前是否又在并行加载 Qwen 模型和大体积 CT

## 8. 常用脚本

### 实验脚本

- `scripts/experiments/run_voxtell_subset.py`
  - 小样本对比实验
- `scripts/experiments/run_voxtell_val.py`
  - 完整 public val 批处理推理

### 分析脚本

- `scripts/analysis/evaluate_voxtell_subset.py`
- `scripts/analysis/evaluate_voxtell_val.py`
- `scripts/analysis/evaluate_voxtell_hybrid.py`
- `scripts/analysis/inspect_structured_prompts.py`

### 文档脚本

- `scripts/docx/create_voxtell_val_report.js`
- `scripts/docx/create_voxtell_lora_plan_doc.js`

## 9. 常用命令

以下命令默认都在项目根目录执行，并优先使用：

- `C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe`

### 9.1 跑 3-case 小样本实验

```powershell
$env:HF_HOME="$PWD\hf_cache"
$env:HF_HUB_CACHE="$PWD\hf_cache\hub"
$env:TRANSFORMERS_NO_TF="1"
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\experiments\run_voxtell_subset.py
```

### 9.2 跑 full public val

```powershell
$env:HF_HOME="$PWD\hf_cache"
$env:HF_HUB_CACHE="$PWD\hf_cache\hub"
$env:TRANSFORMERS_NO_TF="1"
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\experiments\run_voxtell_val.py
```

如果中断后继续跑，脚本会基于已有 `summary_comparison.json` 自动跳过已完成 case。

### 9.3 限制只跑前 N 个 val case

```powershell
$env:HF_HOME="$PWD\hf_cache"
$env:HF_HUB_CACHE="$PWD\hf_cache\hub"
$env:TRANSFORMERS_NO_TF="1"
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\experiments\run_voxtell_val.py --limit 5
```

### 9.4 强制重跑 full val

```powershell
$env:HF_HOME="$PWD\hf_cache"
$env:HF_HUB_CACHE="$PWD\hf_cache\hub"
$env:TRANSFORMERS_NO_TF="1"
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\experiments\run_voxtell_val.py --force
```

### 9.5 评估结果

```powershell
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\analysis\evaluate_voxtell_subset.py
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\analysis\evaluate_voxtell_val.py
C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe scripts\analysis\evaluate_voxtell_hybrid.py
```

### 9.6 生成 Word 文档

```powershell
node scripts\docx\create_voxtell_val_report.js
node scripts\docx\create_voxtell_lora_plan_doc.js
```

## 10. 最短复现路径

如果新接手的人只想最快确认“项目是不是活的”，建议按下面顺序：

1. 确认使用的是 `pytorch` conda 环境，而不是系统 `python`。
2. 确认本地关键目录存在：
   - `models/voxtell_v1.1/`
   - `datasets/ReXGroundingCT_subset/images_flat/`
   - `datasets/ReXGroundingCT_subset/segmentations/`
3. 在项目根目录设置：
   - `HF_HOME`
   - `HF_HUB_CACHE`
   - `TRANSFORMERS_NO_TF=1`
4. 先跑 3-case 小样本实验：
   - `scripts/experiments/run_voxtell_subset.py`
5. 再跑 subset 评估：
   - `scripts/analysis/evaluate_voxtell_subset.py`
6. 如果这两步都正常，再考虑跑 full val。

这是当前最短、最稳的复现路径。

## 11. Prompt 实验约定

目前实验的核心对比对象是：

- `raw prompt`
  - 直接使用原始 findings 文本
- `structured prompt`
  - 将 findings 解析为结构化字段，再拼模板
- `hybrid-by-category`
  - 按 category 选择用 `raw` 或 `structured`

不要在没有单独记录实验条件的情况下，把：

- prompt 改动
- 推理逻辑改动
- 评估逻辑改动
- 微调改动

混在同一轮实验里。否则后面无法归因。

## 12. LoRA 工作约定

当前推荐的最小可行方向：

- 冻结 `Qwen`
- 冻结 `encoder`
- 冻结卷积 `decoder`
- 只对以下模块做 LoRA：
  - `project_text_embed`
  - `transformer_decoder`
  - `project_to_decoder_channels`

第一阶段不要直接对整套 Qwen 做 LoRA，也不要先改 3D 卷积主干。

### 12.1 当前推荐的具体 LoRA 模块

建议优先考虑以下线性层：

- `project_text_embed.0`
- `project_text_embed.2`
- `transformer_decoder.layers.0.linear1`
- `transformer_decoder.layers.0.linear2`
- `transformer_decoder.layers.0.self_attn.out_proj`
- `transformer_decoder.layers.0.multihead_attn.out_proj`
- `transformer_decoder.layers.1.linear1`
- `transformer_decoder.layers.1.linear2`
- `transformer_decoder.layers.1.self_attn.out_proj`
- `transformer_decoder.layers.1.multihead_attn.out_proj`
- `transformer_decoder.layers.2.linear1`
- `transformer_decoder.layers.2.linear2`
- `transformer_decoder.layers.2.self_attn.out_proj`
- `transformer_decoder.layers.2.multihead_attn.out_proj`
- `transformer_decoder.layers.3.linear1`
- `transformer_decoder.layers.3.linear2`
- `transformer_decoder.layers.3.self_attn.out_proj`
- `transformer_decoder.layers.3.multihead_attn.out_proj`
- `transformer_decoder.layers.4.linear1`
- `transformer_decoder.layers.4.linear2`
- `transformer_decoder.layers.4.self_attn.out_proj`
- `transformer_decoder.layers.4.multihead_attn.out_proj`
- `transformer_decoder.layers.5.linear1`
- `transformer_decoder.layers.5.linear2`
- `transformer_decoder.layers.5.self_attn.out_proj`
- `transformer_decoder.layers.5.multihead_attn.out_proj`
- `project_to_decoder_channels.0.0`
- `project_to_decoder_channels.0.2`
- `project_to_decoder_channels.1.0`
- `project_to_decoder_channels.1.2`
- `project_to_decoder_channels.2.0`
- `project_to_decoder_channels.2.2`
- `project_to_decoder_channels.3.0`
- `project_to_decoder_channels.3.2`
- `project_to_decoder_channels.4.0`
- `project_to_decoder_channels.4.2`

### 12.2 当前不建议第一阶段直接做的事

- 不要优先对 `Qwen/Qwen3-Embedding-4B` 本体做 LoRA
- 不要优先改 `encoder`
- 不要优先改 3D 卷积 `decoder`
- 不要第一阶段直接处理 `MultiheadAttention.in_proj_weight`

原因是 `in_proj_weight` 把 `q/k/v` 打包在一起，现成 `PEFT` 不够顺手。若后面 Phase A 有明显收益，再考虑把 attention 改写成显式的 `q_proj / k_proj / v_proj`。

## 13. 修改时的边界

### 可以做的

- 新增实验脚本
- 新增分析脚本
- 新增报告和方案文档
- 在不破坏已有结论的前提下补充实验

### 不要随便做的

- 不要删除 `outputs/` 里已有结果
- 不要重命名 `datasets/`、`models/` 下的核心目录
- 不要覆盖已有报告文件，除非明确要更新对应版本
- 不要默认清空 `hf_cache/`
- 不要默认移动 `VoxTell/` 源码结构

## 14. 新增实验前的检查清单

开始任何新实验前，先确认：

1. 这轮实验只改一个主变量。
2. 结果输出目录不会覆盖已有结论。
3. 评估脚本和推理脚本使用的是同一批 case。
4. 当前系统没有残留的大内存 `python.exe`。
5. `Acrobat.exe`、`ollama.exe` 等高占用进程没有占满虚拟内存。
6. `hf_cache/` 路径和当前命令一致。

## 15. 建议的下一步顺序

如果接下来继续推进，建议按这个顺序：

1. 基于现有 `VoxTell` 推理代码，补一个最小 LoRA 微调脚本。
2. 先做小规模 smoke test，确认 loss 能下降、输出不崩。
3. 再用统一 val 集对比：
   - `raw baseline`
   - `structured baseline`
   - `hybrid baseline`
   - `raw + LoRA`
4. 最后再考虑是否需要更细粒度的 prompt routing 或 attention 结构改造。

## 16. 一句话原则

这个项目现在最重要的不是“再加更多零散尝试”，而是：

在保持现有实验可复现的前提下，把 `LoRA` 微调链路稳稳接上。
