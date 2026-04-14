# VoxTell-Chest: 胸部专用多模态医疗影像分析方案

本仓库包含参加**生医工大赛**题目 2 的实验方案与改进代码。本项目在开源模型 [VoxTell](https://github.com/mrokuss/VoxTell) 的基础上，针对胸部 CT 影像的特点，设计并实现了一个轻量化的文本引导适配模块（ChestTextGuidedAdapter）。

## 🚀 核心改进：ChestTextGuidedAdapter

针对原模型在胸部 CT 细小病灶分割中存在的“召回率高但误报严重”的问题，本项目提出了 **ChestTextGuidedAdapter** 方案：

- **技术原理**：在视觉编码器与解码器之间插入门控 FiLM (Feature-wise Linear Modulation) 模块。
- **动态调制**：利用文本 Embedding 生成 `gamma` (缩放) 和 `beta` (偏移) 参数，对 3D 视觉特征进行动态调制。
- **版本迭代**：
  - **v1**：初版实现，验证了模块的可行性，但仍存在过分割现象。
  - **v2 (当前最佳)**：将插入点前移至预解码阶段，并引入分组 Gate 机制，显著提升了分割精度（Mean Dice 从 0.2139 提升至 0.2206）。

## 📂 项目结构

```text
├── VoxTell/            # 核心模型代码（已集成 ChestTextGuidedAdapter）
├── scripts/            # 实验脚本
│   ├── training/       # 模型训练脚本 (Adapter/LoRA)
│   ├── experiments/    # 推理与验证实验
│   └── analysis/       # 结果评估与指标计算
├── docs/               # 方案设计说明与实验报告
├── 实验记录/           # 详细的模块开发路线与验证逻辑
└── PROJECT_LAYOUT.md   # 详细的项目布局说明
```

## 🛠️ 快速开始

### 1. 环境准备
本项目建议使用 `uv` 或 `pip` 安装依赖：
```bash
cd VoxTell
pip install -e .
```

### 2. 模型训练
运行以下脚本启动胸部专用适配器的微调：
```bash
python scripts/training/train_voxtell_adapter.py
```

### 3. 实验验证
验证 Adapter v2 在验证集上的表现：
```bash
python scripts/experiments/run_voxtell_val_adapter_raw.py
```

## 📊 实验结论
实验证明，相比于通用的 LoRA 微调，**ChestTextGuidedAdapter v2** 能够更显式地引导模型关注文本描述的解剖区域，有效抑制了无约束的分割扩张，更符合医疗影像诊断的严谨性要求。

---
*注：本项目仅供学术交流与比赛使用，不包含 ReXGroundingCT 数据集原始数据。*
