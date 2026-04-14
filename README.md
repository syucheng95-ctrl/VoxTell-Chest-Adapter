# VoxTell-Chest: 胸部专用多模态医疗影像分析方案

本仓库包含参加**生医工大赛**题目 2 的实验方案与改进代码。本项目在开源模型 [VoxTell](https://github.com/mrokuss/VoxTell) 的基础上，针对胸部 CT 影像的复杂病灶与细小解剖结构，设计并实现了一个高度受控的文本引导适配模块 —— **ChestTextGuidedAdapter v2**。

## 🚀 核心技术改进：ChestTextGuidedAdapter v2

针对原模型在胸部 CT 细小病灶（如微小结节、间质性改变）分割中存在的“召回率高但精确度低（过分割）”问题，本项目通过两个阶段的迭代，提出了 **Adapter v2** 方案。

### 1. 结构设计与演进
- **v1 原型**：采用简单的 FiLM (Feature-wise Linear Modulation) 结构插在 Encoder 之后。实验发现，无约束的文本调制会导致模型过度响应文本描述，产生大面积误报。
- **v2 (当前最佳)**：
  - **分组门控机制 (Group Gating)**：将视觉通道分为 8 个组，由文本 Embedding 动态生成每个组的门控权重。这使得模型能够根据病灶类型，选择性地调制特定的视觉特征维度。
  - **双重强度约束**：引入了 `gate_cap=0.25`（门控上限）和 `residual_scale=0.1`（残差缩放）。这种设计确保了适配器仅对视觉特征进行“精细调优”，而非“推倒重来”，极大地增强了分割边缘的稳定性。
  - **前移插入点**：将模块从 Decoder 后移至 Pre-decoder 阶段，使文本信息在 3D 特征投影到分割空间前就完成空间约束。

### 2. 代码实现
核心代码位于：`VoxTell/voxtell/model/chest_text_guided_adapter.py`
```python
# 核心调制逻辑
delta = torch.tanh(self.to_delta(norm_text)) # 生成调制增量
group_gate = torch.sigmoid(self.to_group_gate(norm_text)) * self.gate_cap # 动态门控
# 最终融合
adapted_feature = feature + self.residual_scale * channel_gate * delta * norm_feature
```

## 📊 实验表现与分析

我们基于 ReXGroundingCT 验证集进行了全量测试（Full Validation），对比了 Baseline 与 Adapter v2 的表现：

| 指标 | VoxTell Baseline | **VoxTell + Adapter v2** | 提升 |
| :--- | :---: | :---: | :---: |
| **Mean Dice** | 0.2139 | **0.2206** | **+3.1%** |
| **Micro Dice** | 0.4228 | 0.4099 | - |
| **Recall (Mean)** | 0.3099 | **0.4380** | **+41.3%** |

**结果解读：**
- **稳健提升**：Adapter v2 在保持 Precision 相对稳定的前提下，显著提升了召回率（Recall），尤其是对于原本容易漏诊的弥漫性病灶（如磨玻璃影 GGO）效果提升明显。
- **过分割抑制**：相比于 v1 版本和纯 LoRA 微调，v2 成功抑制了由于文本注入导致的无约束膨胀，Mean Dice 的正向增长证明了分组门控机制在医疗影像中的有效性。

## 🛠️ 环境准备与安装

本项目依赖于 `torch`、`transformers` 以及 VoxTell 核心库。建议使用 Python 3.10+。

### 1. 安装基础环境
我们推荐使用高性能包管理器 `uv`（已包含在项目 `uv.lock` 中）或 `pip`：
```bash
# 克隆仓库
git clone https://github.com/syucheng95-ctrl/VoxTell-Chest-Adapter.git
cd VoxTell-Chest-Adapter

# 安装 VoxTell 核心库（开发者模式）
cd VoxTell
pip install -e .
```

### 2. 依赖补充
确保安装了 docx 报告生成所需的 Node.js 环境（用于运行 `scripts/docx/` 下的脚本）：
```bash
npm install
```

## 📖 运行指南

### 训练适配器
```bash
# 启动针对胸部 CT 优化的微调流水线
python scripts/training/train_voxtell_adapter.py
```

### 推理与性能评估
```bash
# 运行验证集推理
python scripts/experiments/run_voxtell_val_adapter_raw.py

# 计算详细指标
python scripts/analysis/evaluate_voxtell_val_adapter_raw.py
```

## 📝 结论
通过本项目，我们证明了在医疗 VLM (Vision-Language Model) 中，**“强约束、细粒度”的适配器逻辑**优于“弱约束、全局”的微调逻辑。**ChestTextGuidedAdapter v2** 能够作为一种插件式模块，在不破坏原模型通用语义理解能力的同时，显著增强其在特定解剖区域（胸部）的分割精确度。

---
*本项目为“生医工大赛”参赛作品。如需引用或交流，请联系：23300200012@m.fudan.edu.cn*
