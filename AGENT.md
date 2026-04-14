# AGENT.md

本文件面向进入本项目协作的智能体或开发者，旨在提供最新的项目状态、技术共识以及 V3 演进路径。

## 1. 项目目标

本项目致力于在 `VoxTell` 基础上开发针对胸部 CT 影像的精准分割方案。当前核心工作：
1. 迭代并优化 **ChestTextGuidedAdapter** 模块。
2. 通过文本引导的门控机制，提升病灶分割的召回率并压低误报。

## 2. 当前结论

- **核心突破**：`ChestTextGuidedAdapter v2` 成功将 Mean Dice 指标从 Baseline 的 0.2139 提升至 **0.2206**。
- **性能分析**：Recall 大幅增长（+41.3%），证明适配器对病灶的敏感度极佳。
- **当前瓶颈**：Micro Dice 仍略低于 Baseline，假阳性（过分割）是下一步必须攻克的重点。
- **技术定位**：专用 Adapter 架构在医疗影像微调中展现出比通用的 LoRA 更好的可解释性和精度上限。

## 3. 关键环境

- **Conda 环境**：`pytorch` (位于 `C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe`)
- **项目根目录**：`C:\Users\Sunyucheng\Desktop\作业(大学相关）\生医工大赛\题目2-demo`
- **硬件注意**：Windows 内存压力大时需清理残留进程。

## 4. 重要文件入口

- `VoxTell/voxtell/model/chest_text_guided_adapter.py`：核心适配器代码。
- `outputs/voxtell_val_adapter_raw_v2_metrics.json`：当前标杆（v2）的验证结果。
- `实验记录/胸部专用融合模块_最小实现方案.md`：记录了 v1/v2 的迭代心路。
- `实验计划/下一阶段结构改进计划.md`：**V3 架构设计的详细说明书。**

## 5. 常用命令

### 5.1 训练适配器
```powershell
# 设置缓存路径并运行
$env:HF_HOME="$PWD\hf_cache"; $env:TRANSFORMERS_NO_TF="1"
python scripts\training\train_voxtell_adapter.py
```

### 5.2 验证 v2 指标
```powershell
python scripts\experiments\run_voxtell_val_adapter_raw.py
python scripts\analysis\evaluate_voxtell_val_adapter_raw.py
```

## 6. Adapter 演进约定 (Phase: V3)

当前项目的重心已完全转向 **Adapter v3** 的开发：

- **设计原则 1：类别感知 (Category-aware)** 
  - 针对结节、积液、实变等不同病灶，分配不同的动态调制 Gate。
- **设计原则 2：抑制分支 (Suppression Branch)** 
  - 引入负向门控机制，显式抑制非病灶区域的响应。
- **冻结约定**：除非万不得已，否则不要修改 `Qwen` 本身，所有创新集中在 Adapter 这一轻量插件中。

## 7. 建议的下一步顺序

1. **实现 V3 架构**：在 `chest_text_guided_adapter.py` 中增加类别感知的 MLP 头。
2. **优化 Loss 函数**：在训练逻辑中增加对假阳性的额外惩罚权重。
3. **闭环测试**：目标是实现 `Micro Dice > 0.4228` 且 `Mean Dice > 0.2206`。

## 8. 一句话原则

我们不再盲目增加参数，而是要通过**引入医学逻辑（类别感知与背景抑制）**来精炼模型的分割决策。
