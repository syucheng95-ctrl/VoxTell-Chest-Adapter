# AGENT.md

本文件面向进入本项目协作的智能体或开发者，记录当前主线、关键结论、代码入口与工程约定。

## 1. 项目目标

本项目基于 `VoxTell + nnUNet`，面向胸部 CT finding segmentation。当前目标不是重写视觉主干，而是：

- 保住或提升 case-level `mean_dice`
- 尽量把 voxel-level `micro_dice` 保持在 baseline 以上
- 搞清楚后续真正值得投入的方向：数据、监督、训练、结构

当前共识：

- `nnUNet` 主干不是当前主战场。
- 纯重型 text-grounding 路线在当前数据规模下不稳定，不再作为主推主线。
- adapter 线已经接近“局部最优微调”上限，但仍然是很重要的对照和保底线。

## 2. 当前版本总览

### 2.1 关键结果表

| 演进阶段 | 版本 | Mean Dice | Micro Dice | Precision | Recall | 定位与战果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 基准 | Baseline | 0.2139 | 0.4228 | 0.2493 | 0.3099 | 原始官方模型，参考线 |
| 第一代 | v2 | 0.2206 | 0.4099 | 0.2435 | 0.3461 | 第一版稳定超过 baseline 的 adapter，证明“强约束 adapter”成立 |
| 第一代 | v4_catfix | **0.2214** | 0.4054 | 0.2358 | **0.3524** | “激进派”：当前 `mean_dice / recall` 峰值，但 FP 偏多 |
| 第二代 | v5.1 | 0.2209 | 0.4080 | 0.2372 | 0.3497 | supervision 重构成功版，综合最强的 recall/mean 对照 |
| 第三代 | v6.4_stage3_fix | 0.2141 | 0.4234 | **0.2586** | 0.3089 | “洁癖派”：candidate+risk 落地后 precision 峰值 |
| 第三代 | v6.5 | 0.2133 | 0.4242 | 0.2509 | 0.3069 | 当前 `v6` 稳健主干，`micro` 长期可复现强点 |
| 第四代 | v6.6 | 0.2126 | **0.4269** | 0.2548 | 0.3016 | “极端派”：门控过硬，`micro` 最高但 recall 最低 |
| 第四代 | v6.6b | 0.2146 | 0.4214 | 0.2565 | **0.3124** | “松绑版”：成功救回 recall/mean，失去 `micro` 优势 |
| 当前 | v6.7 | 0.2146 | 0.4215 | 0.2565 | 0.3123 | “均衡派”：基本等价 `v6.6b`，没有形成新 Pareto 前沿 |

### 2.2 当前判断

- 如果按 `mean_dice / recall` 看，当前代表版仍是 `v4_catfix` 和 `v5.1`。
- 如果按 `micro_dice` 看，当前最高点是 `v6.6`，但它过于保守，不适合作为唯一主线。
- 如果按“稳定、低 FP、可继续扩展”看，`v6.5` 仍是更适合做结构基线的版本。
- `v6.6b / v6.7` 证明 MSAR 的“局部化”思路不是错的，但只靠 adapter 小改已经很难再拉出明显新前沿。

## 3. adapter 线真正学到了什么

- adapter 在当前设定下更像“行为调节器”，不太像能大幅重建表征的主引擎。
- 更强结构约束比更重的文本调制更靠谱。
- supervision 的收益通常比继续叠结构头更扎实。
- 训练策略不是附属项，而是结构成败的一部分。
- `candidate -> risk -> refine` 的职责拆分，是 `v6` 系列最有价值的遗产。

## 4. 重复出现的坑

- 只要结构本质上在放大阳性响应，而没有精准 FP 约束，最后大概率就是高 recall、低 precision。
- suppression / risk 很容易学成“整体保守化”，而不是精准打掉 FP。
- 训练统计变好，不代表 full val 会变好。
- 结构收益经常会被训练设计掩盖。
- 经验参数一旦过多，结论会变得模糊，很难知道到底是思路成立还是调参碰巧。

## 5. 当前主线建议

adapter 线可以收尾，但不应该再作为唯一主线。后续更值得投入的四条主线是：

1. 数据：覆盖、采样、hard negative、小病灶、稀有类别
2. 监督：FP-aware、boundary-aware、lesion-size-aware、category-aware supervision
3. 训练：stage 设计、冻结/解冻、negative sampling schedule、部分主干训练
4. 结构：每次只回答一个问题，避免继续滚“多头叠加式”版本树

## 6. 关键代码入口

### adapter 结构

- `VoxTell/voxtell/model/chest_candidate_risk_adapter.py`
  - `v6.4` 的 `candidate + risk` 双头
- `VoxTell/voxtell/model/chest_candidate_risk_refine_adapter.py`
  - `v6.5` 的 `candidate + risk + weak refine`
- `VoxTell/voxtell/model/chest_masked_risk_refine_adapter.py`
  - `v6.6` 的 `Masked Risk + Uncertainty Refine`
- `VoxTell/voxtell/model/chest_masked_risk_refine_adapter_v66b.py`
  - `v6.6b` 的少参数松绑版
- `VoxTell/voxtell/model/chest_candidate_selective_risk_refine_adapter.py`
  - `v6.7` 的 selective-risk 版

### wrapper

- `VoxTell/voxtell/model/v6_adapter_model.py`
- `VoxTell/voxtell/model/v65_adapter_model.py`
- `VoxTell/voxtell/model/v66_adapter_model.py`
- `VoxTell/voxtell/model/v66b_adapter_model.py`
- `VoxTell/voxtell/model/v67_adapter_model.py`

### 训练与验证

- `scripts/training/train_voxtell_adapter.py`
  - 当前支持 `--adapter-version v6_4 / v6_5 / v6_6 / v6_6b / v6_7`
- `scripts/experiments/run_voxtell_val_adapter_raw.py`
  - 当前支持同样的 `adapter-version`
- `scripts/analysis/evaluate_voxtell_val_adapter_raw.py`
  - full val 后汇总指标

## 7. 已完成验证状态

已完成：

- `v6.6 full train + full val + evaluate`
- `v6.6b full train + full val + evaluate`
- `v6.7 full train + full val + evaluate`
- `v6.7 vs v6.6b` 单 case 中间张量对比

已确认结论：

- `v6.7` 不是“代码没生效”，而是“效果太弱，没把结果从 v6.6b 那个平衡点拉出来”。
- adapter 线目前更像在同一个窄 Pareto 区间内重新分配 `precision / recall / micro / mean`。
- 下一阶段真正更值得押注的，不是继续滚 adapter 小版本，而是四条主线的系统对比。

## 8. 工程与空间约定

### Prompt cache

- 能复用 `train_prompt_embeddings.pt` 就必须复用。

### Smoke

- `smoke` 只做链路验证。
- 跑完即删，不长期保留。

### 保存逻辑

- 默认不保存 `training_state.pt`。
- 中间里程碑保存轻量 `adapter_step_xxxx.pt`。
- final run 只保留 `adapter_state.pt`、`train_metrics.json` 和最终指标。

## 9. 一句话原则

当前不再追求“更强文本主导”，而是追求：
**先提候选，再显式识别并控制误报风险，同时把真正的瓶颈转移到数据、监督、训练和结构四条主线上。**
