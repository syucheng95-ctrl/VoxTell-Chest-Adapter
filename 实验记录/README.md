# 实验记录

本目录用于沉淀本项目已经完成的实验、当前结论、失败原因与下一步决策，避免后续重复试错。

## 0. 快速入口

- adapter 线阶段总结：[adapter线阶段总结.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/实验记录/adapter线阶段总结.md)
- 项目总览：[README.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/README.md)
- 协作说明：[AGENT.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/AGENT.md)

## 1. 项目阶段

当前项目围绕 `VoxTell` 在 `ReXGroundingCT` 数据上的胸部 CT finding grounding / segmentation 实验展开，主线经历了三个阶段：

1. 跑通 `VoxTell` 原始推理链路。
2. 对比 `raw prompt / structured prompt / hybrid-by-category`。
3. 尝试以 `LoRA` 形式做最小可行微调。

当前阶段结论是：

- `VoxTell` 原始推理已跑通。
- `prompt engineering` 有局部收益，但不能单独解决问题。
- 当前这版 `LoRA` 训练链路可运行，但验证集表现没有优于 `raw baseline`。
- 当前 `ChestTextGuidedAdapter v1` 也已完成训练与 full val，但同样没有优于 `raw baseline`。
- `ChestTextGuidedAdapter v2` 已经把 `mean_dice` 提升到高于 `raw baseline`，但 `micro_dice` 仍略低。
- 下一步不建议回到 `LoRA`，应继续沿“更强约束的 adapter”方向细化。

## 2. 已完成实验总览

### 2.1 3-case subset: raw vs structured

结果文件：

- [outputs/voxtell_subset_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_subset_metrics.json)

聚合结果：

- `raw`
  - `mean_dice = 0.2697`
  - `micro_dice = 0.1296`
- `structured`
  - `mean_dice = 0.2908`
  - `micro_dice = 0.0102`

解释：

- 在少量 case 上，`structured` 能让部分样本的 `mean_dice` 上升。
- 但 `micro_dice` 极差，说明 `structured` 在总体体素级别上非常不稳定。
- 结论：`structured` 不能直接替代 `raw`。

### 2.2 full public val: raw vs structured

结果文件：

- [outputs/voxtell_val_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_metrics.json)

聚合结果：

- `raw`
  - `mean_dice = 0.2139`
  - `micro_dice = 0.4228`
  - `mean_precision = 0.2493`
  - `mean_recall = 0.3099`
- `structured`
  - `mean_dice = 0.2106`
  - `micro_dice = 0.2234`
  - `mean_precision = 0.2556`
  - `mean_recall = 0.2959`

解释：

- `structured` 的 `mean_dice` 与 `raw` 接近，但 `micro_dice` 明显更差。
- 说明它能改变模型行为，但不能在当前形态下全面胜过原始文本。

### 2.3 full public val: raw vs structured vs hybrid-by-category

结果文件：

- [outputs/voxtell_val_hybrid_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_hybrid_metrics.json)

聚合结果：

- `hybrid`
  - `mean_dice = 0.2273`
  - `micro_dice = 0.3663`
- `raw`
  - `mean_dice = 0.2139`
  - `micro_dice = 0.4228`
- `structured`
  - `mean_dice = 0.2106`
  - `micro_dice = 0.2234`

解释：

- `hybrid` 在 `mean_dice` 上优于 `raw` 和 `structured`。
- 但 `micro_dice` 仍然低于 `raw`。
- 结论：类别路由是有效思路，但仍没有形成稳定优于 `raw` 的主结果。

### 2.4 LoRA fullrun: raw baseline vs raw + LoRA

训练结果目录：

- [outputs/voxtell_lora_mixed_fullrun](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_lora_mixed_fullrun)

验证结果文件：

- [outputs/voxtell_val_lora_raw_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_lora_raw_metrics.json)

训练摘要：

- `steps_completed = 500`
- `best_loss = 0.1424`
- `final_loss = 1.0459`

验证聚合结果：

- `raw baseline`
  - `mean_dice = 0.2139`
  - `micro_dice = 0.4228`
  - `mean_precision = 0.2493`
  - `mean_recall = 0.3099`
- `raw_lora`
  - `mean_dice = 0.1278`
  - `micro_dice = 0.2549`
  - `mean_precision = 0.1188`
  - `mean_recall = 0.4294`

解释：

- LoRA 训练损失下降，说明模型确实学到了一些东西。
- 但验证集上表现为：
  - `recall` 提升
  - `precision` 明显下降
  - `Dice` 变差
- 这说明当前 LoRA 更像学到了“更激进地分”，而不是“更准确地分”。

### 2.5 LoRA(2023): 中途断点版快速验证

训练结果目录：

- [outputs/voxtell_lora_mixed_fullrun_20260413_2023](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_lora_mixed_fullrun_20260413_2023)

状态摘要：

- `step = 220`
- `current_epoch = 2`
- `best_loss_so_far = 0.2574`

快速验证输出目录：

- [outputs/voxtell_val_lora_raw_2023](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_lora_raw_2023)

5-case 快速对比结果：

- `raw baseline`
  - `count = 14`
  - `mean_dice = 0.1653`
  - `micro_dice = 0.1638`
  - `mean_precision = 0.2172`
  - `mean_recall = 0.2134`
- `lora_2023`
  - `count = 14`
  - `mean_dice = 0.0649`
  - `micro_dice = 0.0632`
  - `mean_precision = 0.0483`
  - `mean_recall = 0.3594`

解释：

- `2023` 版本与 fullrun 一样，仍然表现为高召回、低精度、过分割严重。
- 当前没有证据表明 `2023` 是一个真正不同的训练方案版本，更像是同一套训练策略的半程 checkpoint。

### 2.6 ChestTextGuidedAdapter v1: raw baseline vs raw + adapter

训练结果目录：

- [outputs/voxtell_adapter_v1_fullrun](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_adapter_v1_fullrun)

验证结果文件：

- [outputs/voxtell_val_adapter_raw_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_adapter_raw_metrics.json)

训练摘要：

- `steps_completed = 500`
- `trainable_params = 8,132,609 / 448,162,150`
- `best_loss = 0.2012`
- `final_loss = 1.0124`

验证聚合结果：

- `raw baseline`
  - `mean_dice = 0.2139`
  - `micro_dice = 0.4228`
  - `mean_precision = 0.2493`
  - `mean_recall = 0.3099`
- `raw_adapter`
  - `mean_dice = 0.0912`
  - `micro_dice = 0.2422`
  - `mean_precision = 0.0657`
  - `mean_recall = 0.4380`

解释：

- `adapter v1` 和 LoRA 一样，表现为更激进地预测阳性区域。
- `recall` 上升，但 `precision` 明显下降，导致 `Dice` 和 `micro_dice` 均显著差于 `raw baseline`。
- 这说明“单点 gated FiLM 式 adapter”当前也没有解决过分割问题。

### 2.7 ChestTextGuidedAdapter v2: pre-decoder conservative adapter

训练结果目录：

- [outputs/voxtell_adapter_v2_fullrun](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_adapter_v2_fullrun)

验证结果文件：

- [outputs/voxtell_val_adapter_raw_v2_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_adapter_raw_v2_metrics.json)

训练摘要：

- `steps_completed = 500`
- `trainable_params = 6,046,216 / 446,075,757`
- 关键改动：
  - adapter 从 `post_decoder` 前移到 `pre_decoder`
  - 调制改成更保守的分组 gate + 小残差
  - 训练中提高负样本权重和假阳性惩罚

验证聚合结果：

- `raw baseline`
  - `mean_dice = 0.2139`
  - `micro_dice = 0.4228`
  - `mean_precision = 0.2493`
  - `mean_recall = 0.3099`
- `raw_adapter_v2`
  - `mean_dice = 0.2206`
  - `micro_dice = 0.4099`
  - `mean_precision = 0.2435`
  - `mean_recall = 0.3461`

解释：

- `v2` 首次把 `mean_dice` 提升到高于 `raw baseline`。
- 相比 `v1`，`precision` 被明显拉回，过分割问题得到缓解。
- 但 `micro_dice` 仍略低于 baseline，说明总体体素级误报还没有完全压住。
- 当前最准确的结论是：`v2` 已经证明“更强约束的 adapter”方向有效，但还没有全面赢过 baseline。

### 2.8 ChestTextGuidedAdapter v3: category-aware + suppression branch

训练结果目录：

- [outputs/voxtell_adapter_v3_fullrun](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_adapter_v3_fullrun)

验证结果文件：

- [outputs/voxtell_val_adapter_raw_v3_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_adapter_raw_v3_metrics.json)

训练摘要：

- `steps_completed = 500`
- `trainable_params = 7,398,921 / 447,428,462`
- 关键改动：
  - 在 `pre_decoder` 位置保留 v2 的保守 residual gate
  - 将 category-aware 分支改成直接对齐 `ReXGroundingCT` 原始类别编码 `1a..2h`
  - 新增 suppression branch 作为背景抑制信号
- 训练现象：
  - `final_loss = 0.0`
  - `best_loss = 0.0`
  - `zero_loss_ratio = 0.306`
  - `suppression_mean_avg = 0.982`

验证聚合结果：

- `raw baseline`
  - `mean_dice = 0.2139`
  - `micro_dice = 0.4228`
  - `mean_precision = 0.2493`
  - `mean_recall = 0.3099`
- `raw_adapter_v3`
  - `mean_dice = 0.2195`
  - `micro_dice = 0.4113`
  - `mean_precision = 0.2453`
  - `mean_recall = 0.3390`

解释：

- `v3` 延续了 `v2` 的正向趋势，`mean_dice` 和 `recall` 继续高于 baseline。
- 但 `micro_dice` 仍未超过 baseline，且略低于 `v2` 的 `0.4099 -> 0.4113` 这一档改善幅度仍不足以证明 suppression 已经生效。
- 从训练日志看，suppression 分支输出明显偏高，接近“全开”，说明它尚未形成有效的背景抑制能力。
- 当前最准确的结论是：`v3` 不是失败版，但属于“有正向信号的过渡版本”，证明方向没有崩，但还没有完成压制假阳性的目标。

### 2.9 category_ids 修复后的 full val 重跑：v3 / v4

背景说明：

- 在后续排查中确认，旧版 `run_voxtell_val_adapter_raw.py` 在推理阶段没有把 `category_ids` 传进模型。
- 因此此前 `v3` / `v4` 的 full val 结果只反映了“未完整启用 category-aware 条件”的行为。
- 修复推理链后，已重新完成 `v3` 与 `v4` 的 full val 推理与评估。

结果文件：

- [outputs/voxtell_val_adapter_raw_v3_catfix_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_adapter_raw_v3_catfix_metrics.json)
- [outputs/voxtell_val_adapter_raw_v4_catfix_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_adapter_raw_v4_catfix_metrics.json)

验证聚合结果：

- `raw baseline`
  - `mean_dice = 0.2139`
  - `micro_dice = 0.4228`
  - `mean_precision = 0.2493`
  - `mean_recall = 0.3099`
- `raw_adapter_v3_catfix`
  - `mean_dice = 0.2197`
  - `micro_dice = 0.4094`
  - `mean_precision = 0.2433`
  - `mean_recall = 0.3415`
- `raw_adapter_v4_catfix`
  - `mean_dice = 0.2214`
  - `micro_dice = 0.4054`
  - `mean_precision = 0.2358`
  - `mean_recall = 0.3524`

解释：

- 修复 `category_ids` 后，`v3` 与 `v4` 的 category-aware 分支在推理阶段真正生效。
- 修复后的结果说明：
  - `v4` 的 `mean_dice` 和 `recall` 均高于 `v3`；
  - 但 `precision` 进一步下降，`micro_dice` 也没有回到 baseline 之上。
- 当前最准确的结论是：
  - `v4` 是当前 adapter 系列里 case-level overlap 最强的版本；
  - 但它的主要收益来自更高召回，而不是误报被有效压住；
  - 因此 `v4` 仍不能替代 baseline 作为最终主结果。

### 2.10 v4 probe 消融：suppression 是否真正起作用

定位结果文件：

- [问题定位/v4_probe_cases.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/问题定位/v4_probe_cases.json)
- [问题定位/probe_case_comparison_catfix.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/问题定位/probe_case_comparison_catfix.md)
- [问题定位/v4_suppression_behavior_catfix.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/问题定位/v4_suppression_behavior_catfix.md)

probe 聚合结果：

- `v3_catfix`
  - `mean_dice = 0.2352`
  - `micro_dice = 0.5454`
  - `mean_precision = 0.2486`
  - `mean_recall = 0.3384`
- `v4_catfix`
  - `mean_dice = 0.2489`
  - `micro_dice = 0.5348`
  - `mean_precision = 0.2258`
  - `mean_recall = 0.3615`
- `v4_no_suppression`
  - `mean_dice = 0.2469`
  - `micro_dice = 0.5306`
  - `mean_precision = 0.2214`
  - `mean_recall = 0.3644`

解释：

- 在固定 8-case probe 集上，`v4` 明确强于 `v3`，说明 `v4` 不是无效改动。
- `v4 full` 比 `v4 no_suppression` 略好，但差距很小：
  - `mean_dice: 0.2489 vs 0.2469`
  - `micro_dice: 0.5348 vs 0.5306`
- 这说明 suppression 不是完全无效，但贡献偏弱，不是 v4 主效应来源。

本轮最正式的结论：

- `v4` 的主要增益来自整体响应增强和类别条件化带来的召回提升。
- suppression 分支已经参与预测，但其对误报抑制的贡献很小，尚不足以改变模型“高召回、低精度”的整体预测风格。
- 因此，当前主瓶颈不是 suppression 完全失效，而是其作用过弱、监督目标过粗，尚未转化为显著的 voxel-level FP 抑制收益。

### 2.11 v5.1 supervision 重构：full train + full val

本轮目标：

- 不再小修 suppression target，而是直接重构 supervision
- 把 suppression 的监督从简单常数 target 改成：
  - `sample_mode` 感知
  - `patch_fg_fraction` 感知
  - `pred_fg_fraction` / `fp_penalty` 感知
- 同时引入 negative patch 的 `fp-aware penalty`

关键实现：

- [scripts/training/train_voxtell_adapter.py](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/scripts/training/train_voxtell_adapter.py)
- 训练输出：
  - [outputs/voxtell_adapter_v5_1_fullrun](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_adapter_v5_1_fullrun)
- 验证输出：
  - [outputs/voxtell_val_adapter_raw_v5_1](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_adapter_raw_v5_1)
  - [outputs/voxtell_val_adapter_raw_v5_1_metrics.json](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/outputs/voxtell_val_adapter_raw_v5_1_metrics.json)

full val 聚合结果：

- `raw baseline`
  - `mean_dice = 0.2139`
  - `micro_dice = 0.4228`
  - `mean_precision = 0.2493`
  - `mean_recall = 0.3099`
- `adapter v5.1`
  - `mean_dice = 0.2209`
  - `micro_dice = 0.4080`
  - `mean_precision = 0.2372`
  - `mean_recall = 0.3497`

与前几版对比：

- 相比 `v4_catfix`：
  - `mean_dice: 0.2214 -> 0.2209`
  - `micro_dice: 0.4054 -> 0.4080`
  - `mean_precision: 0.2358 -> 0.2372`
  - `mean_recall: 0.3524 -> 0.3497`
- 相比 `v3_catfix`：
  - `mean_dice` 更高
  - `recall` 更高
  - 但 `micro_dice` 仍低于 `0.4094`

训练诊断：

- `zero_loss_count = 0`
- `hard_negative_count = 150`
- `random_negative_count = 122`
- `positive_count = 228`
- `suppression_mean_pos_avg = 0.3826`
- `suppression_mean_neg_avg = 0.3176`
- `avg_suppression_target_positive = 0.8471`
- `avg_suppression_target_negative = 0.1712`

解释：

- 这说明 v5.1 的 supervision 重构**确实改变了 suppression 的训练行为**：
  - 正负样本的 suppression 均值已经明显拉开
  - negative target 也不再接近正样本
- 但最终指标只出现了**有限回调**：
  - `micro_dice` 相比 `v4` 有所恢复
  - `precision` 也略回升
  - 但恢复幅度不大，仍未超过 `v2` / `v3`

本轮结论：

- v5.1 证明“**大改 supervision 是有效方向**”。
- 它不是最终成功版，但相比 v4，已经出现了更符合预期的变化：
  - 更强 supervision 能把一部分 `micro_dice` 和 `precision` 拉回来。
- 这说明当前主问题确实主要在 supervision，而不只是 suppression 模块结构本身。
- 但同时也说明：单靠这版 supervision 重构，仍不足以把结果推回 baseline 之上。

## 3. 为什么当前 LoRA 不再适合作为主线

基于上面的实验，可以得出三个结论：

1. `LoRA` 训练链路已经验证为“可运行”，不需要再证明它能不能训。
2. 当前这套 LoRA 方案在验证集上稳定表现为：
   - `recall` 上升
   - `precision` 下降
   - `Dice` 下降
3. `ChestTextGuidedAdapter v1` 也复现了类似趋势，说明当前问题不只是 LoRA 形式本身，更像是训练目标对假阳性约束不足。
4. `ChestTextGuidedAdapter v2` 通过更早插入、更保守调制、更强负样本约束，已经把 `mean_dice` 顶过 baseline，说明问题方向判断是对的。
5. 后续最值得做的不是回到 LoRA，而是继续压低 `micro` 层面的假阳性。
6. `adapter v3 / v4` 已完成闭环训练、修复推理链后的 full val 重跑，以及 probe 消融分析；当前已明确主问题是“召回提升明显，但 suppression 对误报抑制贡献偏弱”。

因此，当前最合理的决定是：

- 将 LoRA 结论固定为“已验证但不作为主线”。
- 将 `adapter v1` 结论固定为“已实现、已验证，但不作为最终方案”。
- 将后续主线切换为更强约束、更强调 precision 的结构改进。

## 4. 当前明确结论

### 可以保留的结论

- `raw prompt` 仍然是最稳的 baseline。
- `hybrid-by-category` 是一个有效方向，但还不是最终答案。
- 当前只有“更强约束的 adapter”开始出现稳定正向信号。
- `adapter v3 / v4` 说明“类别感知 + 抑制分支”值得继续，但当前实现主要提升的是召回，仍没有把 `micro_dice` 推过 baseline。

### 不建议继续做的事

- 不建议继续把 `LoRA` 作为主结果方向。
- 不建议继续把当前 `adapter v1` 直接扩成更多轮训练。
- 不建议继续优先补 `full LoRA` 更多训练时长。
- 不建议继续沿用“只加轻量调制、损失不变”的同类路线。
- 不建议把当前 `v4` 直接当成最终主结果提交，应先解释为什么召回提升没有转化为 `micro_dice` 提升。

### 建议切换的新主线

- 基于 `改进方案.pdf`，优先做“胸部专用融合模块 / Text-Guided Adapter”。
- 核心目标是：
  - 在不重写整套 VoxTell 的前提下
  - 明确增强文本与胸部 3D 特征的融合方式
  - 提高 precision，抑制当前的过分割倾向
  - 同时引入更明确的假阳性约束，而不是只提高阳性响应

## 5. 下一步文档

请继续查看：

- [胸部专用融合模块_最小实现方案.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/实验记录/胸部专用融合模块_最小实现方案.md)
