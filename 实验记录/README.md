# 实验记录

本目录用于沉淀本项目已经完成的实验、当前结论、失败原因与下一步决策，避免后续重复试错。

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

因此，当前最合理的决定是：

- 将 LoRA 结论固定为“已验证但不作为主线”。
- 将 `adapter v1` 结论固定为“已实现、已验证，但不作为最终方案”。
- 将后续主线切换为更强约束、更强调 precision 的结构改进。

## 4. 当前明确结论

### 可以保留的结论

- `raw prompt` 仍然是最稳的 baseline。
- `hybrid-by-category` 是一个有效方向，但还不是最终答案。
- 当前只有“更强约束的 adapter”开始出现稳定正向信号。

### 不建议继续做的事

- 不建议继续把 `LoRA` 作为主结果方向。
- 不建议继续把当前 `adapter v1` 直接扩成更多轮训练。
- 不建议继续优先补 `full LoRA` 更多训练时长。
- 不建议继续沿用“只加轻量调制、损失不变”的同类路线。

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
