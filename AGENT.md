# AGENT.md

本文件面向进入本项目协作的智能体或开发者，记录当前主线、已验证结论、代码入口与工程约定。

## 1. 项目目标

本项目基于 `VoxTell + nnUNet`，面向胸部 CT finding segmentation，目标是在保住 case-level `mean_dice` 的同时，把 voxel-level `micro_dice` 拉回 baseline 以上。

当前共识：
- 主干 `nnUNet` 不是主战场。
- 纯重型 text-grounding 路线在当前数据规模下不稳定，不再作为主推主线。
- 主线回到 adapter 系列，并继续围绕 `FP / micro_dice` 做结构和训练升级。

## 2. 当前结果与主线判断

已验证关键结果：
- baseline：`mean_dice = 0.2139`，`micro_dice = 0.4228`
- `v2`：`mean_dice = 0.2206`，`micro_dice = 0.4099`
- `v4_catfix`：`mean_dice = 0.2214`，`micro_dice = 0.4054`
- `v5.1`：`mean_dice = 0.2209`，`micro_dice = 0.4080`
- `v6.4_stage3_fix`：`mean_dice = 0.2141`，`micro_dice = 0.4234`
- `v6.5`：`mean_dice = 0.2133`，`micro_dice = 0.4242`

阶段性结论：
- `v4` 是 adapter 系列里 `mean_dice / recall` 最强的一版，但偏激进。
- `v5.1` 通过更强 supervision / FP-aware 训练把 `precision` 和 `micro_dice` 稍微拉回来了。
- `v5.2`（去 category + 更干净 suppression）和 `v5.3`（弱 category + 软 suppression）都没有超过 `v5.1`。
- `v6.4` 的原始 fullrun 失败主要是 `stage3` 训练设计问题，不再应简单归因为结构失效。
- `v6.4_stage3_fix` 证明 `v6.4` 在修正训练后可以稳定回到 baseline 上方。
- `v6.5` 在 `v6.4` 稳定主干上加入弱 refine/suppress 头后，没有翻车，并拿到了当前最高 `micro_dice`。
- 之前的 raw-only / token-fusion / 3-stage grounding 大改在 full val 上明显退化，已降级为探索线。

当前主线判断：
- 如果按 `mean_dice` 看，当前冠军仍是 `v4_catfix` / `v5.1` 这一档。
- 如果按 `micro_dice` 看，当前冠军是 `v6.5`。
- 如果按“综合稳定性 + 低 FP + 不低于 baseline 的 voxel 指标”看，当前最值得继续沿着做的是 `v6.5`，其次是 `v6.4_stage3_fix`。

## 3. v6 系列当前进度

`v6` 的目标不是继续做重型文本融合，而是在旧 adapter 主线内部重写“提候选 / 扣误报”的职责分工。

### A：suppression -> explicit FP risk head
- 不再只是乘性抑制特征。
- 单独建一条 `risk head`，显式预测 false-positive risk。

### B：single-shot adapter -> candidate + reject/refine two-stage adapter
- 不再一次性用文本对特征做加减法。
- 改成两步：
  - `candidate_head`：先提候选区域
  - `risk_head`：再减去高风险误报

阶段性版本结论：
- `v6`：并联 candidate/risk residual，full val 证明是高 recall / massive FP，已淘汰。
- `v6.1`：弱串联但仍停留在 feature-space 修补，和旧 `v6` 差异不够大，已淘汰。
- `v6.2`：输出空间 rectification 第一次出现有效新行为，但 `200 step` probe 已显示后段开始失稳。
- `v6.3`：把 `v6.2` 的后段不稳压住了，但基本收回到 `v5.2/v5.3` 的保守区间，没有形成更强平衡。
- **`v6.4`：第一版工程上成立的 logit-level candidate/risk 双头。**
  - 保留 `logit-level candidate + risk` 双头
  - 弱 `category bias` 只作用于 candidate
  - 训练改成 `candidate -> risk -> short joint` 的冻结式专训
  - 原始 fullrun 因 `stage3` 解冻范围过大，出现高 recall / massive FP
  - 修正后 `v6.4_stage3_fix`：`mean_dice = 0.2141`，`micro_dice = 0.4234`
- **`v6.5`：当前最新探索版。**
  - 在 `v6.4` 上增加 bounded serial `refine` 头
  - 保留 `stage3` 只训 adapter，不再碰 `project_text_embed`
  - fullrun 结果：`mean_dice = 0.2133`，`micro_dice = 0.4242`
  - 当前结论：没有把 recall 拉高，但把低 FP / 高 micro 的稳健线又往前推了一点

## 4. 关键代码入口

### v6.4 / v6.5 结构
- `VoxTell/voxtell/model/chest_candidate_risk_adapter.py`
  - `ChestCandidateRiskAdapter`
  - `logit_candidate_head + logit_risk_head`
  - 弱 category bias 只进入 candidate proposal
- `VoxTell/voxtell/model/v6_adapter_model.py`
  - `VoxTellV64AdapterModel`
  - 保留原 VoxTell decoder/query 主干
  - 最终在 output/logit 空间做 `candidate - risk` 组合
- `VoxTell/voxtell/model/chest_candidate_risk_refine_adapter.py`
  - `ChestCandidateRiskRefineAdapter`
  - 在 `candidate + risk` 之外，再加一条弱 `refine` 抑制头
- `VoxTell/voxtell/model/v65_adapter_model.py`
  - `VoxTellV65AdapterModel`
  - 最终在 output/logit 空间做 `candidate - risk - weak_refine`

### 训练与验证入口
- `scripts/training/train_voxtell_adapter.py`
  - 当前支持 `--adapter-version v6_4 / v6_5`
  - `stage1` 主训 candidate
  - `stage2` 主训 risk（`v6.5` 同时带 refine）
  - `stage3` 只做短 joint 收尾
- `scripts/experiments/run_voxtell_val_adapter_raw.py`
  - 当前支持 `--adapter-version v6_4 / v6_5`

### 历史主线结果
- `outputs/voxtell_val_adapter_raw_v2_metrics.json`
- `outputs/voxtell_val_adapter_raw_v4_catfix_metrics.json`
- `outputs/voxtell_val_adapter_raw_v5_1_metrics.json`
- `outputs/voxtell_val_adapter_raw_v5_2_pilot_200_metrics.json`

## 5. v6.4 / v6.5 当前验证状态

当前已完成：
- `py_compile`
- 标准 patch 单次前向
- `v6.4 200-step pilot`
- `v6.4 5-case probe val`
- `v6.4 fullrun / step100 / step200 / stage3_fix`
- `v6.5 fullrun / full val / evaluate`

当前关键结论：
- `v6.4` 把 `v6.2@200` 的后段不稳定性压住了
- `v6.4` 原始 fullrun 的问题被定位为 `stage3` 训练设计问题，不是纯结构失效
- `v6.4_stage3_fix` 和 `v6.5` 都已经证明可以稳定跑完 full val，不再是“不能 full”的状态
- `v6.5` 当前是 `micro_dice` 冠军，但还没有把 `mean_dice / recall` 拉到 `v5.1` 那一档

## 6. 训练与缓存约定

### Prompt cache
- 能复用 `train_prompt_embeddings.pt` 就必须复用。
- 统一使用：
```powershell
python scripts\training\train_voxtell_adapter.py --output-dir outputs\new_run --resume-prompt-cache-from outputs\some_previous_run
```

### Smoke
- `smoke` 只做链路验证。
- 跑完即删，不长期保留。

### 保存逻辑
- 默认不保存 `training_state.pt`。
- 中间里程碑保存轻量 `adapter_step_xxxx.pt`。
- 默认最多保留最近 3 个里程碑。
- 如果最后一步正好是里程碑：
  - 不再同时保留 `adapter_step_xxxx.pt + adapter_state.pt`
  - 会直接转成单份 `adapter_state.pt`

## 7. 空间管理现状

当前 `outputs` 已清理到约 `14.6 GB`。

保留策略：
- 保留主结果：
  - `v2`
  - `v4`
  - `v5.1`
  - `v5.2_200`
  - `v6.4_stage3_fix`
  - `v6.5`
- 已删除明显失败或过期的：
  - `3stage` 系列大目录
  - 旧 raw-only queryschedule 系列
  - 重复 smoke 目录

## 8. 下一步建议

1. `v5.1` 仍是 `mean_dice / recall` 主线对照。
2. `v6.5` 是当前 `micro_dice / 低 FP` 冠军。
3. 如果继续做 `v6`，重点不再是“先防翻车”，而是：
   - 怎么在保持 `v6.5` 低 FP 的前提下，把 recall 再拉回去。

## 9. 一句话原则

当前不再追求“更强文本主导”，而是追求：
**先提候选，再在输出空间显式识别并扣除误报风险。**
