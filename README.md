# VoxTell-Chest

基于 `VoxTell + nnUNet` 的胸部 CT finding segmentation 项目。项目重点不是重写视觉主干，而是在保持主干稳定的前提下，迭代文本引导 adapter，并尽量解决 `micro_dice` 被 false positive 拖累的问题。

## 当前状态

当前已经跑出的主结果：

| 版本 | Mean Dice | Micro Dice | 说明 |
| --- | ---: | ---: | --- |
| Baseline | 0.2139 | 0.4228 | 原始 VoxTell |
| v2 | 0.2206 | 0.4099 | 第一版有效 adapter 主线 |
| v4_catfix | 0.2214 | 0.4054 | recall / mean dice 最强，但偏激进 |
| v5.1 | 0.2209 | 0.4080 | supervision / FP-aware 训练更成熟 |
| v5.2 (200 step) | 0.1925 | 0.3124 | 更干净但 recall 掉太多 |
| v6.4_stage3_fix | 0.2141 | 0.4234 | 修复 stage3 后稳定回到 baseline 上方 |
| v6.5 | 0.2133 | 0.4242 | 当前最高 micro_dice，低 FP 稳健线 |

阶段性结论：
- `v5.1` 仍是当前 `mean_dice / recall` 最稳的 adapter 主线。
- `v5.2 / v5.3` 证明“更强 suppression、更弱 category”会让模型更保守，但没有超过 `v5.1`。
- 更重的 token-grounding / raw-only 文本融合路线在当前数据规模下 full val 不成立，已经降级为探索线。
- `v6.4` 原始 fullrun 的失败已经定位为 `stage3` 训练设计问题，不再应简单看作结构完全失效。
- `v6.5` 是当前 `micro_dice` 冠军，但还没有在 `mean_dice / recall` 上超过 `v5.1`。

## 当前探索线：v6.4 / v6.5

`v6.4` 不再继续做重型 text-grounding，而是在旧 adapter 思路上做一次中等力度的结构升级。

### A. suppression -> explicit FP risk head
- 原来的 suppression 主要是 feature 级控制项。
- `v6` 系列把它升级成显式的 `risk head`，直接预测 false-positive risk。

### B. single-shot adapter -> candidate + reject/refine two-stage adapter
- 原来 adapter 更像一次性调特征。
- `v6.4` 当前实现是：
  - `candidate head`：在 **logit 空间**提出候选
  - `risk head`：在 **logit 空间**做局部误报审查
  - 弱 `category bias` 只作用于 candidate，不再直接回写 feature

### 当前实现形式

`v6.4` 的 adapter 语义是：

```text
final_logit = candidate_logit - lambda * risk_logit
```

其中：
- `candidate_logit` 对应“先提候选”
- `risk_logit` 对应“再扣高风险误报”

`v6.5` 在此基础上再加一条很弱的 refine/suppress 头：

```text
final_logit = candidate_logit - lambda * risk_logit - weak_refine_bias
```

它的目标不是进一步压结构，而是更平滑地回拉 risky positive。

## 代码入口

### v6.4 / v6.5 核心结构
- `VoxTell/voxtell/model/chest_candidate_risk_adapter.py`
  - `ChestCandidateRiskAdapter`
  - 实现 logit-level candidate/risk 双头 adapter
- `VoxTell/voxtell/model/v6_adapter_model.py`
  - `VoxTellV64AdapterModel`
  - 保留原 VoxTell decoder/query 主干
- `VoxTell/voxtell/model/chest_candidate_risk_refine_adapter.py`
  - `ChestCandidateRiskRefineAdapter`
  - 在 `candidate + risk` 外增加 bounded refine/suppress 头
- `VoxTell/voxtell/model/v65_adapter_model.py`
  - `VoxTellV65AdapterModel`

### 训练与验证
- `scripts/training/train_voxtell_adapter.py`
  - 当前支持 `--adapter-version v6_4 / v6_5`
  - 使用 `candidate -> risk -> short joint` 的冻结式训练节奏
- `scripts/experiments/run_voxtell_val_adapter_raw.py`
  - 当前支持 `--adapter-version v6_4 / v6_5`
- `scripts/analysis/evaluate_voxtell_val_adapter_raw.py`
  - full val 后用于汇总指标

## 当前工程状态

`v6.4 / v6.5` 当前是：
- **结构代码已完成**
- **训练与验证脚本已接入**
- **静态语法检查已通过**
- **标准 patch 前向已通过**
- **`v6.4 fullrun/stage3_fix` 已完成**
- **`v6.5 full train + full val + evaluate` 已完成**

已确认通过的检查：
- `py_compile`
- 标准 patch 单次前向：
  - `FORWARD_OK (1, 1, 192, 192, 192)`
  - `RISK_OK`
  - `CAND_OK`

当前结论：
- `v6.4` 的原始 fullrun 失败主因是 `stage3` 训练设计问题
- `v6.4_stage3_fix` 修正后：`micro_dice = 0.4234`
- `v6.5` 进一步拿到：`micro_dice = 0.4242`
- 但这条线目前仍偏保守，`mean_dice / recall` 还没有超过 `v5.1`

## 常用命令

### 训练
```powershell
python scripts\training\train_voxtell_adapter.py --output-dir outputs\some_run --resume-prompt-cache-from outputs\some_previous_run
```

### 验证
```powershell
python scripts\experiments\run_voxtell_val_adapter_raw.py --adapter-run-dir outputs\some_run --output-dir outputs\some_val_dir
python scripts\analysis\evaluate_voxtell_val_adapter_raw.py --adapter-dir outputs\some_val_dir --output outputs\some_metrics.json
```

## Prompt Cache 与存储约定

- 优先复用 `train_prompt_embeddings.pt`
- `smoke` 只做链路验证，跑完即删
- 默认不保存 `training_state.pt`
- 里程碑只保留轻量 `adapter_step_xxxx.pt`
- 如果最后一步刚好是里程碑，不再双存 `adapter_step_xxxx.pt + adapter_state.pt`

## 下一步

当前最合理的顺序是：
1. `v5.1` 仍作为 `mean_dice / recall` 主线对照
2. `v6.5` 作为当前 `micro_dice / 低 FP` 冠军继续推进
3. 如果继续做 `v6`，重点不是再压 FP，而是在不破坏 `v6.5` 稳定性的前提下把 recall 拉回来
