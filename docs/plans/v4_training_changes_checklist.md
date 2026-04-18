# v4 训练改动具体实现清单

本文档只聚焦 **v4 的训练侧改动**，默认前提是：

- 保留 `ChestTextGuidedAdapter v3` 的主干框架
- 暂不大改 `pre_decoder + conservative residual gate`
- suppression 架构细化方案后续再单独讨论

当前目标不是“重写模型”，而是先验证：

1. suppression 分支能否通过更明确的训练约束学成有效抑制
2. 负样本采样是否是当前 `micro_dice` 上不去的重要原因
3. `zero loss` 过多是否在浪费训练步


## 1. v4 核心目标

v4 的训练改动只服务三个目标：

- 降低 `zero loss` 负样本比例
- 让 suppression 在正负 patch 上拉开差异
- 在尽量不牺牲 `mean_dice` 的情况下，继续提升 `micro_dice`


## 2. 必做改动

### 2.1 增加 suppression loss

目的：

- 把 suppression 从“间接学习”改成“显式受监督”
- 防止 suppression 长期贴近 `1.0`

实现原则：

- 以 patch 为单位构造 suppression target
- 当前版本先做最简单的 patch-level supervision

建议规则：

- `positive patch` -> target = `1`
- `negative patch` -> target = `0`

建议实现形式：

- 从 adapter 读取 `suppression_mean`
- 新增：
  - `suppression_loss = BCE(suppression_mean, suppression_target)`

总损失建议改为：

```python
loss = seg_loss + lambda_supp * suppression_loss
```

建议初始超参数：

- `lambda_supp = 0.1`

建议新增 CLI 参数：

- `--suppression-loss-weight`


### 2.2 重做负样本采样比例

目的：

- 减少纯空 easy negative
- 增加 hard negative 对训练的贡献

当前参数：

- `positive_patch_prob = 0.35`
- `hard_negative_patch_prob = 0.35`

建议改为首版：

- `positive_patch_prob = 0.40`
- `hard_negative_patch_prob = 0.50`
- random negative 剩余 `0.10`

建议保留 CLI 参数可调，不写死。


### 2.3 降低 pure-empty negative 的训练占比

目的：

- 避免训练步浪费在已经学会的纯背景 patch 上

建议策略二选一：

方案 A：

- 若 negative patch 满足：
  - `patch_fg_fraction == 0.0`
  - 且 loss 已接近 0
- 则直接跳过该 step 的 backward

方案 B：

- 不跳过
- 但降低这类样本的采样概率或 loss 权重

当前更推荐：

- **优先做方案 A**

原因：

- 最简单
- 最容易解释
- 最适合本地机制验证


### 2.4 强化训练日志

v4 必须补充这些统计项，否则结果仍然难以解释。

建议在 `train_metrics.json` 中增加：

- `suppression_mean_pos_avg`
- `suppression_mean_neg_avg`
- `zero_loss_ratio`
- `zero_loss_neg_count`
- `hard_negative_count`
- `random_negative_count`
- `positive_count`
- `avg_patch_fg_positive`
- `avg_patch_fg_negative`
- `avg_fp_penalty_positive`
- `avg_fp_penalty_negative`

必要时记录每 step 字段：

- `suppression_target`
- `suppression_loss`
- `is_zero_loss`


### 2.5 轻微加强负样本误报惩罚

目的：

- 继续向“压 FP”方向施压
- 但不要一次把 recall 打崩

建议首版：

- `negative_fp_weight: 0.25 -> 0.35`
- `negative_loss_scale: 2.0 -> 2.5`

建议不要同时大幅增加所有负样本权重。


## 3. 暂不改动

为了保证可归因，v4 首轮训练改动暂时不要碰这些内容：

- 不改 `pre_decoder` 插入点
- 不改主 residual gate 结构
- 不改 category-aware 为每类独立头
- 不上 token-level cross-attention
- 不上 voxel-level / 3D spatial attention
- 不重写 VoxTell 主干


## 4. 代码实现入口

主要涉及文件：

- `scripts/training/train_voxtell_adapter.py`
- `VoxTell/voxtell/model/chest_text_guided_adapter.py`

建议修改分工：

### 4.1 `chest_text_guided_adapter.py`

新增或保证可读出的中间量：

- `last_suppression_mean`
- 如有需要，新增：
  - `last_suppression_tensor`

要求：

- 训练脚本能够直接读取 suppression 的当前值
- 尽量避免在训练脚本中重复 adapter 内部逻辑


### 4.2 `train_voxtell_adapter.py`

必须修改的点：

1. 新增 suppression loss 计算
2. 新增 suppression target 逻辑
3. 修改总 loss 组合
4. 调整 patch 采样概率默认值
5. 处理 zero-loss pure-empty negatives
6. 增加训练日志字段
7. 增加新的 CLI 参数

建议新增 CLI 参数列表：

- `--suppression-loss-weight`
- `--skip-easy-empty-negatives`
- `--easy-negative-loss-threshold`


## 5. 推荐默认参数（v4 首版）

建议首版默认值：

- `positive_patch_prob = 0.40`
- `hard_negative_patch_prob = 0.50`
- `negative_fp_weight = 0.35`
- `negative_loss_scale = 2.5`
- `suppression_loss_weight = 0.1`
- `skip_easy_empty_negatives = true`
- `easy_negative_loss_threshold = 1e-4`


## 6. v4 首轮成功判据

不是直接看最终分数，而是先看训练现象是否变对。

首轮希望看到：

1. `suppression_mean_neg_avg < suppression_mean_pos_avg`
2. `zero_loss_ratio` 明显低于 v3 的 `0.306`
3. `micro_dice` 高于 v3 的 `0.4113`
4. `mean_dice` 不显著低于 v2/v3

如果前三条都没有改善，
说明问题不只在训练侧，
后续才值得继续讨论 suppression 架构本身。


## 7. 一句话实施策略

v4 第一轮不是追求大改，而是：

**用最小训练侧改动，验证 suppression 能不能被真正训练成“负样本该低、正样本可高”的有效抑制器。**
