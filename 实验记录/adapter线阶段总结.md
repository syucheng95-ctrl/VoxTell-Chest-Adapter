# Adapter 线阶段总结

本文件用于给当前 adapter 线做阶段性收尾，总结我们做过什么、真正学到了什么、哪些坑反复出现，以及这些结论如何迁移到下一阶段的四条主线：数据、监督、训练、结构。

## 1. 关键版本总表

| 演进阶段 | 版本 | Mean Dice | Micro Dice | Precision | Recall | 定位与战果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 基准 | Baseline | 0.2139 | 0.4228 | 0.2493 | 0.3099 | 原始官方模型，参考线 |
| 第一代 | v2 | 0.2206 | 0.4099 | 0.2435 | 0.3461 | 第一版稳定超过 baseline 的 adapter，证明“强约束 adapter”方向成立 |
| 第一代 | v4_catfix | **0.2214** | 0.4054 | 0.2358 | **0.3524** | “激进派”：`mean_dice / recall` 峰值，但 FP 偏多 |
| 第二代 | v5.1 | 0.2209 | 0.4080 | 0.2372 | 0.3497 | supervision / FP-aware 重构成功版 |
| 第三代 | v6.4_stage3_fix | 0.2141 | 0.4234 | **0.2586** | 0.3089 | “洁癖派”：candidate+risk 架构落地后 precision 峰值 |
| 第三代 | v6.5 | 0.2133 | 0.4242 | 0.2509 | 0.3069 | `v6` 稳健主干，`micro` 强点 |
| 第四代 | v6.6 | 0.2126 | **0.4269** | 0.2548 | 0.3016 | “极端派”：门控过硬，`micro` 最高但 recall 最低 |
| 第四代 | v6.6b | 0.2146 | 0.4214 | 0.2565 | **0.3124** | “松绑版”：救回 recall/mean，但丢了 `micro` 优势 |
| 当前 | v6.7 | 0.2146 | 0.4215 | 0.2565 | 0.3123 | “均衡派”：基本等价 `v6.6b`，没有形成新 Pareto 前沿 |

## 2. 我们一共做了什么版本

### 2.1 早期 adapter 线

- `v1`
  - 目标：验证轻量 text-guided adapter 能否替代更重的微调方案。
  - 结果：明显走向高 recall、低 precision，过分割严重。
  - 结论：单点 gated FiLM 式 adapter 不足以约束假阳性。

- `v2`
  - 目标：把 adapter 提前到 `pre-decoder`，采用更保守的分组 gate + 小残差。
  - 结果：首次把 `mean_dice` 提到 baseline 之上。
  - 结论：更强约束、更早接入的位置是有效方向。

### 2.2 category-aware / suppression 线

- `v3`
  - 目标：加入 category-aware 分支和 suppression branch。
  - 结果：`mean_dice / recall` 继续提升，但 `micro_dice` 仍低于 baseline。
  - 结论：类别感知有帮助，但 suppression 还没有真正形成强有效的 FP 抑制。

- `v4 / v4_catfix`
  - 目标：修正 category 使用链路，强化类别感知与 suppression 行为。
  - 结果：adapter 系列里 `mean_dice / recall` 最强，但 `precision / micro` 仍不够。
  - 结论：主要收益来自更高召回，不是 suppression 真正压住了误报。

### 2.3 supervision 重构线

- `v5.1`
  - 目标：重构 suppression supervision，引入 sample-mode / patch-FG / pred-FG / FP-aware 的目标设计。
  - 结果：训练行为明显改善，`precision / micro` 有所回调，但没有全面超 baseline。
  - 结论：监督设计比单纯加结构更有效，但仍然不足以突破上限。

- `v5.2 / v5.3`
  - 目标：继续做更干净 suppression、减弱 category。
  - 结果：更保守，但没有超过 `v5.1`。
  - 结论：过度“去 category / 强 suppression”会让模型更谨慎，但不一定更好。

### 2.4 candidate / risk / refine 线

- `v6`
  - 目标：把 suppression 升级成显式 FP risk head。
  - 结果：高 recall、massive FP。
  - 结论：显式 risk 方向对，但第一版行为过激。

- `v6.1 / v6.2 / v6.3`
  - 目标：逐步从 feature-space 修补走向 output/logit-space rectification。
  - 结果：出现了一些新行为，但稳定性和最终 full val 表现不够。
  - 结论：logit-level rectification 比 feature-level 更有希望，但训练设计决定成败。

- `v6.4 / v6.4_stage3_fix`
  - 目标：工程上稳定落地 `candidate + risk` 双头。
  - 结果：修复训练后回到 baseline 上方，结构成立。
  - 结论：candidate/risk 作为“先提候选，再扣误报”的职责拆分是有效的。

- `v6.5`
  - 目标：在 `v6.4` 上加入 bounded serial refine。
  - 结果：当前最稳的 `micro_dice` 主线。
  - 结论：这条线最适合做低 FP / 高 micro 的稳健结构基线。

### 2.5 MSAR 局部化尝试

- `v6.6`
  - 目标：做 `Masked Risk + Uncertainty Refine`。
  - 结果：`micro / precision` 上升，但 `recall / mean_dice` 下降。
  - 结论：方向没错，但门控太硬，模型过度保守。

- `v6.6b`
  - 目标：用更少参数、更软的方式保留 `Masked Risk + Uncertainty Refine`。
  - 结果：`mean_dice / recall / precision` 回升，但 `micro` 掉回去。
  - 结论：证明局部化思路不是错的，但很难同时保住 `micro`。

- `v6.7`
  - 目标：以 `v6.5` 为底，做更温和的 selective risk，并调整 `stage2` 只训 `risk`。
  - 结果：基本等价于 `v6.6b`，没有形成新的 Pareto 前沿。
  - 结论：adapter 结构小改已经很难继续拉开差距。

## 3. 每条路线真正学到了什么

- adapter 不是没用，但更像“行为调节器”
  - 它能明显改变模型风格：更偏 recall、更偏 precision、更偏 micro、更偏 mean。
  - 但很难把所有指标一起显著抬高。

- 更强结构约束比“更强文本调制”更靠谱
  - 真正跑通并形成正向信号的，是更强约束的 adapter 设计。
  - 当前瓶颈不在“再让文本更强地主导”，而在“怎么更精准地处理 FP 与边界”。

- supervision 的价值很高
  - 多轮结果说明：单纯加结构不一定带来稳定收益，但监督改造通常更容易改变真实行为。
  - `v5.1` 是非常关键的证据。

- 训练策略不是附属项，而是结构成败的一部分
  - `stage` 切换、冻结范围、联合训练时机，对结果影响非常大。
  - 后续不能再把训练只当“配套脚本”。

- `candidate / risk / refine` 职责拆分是 adapter 线最有价值的遗产
  - `candidate` 负责提阳性
  - `risk` 负责扣误报
  - `refine` 负责边界级修正

## 4. 哪些坑重复出现

- “提 recall 但 precision 崩”
  - LoRA、`v1`、早期激进版本都掉进过这个坑。
  - 教训：只要结构本质上是在放大阳性响应，而没有精准 FP 约束，最后大概率就是高 recall、低 precision。

- suppression / risk 学成“整体保守化”
  - 抑制模块经常学成整体缩放，而不是精准抑制。
  - 教训：抑制模块必须局部化、条件化，否则很容易误伤 TP。

- 训练统计变好，但 full val 不一定变好
  - 多次出现 suppression mean 拉开了，但 full val 没兑现。
  - 教训：训练诊断不能直接等价成最终收益。

- 新结构收益被训练设计掩盖
  - `v6.4` 的经验最典型。
  - 教训：结构实验如果没有配套训练策略，结论很容易是假负例。

- 经验参数过多会掩盖真正问题
  - `v6.6 -> v6.6b -> v6.7` 已经很说明这一点。
  - 教训：当版本差异已进入窄区间摆动时，继续叠手工 gate 和经验系数，只会让结论更模糊。

## 5. 哪些结论可以迁移到后面四条主线

### 5.1 数据

- 当前很多版本之间的差距很小，说明数据覆盖和采样很可能才是更大的上限因素。
- 后续数据线要重点看：
  - 小病灶覆盖
  - 稀有类别覆盖
  - hard negative 构造
  - 训练/验证分布一致性

### 5.2 监督

- supervision 的改动通常比结构小修更扎实。
- 后续优先值得尝试：
  - FP-aware supervision
  - boundary-aware supervision
  - lesion-size aware weighting
  - per-category / per-pattern 的差异化监督

### 5.3 训练

- stage 设计、冻结/解冻、负样本采样节奏，对结果影响非常大。
- 后续训练线应该正式研究：
  - 不同阶段训哪些模块
  - risk/refine 什么时候进入
  - negative sampling schedule 怎么设计
  - 是否需要部分解冻主干

### 5.4 结构

- 结构改造仍然值得做，但必须更克制。
- 后面结构实验最好每次只回答一个问题：
  - 候选区局部化到底有多大收益
  - risk 和 refine 是否该进一步解耦
  - scale-aware 是否真能保护小目标
  - logit-level 改造和 supervision 如何配套

## 6. 当前总体判断

- adapter 线不是失败的。
- 它已经完成了两个关键任务：
  - 找到了不同指标之间可控的 Pareto 点
  - 帮我们定位出更值得投入的上层问题：数据、监督、训练、结构

但也要承认：

- 在当前数据规模和训练设定下，adapter 线大概率已经接近“局部最优微调”上限。
- 后面即使继续榨，也更像是 `0.002 ~ 0.008` 量级的优化。
- 真正更值得投入的，是后面四条主线。

## 7. 建议的下一阶段顺序

1. 保留 `v6.5` 作为 `micro / 低 FP` 结构基线
2. 保留 `v5.1` 作为 `mean / recall` 监督基线
3. 以这两个锚点为对照，正式启动四条主线对比：
   - 数据
   - 监督
   - 训练
   - 结构
