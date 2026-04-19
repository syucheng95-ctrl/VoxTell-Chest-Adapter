# VoxTell-Chest

基于 `VoxTell + nnUNet` 的胸部 CT finding segmentation 项目。当前重点不是重写视觉主干，而是在保持主干稳定的前提下，围绕 adapter、监督和训练策略逐步优化 `mean_dice / micro_dice / precision / recall` 的平衡。

## 当前总览

### 版本演进总表

| 演进阶段 | 版本 | Mean Dice | Micro Dice | Precision | Recall | 定位与战果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 基准 | Baseline | 0.2139 | 0.4228 | 0.2493 | 0.3099 | 原始官方模型，参考线 |
| 第一代 | v2 | 0.2206 | 0.4099 | 0.2435 | 0.3461 | 第一版稳定超过 baseline 的 adapter，证明“强约束 adapter”方向成立 |
| 第一代 | v4_catfix | **0.2214** | 0.4054 | 0.2358 | **0.3524** | “激进派”：`mean_dice / recall` 最强，但 FP 偏多 |
| 第二代 | v5.1 | 0.2209 | 0.4080 | 0.2372 | 0.3497 | supervision / FP-aware 重构成功版 |
| 第三代 | v6.4_stage3_fix | 0.2141 | 0.4234 | **0.2586** | 0.3089 | “洁癖派”：candidate+risk 架构落地后 precision 峰值 |
| 第三代 | v6.5 | 0.2133 | 0.4242 | 0.2509 | 0.3069 | `v6` 稳健主干，长期可复现的 `micro` 强点 |
| 第四代 | v6.6 | 0.2126 | **0.4269** | 0.2548 | 0.3016 | “极端派”：门控过硬，`micro` 最高但 recall 最低 |
| 第四代 | v6.6b | 0.2146 | 0.4214 | 0.2565 | **0.3124** | “松绑版”：救回 recall/mean，但丢了 `micro` 优势 |
| 当前 | v6.7 | 0.2146 | 0.4215 | 0.2565 | 0.3123 | “均衡派”：基本等价 `v6.6b`，没有形成新 Pareto 前沿 |

### 当前判断

- `v4_catfix / v5.1` 仍是 `mean_dice / recall` 最强对照。
- `v6.6` 有最高 `micro_dice`，但行为过于保守，不适合作为唯一主线。
- `v6.5` 仍是最适合继续当结构基线的 `v6` 版本。
- `v6.6b / v6.7` 说明 adapter 局部化方向成立，但 adapter 线本身已经接近上限。

## adapter 线结论

这条线不是失败的。它已经完成了两件很重要的事：

1. 找到了多个不同指标侧重下的 Pareto 点
2. 帮我们定位出真正更值得投入的上层问题：数据、监督、训练、结构

当前更准确的判断是：

- adapter 能稳定改行为风格
- 但很难把所有指标一起显著推高
- 在当前数据规模和训练设定下，它更像“决策边界调节器”，而不是“表征重建器”

## 为什么不同版本的 adapter 挂载位置不一样

`v1 -> v6.6` 这条线里，adapter 的位置不是随手改的，而是随着问题认识不断后移。整体可以概括成三步：

### 1. 早期版本：先在 feature/query 空间动手

- `v1` 的核心想法是用轻量 text-guided adapter 替代更重的微调。
- `v2` 开始把 adapter 前移到 `pre_decoder`，也就是先改 encoder bottleneck / transformer memory，再让原始 prompt decoder 去生成 `mask_embedding`。
- 到 `v5.2 / v5.3`，代码里同时保留了两种挂载方式：
  - `pre_decoder`：先改 `prompt_memory`
  - `post_decoder`：先出 `mask_embedding`，再改 query 对应表示

这一阶段的收益主要体现在召回提升，而不是误报被真正压住：

| 版本 | 主要挂载思路 | Mean Dice | Micro Dice | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | 无 adapter | 0.2139 | **0.4228** | 0.2493 | 0.3099 |
| v2 | `pre_decoder` 为主 | 0.2206 | 0.4099 | 0.2435 | 0.3461 |
| v4_catfix | feature-space + category/suppression | **0.2214** | 0.4054 | 0.2358 | **0.3524** |
| v5.1 | supervision 重构 | 0.2209 | 0.4080 | 0.2372 | 0.3497 |

从这些数能看出来，前期 adapter 更像“放大阳性响应”的工具：`mean_dice / recall` 上去了，但 `micro_dice / precision` 往往不占优。这说明它在 feature/query 空间里虽然能改行为，但副作用也会直接传进 decoder，容易把真正的 TP/FP 边界一起扰乱。

### 2. 中期版本：从 feature 调制转向 logit rectification

- `v6` 开始把“suppression”重解释成显式 `candidate / risk` 分工。
- `v6.1 / v6.2 / v6.3` 这几轮逐步从 feature-space 修补转向 output/logit-space 修正。
- 到 `v6.4` 之后，主思路已经稳定成：
  - 前面最多保留一个轻量 `pre_decoder` 入口
  - 真正有分量的 adapter 放到 decoder 输出之后，用 `rectify_logits(...)` 修正最终预测

这样做的原因很直接：如果目标是“压误报、修局部决策边界”，那直接改最终 logit 往往比提前改 memory/query 更稳，因为它不会把原始 decoder 的判别过程整体带偏。

### 3. v6.4 到 v6.6：保留轻前置入口，但主战场已经在输出端

`v6.4 / v6.5 / v6.6` 的 wrapper 代码都体现了同一个结构趋势：

- 前置入口仍然存在：兼容 `adapter_insertion_point == "pre_decoder"`
- 但不再像 `v5.2 / v5.3` 那样提供对等的 `post_decoder mask_embedding` 改写
- 真正的主逻辑统一放在 `decoded = self.base_model.decoder(...)` 之后，再走 `self.text_guided_adapter.rectify_logits(...)`

这一变化对应的指标风格也很明显：

| 版本 | 主体位置 | Mean Dice | Micro Dice | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| v6.4_stage3_fix | 轻 `pre_decoder` + logit rectification | 0.2141 | 0.4234 | **0.2586** | 0.3089 |
| v6.5 | 轻 `pre_decoder` + candidate/risk/refine logit path | 0.2133 | 0.4242 | 0.2509 | 0.3069 |
| v6.6 | 轻 `pre_decoder` + masked risk/refine logit path | 0.2126 | **0.4269** | 0.2548 | 0.3016 |

相对 baseline：

- `v6.4_stage3_fix`：`precision +0.0093`，`micro_dice +0.0006`
- `v6.5`：`micro_dice +0.0014`
- `v6.6`：`micro_dice +0.0041`，但 `recall -0.0083`

这组数据说明后期版本的目标已经不再是“尽量多激活阳性”，而是“尽量少伤主干，同时把错误阳性局部扣掉”。所以 adapter 看起来像是从前面挪到了后面，本质上是职责发生了变化：

- 早期：adapter 更像 feature/query 调制器
- 后期：adapter 更像 logit-level 决策边界修正器

### 当前结论

所以 `version` 不同、adapter 放置位置不同，不是代码风格不统一，而是实验路线真的换过一次挡：

- `v1 -> v5.3`：重点在“前面怎么改 feature / memory / query 才能把文本信号打进去”
- `v6.1 -> v6.6`：重点在“后面怎么直接修正 logit，才能更稳地抑制 FP”

现在保留 `pre_decoder` 参数，更多是为了兼容和做轻量前置条件化；真正决定 `v6.x` 行为风格的，已经是输出端的 `candidate / risk / refine / masked risk` 这一套 rectification 路径。

## 下一阶段主线

后续工作从“继续滚 adapter 小版本”转向四条更值得投入的主线：

1. 数据：覆盖、采样、hard negative、小病灶、稀有类别
2. 监督：FP-aware、boundary-aware、lesion-size-aware、category-aware supervision
3. 训练：stage 设计、冻结/解冻、negative sampling schedule、部分主干训练
4. 结构：每次只回答一个明确问题，避免继续多头叠加式试错

## 代码入口

### adapter 核心结构

- `VoxTell/voxtell/model/chest_candidate_risk_adapter.py`
- `VoxTell/voxtell/model/chest_candidate_risk_refine_adapter.py`
- `VoxTell/voxtell/model/chest_masked_risk_refine_adapter.py`
- `VoxTell/voxtell/model/chest_masked_risk_refine_adapter_v66b.py`
- `VoxTell/voxtell/model/chest_candidate_selective_risk_refine_adapter.py`

### wrapper

- `VoxTell/voxtell/model/v6_adapter_model.py`
- `VoxTell/voxtell/model/v65_adapter_model.py`
- `VoxTell/voxtell/model/v66_adapter_model.py`
- `VoxTell/voxtell/model/v66b_adapter_model.py`
- `VoxTell/voxtell/model/v67_adapter_model.py`

### 训练、验证、评估

- `scripts/training/train_voxtell_adapter.py`
- `scripts/experiments/run_voxtell_val_adapter_raw.py`
- `scripts/analysis/evaluate_voxtell_val_adapter_raw.py`

## 🚀 快速资源获取 (Assets)

本项目所需的所有数据子集、基础模型及已训练好的各版本 Adapter 权重均已统一托管至 ModelScope：

- **资源仓库**：[clover259/VoxTell-Complete-Assets](https://www.modelscope.cn/datasets/clover259/VoxTell-Complete-Assets)

### 环境复现步骤
为了确保代码能直接运行，请在下载资产仓库后，将对应的文件夹按以下结构摆放在本项目根目录下：
1. **数据层**：将 `datasets/` 文件夹整体放入根目录。
2. **基础模型**：将 `models/` 文件夹整体放入根目录。
3. **已训练权重**：将 `outputs/` 文件夹整体放入根目录。

放置后的目录结构应如下所示：
```text
题目2-demo/
├── datasets/
├── models/
├── outputs/
├── VoxTell/
└── scripts/
└── README.md (本文件)
```

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

## 文档索引

- adapter 线总结见 [实验记录/adapter线阶段总结.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/实验记录/adapter线阶段总结.md)
- 实验沉淀主文档见 [实验记录/README.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/实验记录/README.md)
