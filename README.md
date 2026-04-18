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

## 📥 资源下载

- 模型 `v1.1`：[Hugging Face](https://huggingface.co/mrokuss/VoxTell/tree/main/voxtell_v1.1)
- 数据 `Train50`：[ReXRank 官网](https://rexrank.ai/ReXGroundingCT/index.html)
- 元数据镜像：`datasets/ReXGroundingCT_mirror_meta/`

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
