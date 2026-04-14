# 实验计划

本目录用于沉淀当前阶段之后的实验主线、优先级和执行顺序。  
和 `实验记录` 不同，这里不回顾历史细节，重点只放“接下来做什么、为什么这么做、先后顺序是什么”。

## 当前主结论

- `LoRA` 不再作为主线。
- `ChestTextGuidedAdapter v1` 已验证失败。
- `ChestTextGuidedAdapter v2` 已经把 `mean_dice` 提升到高于 `raw baseline`。
- 当前真正剩下的问题是：`micro_dice` 还略低，说明总体体素级误报仍偏多。

因此，下一阶段的主目标不是“继续放大召回”，而是：

- 保住 `v2` 已经得到的 `mean_dice` 优势
- 继续压低假阳性
- 把 `micro_dice` 也推过 baseline

## 当前推荐主线

当前最值得继续的方向是：

- `ChestTextGuidedAdapter v3`

推荐定义为：

- 保留 `pre-decoder` 接线
- 保留保守 gate + 小残差
- 引入更强的“背景抑制 / suppression”能力
- 逐步加入类别感知和区域约束

## 文件导航

- [下一阶段结构改进计划.md](/C:/Users/Sunyucheng/Desktop/作业(大学相关）/生医工大赛/题目2-demo/实验计划/下一阶段结构改进计划.md)

## 一句话决策

后面不再回到 `LoRA`，而是在 `adapter v2` 的基础上继续做更强约束的结构改进，目标是把 `micro_dice` 也推过 baseline。
