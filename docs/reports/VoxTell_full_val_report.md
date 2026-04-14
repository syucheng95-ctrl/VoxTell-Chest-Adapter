# VoxTell on ReXGroundingCT Public Val

本报告基于 ReXGroundingCT public val 全部 50 个 case、共 115 个 finding，对 VoxTell 在 raw prompt 和 structured prompt 两种输入方式下的表现进行了对比。

总体结果显示：raw 的 mean Dice 为 0.2139，structured 为 0.2106；raw 的 micro Dice 为 0.4228，structured 为 0.2234。structured 在 full-val 整体上没有超过 raw，当前版本不适合作为默认替代输入。

## Overall

- Cases: 50
- Findings: 115
- Raw mean Dice: 0.2139
- Structured mean Dice: 0.2106
- Raw micro Dice: 0.4228
- Structured micro Dice: 0.2234

## Conclusion

- Structured prompt is not a stable global replacement for raw prompt.
- It helps some focal and location-constrained findings, but hurts a substantial share of findings, especially those requiring broader contextual wording.

## Reasons

- structured prompt 把自然语言压缩成固定字段后，确实提升了语义约束，但也丢失了原句里的上下文细节和描述强度。
- 对局灶、位置明确的 finding，结构化字段通常更有利，因为 lesion type、laterality、anatomy、location 能直接提供空间先验。
- 对范围大、边界弱、语义带模糊程度或病程变化的 finding，模板化表达容易把有效语义过度压缩，导致召回下降。
- 当前模板里的字段覆盖仍然不完整，尤其是 extent、distribution、severity、comparison、texture pattern 这类信息没有被稳定保留下来。
- structured 目前的收益更像是『按类型选择性启用』，而不是『对所有 prompt 一刀切替换』。
