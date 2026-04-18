# scripts

本目录用于存放“问题定位实验”对应的辅助脚本。

建议后续脚本按用途命名，例如：

- `ablate_v4_suppression.py`
- `compare_v2_v3_v4_by_category.py`
- `analyze_v4_suppression_correlation.py`
- `collect_v4_case_review_list.py`

要求：

- 脚本优先服务定位问题，不要和正式训练/正式评估脚本混在一起
- 输出文件路径尽量写到 `outputs/` 或本目录旁边的分析结果位置
- 每个脚本最好都能对应 `v4_问题定位实验清单.md` 里的一个实验项

