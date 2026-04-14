# Train Subset Selection Notes

Date: 2026-04-13

## Purpose

Select a small but category-diverse `train` subset for LoRA fine-tuning without downloading the full training set.

## Source

- Metadata file: `datasets/ReXGroundingCT_mirror_meta/dataset.json`
- Full train split size: `2992` cases

## Selection Principle

- Prefer broad category coverage over prevalence matching
- Prefer multi-finding cases because they give more supervision per downloaded CT
- Control total data size to keep LoRA iteration cheap

## Recommended Options

### Option A: 30-case smoke-test subset

- File: `docs/plans/train_subset_30_cases.txt`
- Estimated image size: about `2.2 GB`
- Estimated segmentation size: about `0.03-0.06 GB`
- Recommended use: first LoRA smoke test

### Option B: 50-case stronger subset

- File: `docs/plans/train_subset_50_cases.txt`
- Estimated image size: about `3.6 GB`
- Estimated segmentation size: about `0.05-0.10 GB`
- Recommended use: first meaningful LoRA experiment after smoke test

## Notes

- The split metadata includes some filenames starting with `valid_`; they still appear inside the metadata train split and were therefore not excluded automatically.
- Category codes are heterogeneous. Some codes map cleanly to one lesion type, while some are mixed.
- Current recommendation is to start from the 30-case subset, validate that the training loop is stable, then expand to the 50-case subset.
