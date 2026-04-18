# v4 问题定位集候选

以下 8 个 case 用于后续 suppression / category-aware 的问题定位。

## v4_worse_than_v3

- `train_13577_a_2.nii.gz`
  `delta_mean_dice = -0.0418`, `v3 = 0.3790`, `v4 = 0.3372`
  `gt_voxels = 533`, `pred_voxels_v4 = 2581`, `item_count = 1`
  `categories = ground_glass_opacity`
  `lesions = ground-glass opacity`
- `train_13316_a_2.nii.gz`
  `delta_mean_dice = -0.0145`, `v3 = 0.5476`, `v4 = 0.5332`
  `gt_voxels = 4634`, `pred_voxels_v4 = 6880`, `item_count = 2`
  `categories = nodule`
  `lesions = nodule`

## v4_better_than_v3

- `train_13155_a_2.nii.gz`
  `delta_mean_dice = 0.0963`, `v3 = 0.0050`, `v4 = 0.1013`
  `gt_voxels = 37930`, `pred_voxels_v4 = 5357`, `item_count = 3`
  `categories = atelectasis, emphysema, nodule`
  `lesions = atelectasis, emphysema, nodule`
- `train_13430_a_2.nii.gz`
  `delta_mean_dice = 0.0209`, `v3 = 0.3467`, `v4 = 0.3675`
  `gt_voxels = 131`, `pred_voxels_v4 = 103`, `item_count = 1`
  `categories = nodule`
  `lesions = nodule`

## small_target_cases

- `train_13417_d_1.nii.gz`
  `delta_mean_dice = 0.0000`, `v3 = 0.0000`, `v4 = 0.0000`
  `gt_voxels = 160`, `pred_voxels_v4 = 2403`, `item_count = 2`
  `categories = atelectasis, nodule`
  `lesions = fibrosis, nodule`
- `train_13444_a_2.nii.gz`
  `delta_mean_dice = 0.0000`, `v3 = 0.0000`, `v4 = 0.0000`
  `gt_voxels = 196`, `pred_voxels_v4 = 1`, `item_count = 1`
  `categories = nodule`
  `lesions = nodule`

## diffuse_cases

- `train_13583_d_2.nii.gz`
  `delta_mean_dice = -0.0046`, `v3 = 0.4119`, `v4 = 0.4073`
  `gt_voxels = 1576032`, `pred_voxels_v4 = 3358107`, `item_count = 3`
  `categories = consolidation, mass`
  `lesions = consolidation, pleural effusion`
- `train_13292_a_2.nii.gz`
  `delta_mean_dice = -0.0001`, `v3 = 0.2347`, `v4 = 0.2346`
  `gt_voxels = 1450648`, `pred_voxels_v4 = 2234131`, `item_count = 3`
  `categories = consolidation, infiltration_or_interstitial_opacity, mass`
  `lesions = atelectasis, nodule, pleural effusion`
