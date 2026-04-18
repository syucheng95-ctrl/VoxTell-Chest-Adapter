# Probe Case Comparison

## Aggregate

- `v3_catfix`: `mean_dice=0.2352`, `micro_dice=0.5454`, `mean_precision=0.2486`, `mean_recall=0.3384`, `count=16`
- `v4_catfix`: `mean_dice=0.2489`, `micro_dice=0.5348`, `mean_precision=0.2258`, `mean_recall=0.3615`, `count=16`
- `v4_probe_full`: `mean_dice=0.2489`, `micro_dice=0.5348`, `mean_precision=0.2259`, `mean_recall=0.3614`, `count=16`
- `v4_no_suppression`: `mean_dice=0.2469`, `micro_dice=0.5306`, `mean_precision=0.2214`, `mean_recall=0.3644`, `count=16`

## Case Level

- `train_13155_a_2.nii.gz`
  `v3_catfix`: `mean_dice=0.0016`, `micro_dice=0.0005`, `precision=0.1765`, `recall=0.0008`
  `v4_catfix`: `mean_dice=0.1028`, `micro_dice=0.0642`, `precision=0.0899`, `recall=0.1200`
  `v4_probe_full`: `mean_dice=0.1028`, `micro_dice=0.0642`, `precision=0.0900`, `recall=0.1199`
  `v4_no_suppression`: `mean_dice=0.0961`, `micro_dice=0.0660`, `precision=0.0775`, `recall=0.1266`
- `train_13292_a_2.nii.gz`
  `v3_catfix`: `mean_dice=0.2353`, `micro_dice=0.6804`, `precision=0.2366`, `recall=0.3080`
  `v4_catfix`: `mean_dice=0.2347`, `micro_dice=0.6778`, `precision=0.2287`, `recall=0.3106`
  `v4_probe_full`: `mean_dice=0.2347`, `micro_dice=0.6778`, `precision=0.2288`, `recall=0.3106`
  `v4_no_suppression`: `mean_dice=0.2349`, `micro_dice=0.6773`, `precision=0.2223`, `recall=0.3114`
- `train_13316_a_2.nii.gz`
  `v3_catfix`: `mean_dice=0.5443`, `micro_dice=0.6759`, `precision=0.4581`, `recall=0.6750`
  `v4_catfix`: `mean_dice=0.5339`, `micro_dice=0.6762`, `precision=0.4595`, `recall=0.6497`
  `v4_probe_full`: `mean_dice=0.5339`, `micro_dice=0.6762`, `precision=0.4595`, `recall=0.6497`
  `v4_no_suppression`: `mean_dice=0.5333`, `micro_dice=0.6755`, `precision=0.4586`, `recall=0.6499`
- `train_13417_d_1.nii.gz`
  `v3_catfix`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v4_catfix`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v4_probe_full`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v4_no_suppression`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
- `train_13430_a_2.nii.gz`
  `v3_catfix`: `mean_dice=0.3540`, `micro_dice=0.3540`, `precision=0.4211`, `recall=0.3053`
  `v4_catfix`: `mean_dice=0.3675`, `micro_dice=0.3675`, `precision=0.4175`, `recall=0.3282`
  `v4_probe_full`: `mean_dice=0.3675`, `micro_dice=0.3675`, `precision=0.4175`, `recall=0.3282`
  `v4_no_suppression`: `mean_dice=0.3675`, `micro_dice=0.3675`, `precision=0.4175`, `recall=0.3282`
- `train_13444_a_2.nii.gz`
  `v3_catfix`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v4_catfix`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v4_probe_full`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v4_no_suppression`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
- `train_13577_a_2.nii.gz`
  `v3_catfix`: `mean_dice=0.3780`, `micro_dice=0.3780`, `precision=0.2339`, `recall=0.9850`
  `v4_catfix`: `mean_dice=0.3154`, `micro_dice=0.3154`, `precision=0.1878`, `recall=0.9850`
  `v4_probe_full`: `mean_dice=0.3155`, `micro_dice=0.3155`, `precision=0.1878`, `recall=0.9850`
  `v4_no_suppression`: `mean_dice=0.3064`, `micro_dice=0.3064`, `precision=0.1814`, `recall=0.9850`
- `train_13583_d_2.nii.gz`
  `v3_catfix`: `mean_dice=0.4107`, `micro_dice=0.4462`, `precision=0.3893`, `recall=0.6157`
  `v4_catfix`: `mean_dice=0.4064`, `micro_dice=0.4335`, `precision=0.3777`, `recall=0.6262`
  `v4_probe_full`: `mean_dice=0.4064`, `micro_dice=0.4334`, `precision=0.3776`, `recall=0.6262`
  `v4_no_suppression`: `mean_dice=0.4058`, `micro_dice=0.4283`, `precision=0.3754`, `recall=0.6343`

## Largest Item Deltas

- `train_13577_a_2.nii.gz` idx=0 `ground_glass_opacity` `delta_dice=-0.0626` `prompt=Focal ground glass opacity in the laterobasal segment of the right lower lobe`
- `train_13583_d_2.nii.gz` idx=2 `consolidation` `delta_dice=-0.0214` `prompt=Consolidation areas showing more confluence`
- `train_13316_a_2.nii.gz` idx=0 `nodule` `delta_dice=-0.0181` `prompt=Multiple parenchymal nodules in both lungs`
- `train_13292_a_2.nii.gz` idx=0 `mass` `delta_dice=-0.0028` `prompt=Bilateral pleural effusion, more prominent on the left`
- `train_13316_a_2.nii.gz` idx=1 `nodule` `delta_dice=-0.0025` `prompt=Largest nodule in the anterobasal segment of the lower lobe of the left lung, measuring 18x15 mm`
- `train_13292_a_2.nii.gz` idx=2 `infiltration_or_interstitial_opacity` `delta_dice=0.0000` `prompt=Subcentimeter centriacinar nodules in the posterior segment of the right upper lobe`
- `train_13292_a_2.nii.gz` idx=1 `consolidation` `delta_dice=0.0010` `prompt=Atelectasis in both lungs adjacent to the pleural effusion`
- `train_13583_d_2.nii.gz` idx=0 `mass` `delta_dice=0.0107` `prompt=Progressive pleural effusion reaching a diameter of 3 cm between the leaves of the right pleura`
- `train_13430_a_2.nii.gz` idx=0 `nodule` `delta_dice=0.0135` `prompt=Pleural-based nodule measuring 5x3 mm in the superior segment of the lower lobe of the right lung`
- `train_13155_a_2.nii.gz` idx=0 `emphysema` `delta_dice=0.3037` `prompt=More prominent emphysematous changes in the upper lobes of both lungs`
