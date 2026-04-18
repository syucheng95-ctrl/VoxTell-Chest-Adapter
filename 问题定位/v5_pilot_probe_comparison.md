# Probe Case Comparison

## Aggregate

- `v4_current`: `mean_dice=0.2486`, `micro_dice=0.5507`, `mean_precision=0.2354`, `mean_recall=0.3468`, `count=16`
- `v5_soft_target`: `mean_dice=0.2487`, `micro_dice=0.5507`, `mean_precision=0.2355`, `mean_recall=0.3468`, `count=16`
- `v5_soft_target_fp`: `mean_dice=0.2487`, `micro_dice=0.5507`, `mean_precision=0.2355`, `mean_recall=0.3468`, `count=16`

## Case Level

- `train_13155_a_2.nii.gz`
  `v4_current`: `mean_dice=0.0920`, `micro_dice=0.0416`, `precision=0.1260`, `recall=0.0724`
  `v5_soft_target`: `mean_dice=0.0920`, `micro_dice=0.0416`, `precision=0.1259`, `recall=0.0724`
  `v5_soft_target_fp`: `mean_dice=0.0918`, `micro_dice=0.0415`, `precision=0.1259`, `recall=0.0722`
- `train_13292_a_2.nii.gz`
  `v4_current`: `mean_dice=0.2344`, `micro_dice=0.6797`, `precision=0.2262`, `recall=0.3059`
  `v5_soft_target`: `mean_dice=0.2344`, `micro_dice=0.6797`, `precision=0.2268`, `recall=0.3059`
  `v5_soft_target_fp`: `mean_dice=0.2344`, `micro_dice=0.6797`, `precision=0.2264`, `recall=0.3059`
- `train_13316_a_2.nii.gz`
  `v4_current`: `mean_dice=0.5549`, `micro_dice=0.6800`, `precision=0.4699`, `recall=0.6841`
  `v5_soft_target`: `mean_dice=0.5552`, `micro_dice=0.6800`, `precision=0.4700`, `recall=0.6846`
  `v5_soft_target_fp`: `mean_dice=0.5552`, `micro_dice=0.6801`, `precision=0.4700`, `recall=0.6846`
- `train_13417_d_1.nii.gz`
  `v4_current`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v5_soft_target`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v5_soft_target_fp`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
- `train_13430_a_2.nii.gz`
  `v4_current`: `mean_dice=0.3182`, `micro_dice=0.3182`, `precision=0.3933`, `recall=0.2672`
  `v5_soft_target`: `mean_dice=0.3182`, `micro_dice=0.3182`, `precision=0.3933`, `recall=0.2672`
  `v5_soft_target_fp`: `mean_dice=0.3182`, `micro_dice=0.3182`, `precision=0.3933`, `recall=0.2672`
- `train_13444_a_2.nii.gz`
  `v4_current`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v5_soft_target`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
  `v5_soft_target_fp`: `mean_dice=0.0000`, `micro_dice=0.0000`, `precision=0.0000`, `recall=0.0000`
- `train_13577_a_2.nii.gz`
  `v4_current`: `mean_dice=0.3461`, `micro_dice=0.3461`, `precision=0.2099`, `recall=0.9850`
  `v5_soft_target`: `mean_dice=0.3463`, `micro_dice=0.3463`, `precision=0.2101`, `recall=0.9850`
  `v5_soft_target_fp`: `mean_dice=0.3463`, `micro_dice=0.3463`, `precision=0.2101`, `recall=0.9850`
- `train_13583_d_2.nii.gz`
  `v4_current`: `mean_dice=0.4084`, `micro_dice=0.4538`, `precision=0.3891`, `recall=0.5977`
  `v5_soft_target`: `mean_dice=0.4084`, `micro_dice=0.4538`, `precision=0.3891`, `recall=0.5977`
  `v5_soft_target_fp`: `mean_dice=0.4084`, `micro_dice=0.4538`, `precision=0.3892`, `recall=0.5977`

## Largest Item Deltas

- `train_13155_a_2.nii.gz` idx=0 `emphysema` `delta_dice=-0.0001` `prompt=More prominent emphysematous changes in the upper lobes of both lungs`
- `train_13316_a_2.nii.gz` idx=1 `nodule` `delta_dice=-0.0001` `prompt=Largest nodule in the anterobasal segment of the lower lobe of the left lung, measuring 18x15 mm`
- `train_13292_a_2.nii.gz` idx=0 `mass` `delta_dice=-0.0000` `prompt=Bilateral pleural effusion, more prominent on the left`
- `train_13583_d_2.nii.gz` idx=0 `mass` `delta_dice=-0.0000` `prompt=Progressive pleural effusion reaching a diameter of 3 cm between the leaves of the right pleura`
- `train_13155_a_2.nii.gz` idx=1 `nodule` `delta_dice=0.0000` `prompt=Several subcentimeter nodules in both lungs, measuring less than 3 mm in short diameter`
- `train_13583_d_2.nii.gz` idx=1 `consolidation` `delta_dice=0.0000` `prompt=Extensive consolidation areas in all segments of both lungs`
- `train_13583_d_2.nii.gz` idx=2 `consolidation` `delta_dice=0.0000` `prompt=Consolidation areas showing more confluence`
- `train_13292_a_2.nii.gz` idx=1 `consolidation` `delta_dice=0.0001` `prompt=Atelectasis in both lungs adjacent to the pleural effusion`
- `train_13577_a_2.nii.gz` idx=0 `ground_glass_opacity` `delta_dice=0.0002` `prompt=Focal ground glass opacity in the laterobasal segment of the right lower lobe`
- `train_13316_a_2.nii.gz` idx=0 `nodule` `delta_dice=0.0006` `prompt=Multiple parenchymal nodules in both lungs`
