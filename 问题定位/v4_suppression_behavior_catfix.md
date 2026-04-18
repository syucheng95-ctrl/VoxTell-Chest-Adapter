# v4 Suppression Behavior Analysis

## Training Diagnostics

- `zero_loss_count`: `0`
- `zero_loss_ratio`: `0.0`
- `zero_loss_neg_count`: `0`
- `hard_negative_count`: `150`
- `random_negative_count`: `122`
- `positive_count`: `228`
- `suppression_mean_pos_avg`: `0.5303847998903509`
- `suppression_mean_neg_avg`: `0.4779227761661305`
- `avg_patch_fg_positive`: `0.020655159209749`
- `avg_patch_fg_negative`: `0.0018707100861991932`
- `avg_fp_penalty_positive`: `0.00383670226063454`
- `avg_fp_penalty_negative`: `0.0010161161227783316`
- `avg_suppression_loss_positive`: `0.6928296674761856`
- `avg_suppression_loss_negative`: `0.7371799244600183`

## By Sample Mode

- `hard_negative`
  `count`: `150`
  `mean_loss`: `2.6322984663645426`
  `mean_seg_loss`: `2.557820113499959`
  `mean_fp_penalty`: `0.0008674436837524506`
  `mean_patch_fg_fraction`: `0.000489551873854649`
  `mean_suppression_mean`: `0.48083251953125`
  `mean_suppression_loss`: `0.7447916666666666`
  `mean_bce`: `0.023098437631851994`
  `mean_dice_loss`: `0.9999081635475159`
- `positive`
  `count`: `198`
  `mean_loss`: `1.5476591337208796`
  `mean_seg_loss`: `1.4717476224959498`
  `mean_fp_penalty`: `0.0039041146770436427`
  `mean_patch_fg_fraction`: `0.02074927798469906`
  `mean_suppression_mean`: `0.4902084812973485`
  `mean_suppression_loss`: `0.759127838443024`
  `mean_bce`: `0.5270106791132929`
  `mean_dice_loss`: `0.9445417408991341`
- `positive_warmup`
  `count`: `30`
  `mean_loss`: `1.5575945854187012`
  `mean_seg_loss`: `1.5320683499177297`
  `mean_fp_penalty`: `0.0033917803123344648`
  `mean_patch_fg_fraction`: `0.020033975295078606`
  `mean_suppression_mean`: `0.7955485026041667`
  `mean_suppression_loss`: `0.25526173909505206`
  `mean_bce`: `0.5587844589725137`
  `mean_dice_loss`: `0.9731143097082774`
- `random_negative`
  `count`: `122`
  `mean_loss`: `1.3087896519019955`
  `mean_seg_loss`: `1.2360069790824515`
  `mean_fp_penalty`: `0.0011989101051872017`
  `mean_patch_fg_fraction`: `0.003568855429245764`
  `mean_suppression_mean`: `0.47434522284836067`
  `mean_suppression_loss`: `0.7278212250256147`
  `mean_bce`: `0.12865867683810436`
  `mean_dice_loss`: `0.36557626724243164`

## Probe Summary

- `label`: `v4_probe_full`
- `aggregate.count`: `16`
- `aggregate.mean_dice`: `0.24891230185267124`
- `aggregate.mean_iou`: `0.17470250192178416`
- `aggregate.mean_precision`: `0.2258532520535596`
- `aggregate.mean_recall`: `0.36143124127776294`
- `aggregate.sum_tp`: `2338256`
- `aggregate.sum_fp`: `3336115`
- `aggregate.sum_fn`: `732008`
- `aggregate.micro_dice`: `0.534786414756019`
- `aggregate.micro_iou`: `0.36498870891029084`
- `suppression_mean_over_items`: `0.4628715515136719`
- `suppression_mean_when_fp_gt_0`: `0.473114013671875`
- `suppression_mean_when_dice_gt_0`: `0.47423095703125`
- `mean_fp_over_items`: `208507.1875`
- `mean_pred_voxels_over_items`: `354648.1875`
