# v4 Suppression Behavior Analysis

## Training Diagnostics

- `zero_loss_count`: `0`
- `zero_loss_ratio`: `0.0`
- `zero_loss_neg_count`: `0`
- `hard_negative_count`: `1`
- `random_negative_count`: `4`
- `positive_count`: `35`
- `suppression_mean_pos_avg`: `0.8470912388392857`
- `suppression_mean_neg_avg`: `0.987890625`
- `avg_patch_fg_positive`: `0.017009032718718998`
- `avg_patch_fg_negative`: `0.0005338033040364583`
- `avg_fp_penalty_positive`: `0.0003730888162473483`
- `avg_fp_penalty_negative`: `0.015601225197315216`
- `avg_suppression_loss_positive`: `0.1900414603097098`
- `avg_suppression_loss_negative`: `4.187890625`

## By Sample Mode

- `hard_negative`
  `count`: `1`
  `mean_loss`: `4.182054042816162`
  `mean_seg_loss`: `3.927171230316162`
  `mean_fp_penalty`: `0.07800612598657608`
  `mean_patch_fg_fraction`: `0.00047782615379050923`
  `mean_suppression_mean`: `0.9853515625`
  `mean_suppression_loss`: `2.548828125`
  `mean_bce`: `0.417061984539032`
  `mean_dice_loss`: `0.9996538758277893`
- `positive`
  `count`: `5`
  `mean_loss`: `1.319238829612732`
  `mean_seg_loss`: `1.3180987596511842`
  `mean_fp_penalty`: `0.0001573042245581746`
  `mean_patch_fg_fraction`: `0.009618603741681135`
  `mean_suppression_mean`: `0.988671875`
  `mean_suppression_loss`: `0.01139984130859375`
  `mean_bce`: `0.31818555258214476`
  `mean_dice_loss`: `0.9999053359031678`
- `positive_warmup`
  `count`: `30`
  `mean_loss`: `1.4647421419620514`
  `mean_seg_loss`: `1.4427624603112539`
  `mean_fp_penalty`: `0.00040905291486221054`
  `mean_patch_fg_fraction`: `0.018240770881558642`
  `mean_suppression_mean`: `0.8234944661458333`
  `mean_suppression_loss`: `0.2198150634765625`
  `mean_bce`: `0.47846703415270897`
  `mean_dice_loss`: `0.9642749706904093`
- `random_negative`
  `count`: `4`
  `mean_loss`: `1.8991246819496155`
  `mean_seg_loss`: `1.4394078850746155`
  `mean_fp_penalty`: `0.0`
  `mean_patch_fg_fraction`: `0.0005477975915979456`
  `mean_suppression_mean`: `0.988525390625`
  `mean_suppression_loss`: `4.59765625`
  `mean_bce`: `0.023421063320711255`
  `mean_dice_loss`: `0.5`

## Probe Summary

- `label`: `v5_soft_target_fp`
- `aggregate.count`: `16`
- `aggregate.mean_dice`: `0.24865733568808995`
- `aggregate.mean_iou`: `0.17455627874651033`
- `aggregate.mean_precision`: `0.23549746419754425`
- `aggregate.mean_recall`: `0.3467956262176651`
- `aggregate.sum_tp`: `2283786`
- `aggregate.sum_fp`: `2939401`
- `aggregate.sum_fn`: `786478`
- `aggregate.micro_dice`: `0.5507444367851212`
- `aggregate.micro_iou`: `0.38001885296434995`
- `suppression_mean_over_items`: `0.9682121276855469`
- `suppression_mean_when_fp_gt_0`: `0.9675038655598959`
- `suppression_mean_when_dice_gt_0`: `0.96590576171875`
- `mean_fp_over_items`: `183712.5625`
- `mean_pred_voxels_over_items`: `326449.1875`
