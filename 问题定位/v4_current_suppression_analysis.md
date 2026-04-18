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
- `avg_suppression_loss_negative`: `4.52578125`

## By Sample Mode

- `hard_negative`
  `count`: `1`
  `mean_loss`: `3.992919683456421`
  `mean_seg_loss`: `3.569091558456421`
  `mean_fp_penalty`: `0.07800612598657608`
  `mean_patch_fg_fraction`: `0.00047782615379050923`
  `mean_suppression_mean`: `0.9853515625`
  `mean_suppression_loss`: `4.23828125`
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
  `mean_loss`: `1.768269419670105`
  `mean_seg_loss`: `1.308552622795105`
  `mean_fp_penalty`: `0.0`
  `mean_patch_fg_fraction`: `0.0005477975915979456`
  `mean_suppression_mean`: `0.988525390625`
  `mean_suppression_loss`: `4.59765625`
  `mean_bce`: `0.023421063320711255`
  `mean_dice_loss`: `0.5`

## Probe Summary

- `label`: `v4_current`
- `aggregate.count`: `16`
- `aggregate.mean_dice`: `0.24864527325611202`
- `aggregate.mean_iou`: `0.17454528050800253`
- `aggregate.mean_precision`: `0.23542398861163522`
- `aggregate.mean_recall`: `0.34677029129267267`
- `aggregate.sum_tp`: `2283800`
- `aggregate.sum_fp`: `2939394`
- `aggregate.sum_fn`: `786464`
- `aggregate.micro_dice`: `0.550747348090507`
- `aggregate.micro_iou`: `0.38002162519065147`
- `suppression_mean_over_items`: `0.9682121276855469`
- `suppression_mean_when_fp_gt_0`: `0.9675038655598959`
- `suppression_mean_when_dice_gt_0`: `0.96590576171875`
- `mean_fp_over_items`: `183712.125`
- `mean_pred_voxels_over_items`: `326449.625`
