$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $root

$python = "C:\Users\Sunyucheng\anaconda3\envs\pytorch\python.exe"
$runId = "voxtell_adapter_v2_fullrun"
$runDir = Join-Path $root ("outputs\" + $runId)
$valDir = Join-Path $root ("outputs\voxtell_val_adapter_raw_v2")
$metricsPath = Join-Path $root "outputs\voxtell_val_adapter_raw_v2_metrics.json"

$env:HF_HOME = "$root\hf_cache"
$env:HF_HUB_CACHE = "$root\hf_cache\hub"
$env:TRANSFORMERS_NO_TF = "1"

Write-Host "== Train adapter =="
& $python scripts\training\train_voxtell_adapter.py `
  --output-dir $runDir `
  --hf-home "$root\hf_cache" `
  --epochs 5 `
  --max-cases 50 `
  --max-findings-per-case 2 `
  --save-every-steps 20 `
  --lr 1e-4 `
  --weight-decay 1e-4 `
  --fp-weight 0.05 `
  --negative-fp-weight 0.25 `
  --negative-bce-weight 4.0 `
  --negative-loss-scale 2.0 `
  --adapter-hidden-dim 1024 `
  --adapter-insertion-point pre_decoder `
  --adapter-num-groups 8 `
  --adapter-residual-scale 0.1 `
  --adapter-gate-cap 0.25 `
  --positive-patch-prob 0.35 `
  --hard-negative-patch-prob 0.35 `
  --negative-max-fg-fraction 0.001 `
  --negative-sampling-after-steps 30 `
  --seed 42

Write-Host "== Run val inference =="
& $python scripts\experiments\run_voxtell_val_adapter_raw.py `
  --adapter-run-dir $runDir `
  --output-dir $valDir `
  --adapter-insertion-point pre_decoder `
  --adapter-num-groups 8 `
  --adapter-residual-scale 0.1 `
  --adapter-gate-cap 0.25 `
  --force

Write-Host "== Evaluate against baseline =="
& $python scripts\analysis\evaluate_voxtell_val_adapter_raw.py `
  --adapter-dir $valDir `
  --output $metricsPath

Write-Host "Pipeline completed."
