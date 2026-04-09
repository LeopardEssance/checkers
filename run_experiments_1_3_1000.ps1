# Runs Experiments 1-3 with 1000 games per side for each experiment.
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\run_experiments_1_3_1000.ps1

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

Write-Host "============================================================"
Write-Host "Running Experiment 1 (Head-to-Head) with 1000 games..."
Write-Host "============================================================"
python -m experiments.experiment1_head_to_head --games 500 --seed 123 --stochastic-tiebreak --opening-random-plies 2

Write-Host ""
Write-Host "============================================================"
Write-Host "Running Experiment 2 (Ablation) with 1000 games..."
Write-Host "============================================================"
python -m experiments.experiment2_ablation --games 500 --seed 123 --stochastic-tiebreak --opening-random-plies 2

Write-Host ""
Write-Host "============================================================"
Write-Host "Running Experiment 3 (Scalability) with 1000 games..."
Write-Host "============================================================"
python -m experiments.experiment3_scalability --games 500 --seed 123 --stochastic-tiebreak --opening-random-plies 2

Write-Host ""
Write-Host "All experiments finished."
