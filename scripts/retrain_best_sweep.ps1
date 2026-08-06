param(
    [string]$PythonExe = "",
    [string]$DataPath = "tests_thermal/data/test_data.csv",
    [string]$ReportRoot = "tests_thermal/reports",
    [string]$SweepDir = "",
    [double]$PhysicsLossWeight = 0.10,
    [int]$Epochs = 40,
    [int]$Patience = 8,
    [int]$BatchSize = 16,
    [int]$Lookahead = 144,
    [string]$FeatureLevel = "standard",
    [string]$TargetCols = "room_temp,electric_power,gas_consumption",
    [int]$MaxRuns = 0,
    [switch]$RunAllPhases
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $venvPython = Join-Path $repoRoot ".venv311/Scripts/python.exe"
    if (Test-Path $venvPython) {
        $PythonExe = $venvPython
    }
    else {
        $PythonExe = "python"
    }
}

if ([string]::IsNullOrWhiteSpace($SweepDir)) {
    $reportsAbs = Join-Path $repoRoot $ReportRoot
    if (-not (Test-Path $reportsAbs)) {
        throw "ReportRoot not found: $reportsAbs"
    }

    $latest = Get-ChildItem -Path $reportsAbs -Directory |
        Where-Object { $_.Name -like "physics_balance_sweep_*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $latest) {
        throw "No sweep folders found under $reportsAbs"
    }

    $SweepDir = $latest.FullName
}

$summaryPath = Join-Path $SweepDir "sweep_summary.csv"
if (-not (Test-Path $summaryPath)) {
    throw "sweep_summary.csv not found: $summaryPath"
}

function Convert-ToDoubleInvariant {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    $normalized = $Value.Trim().Replace(",", ".")
    return [double]::Parse($normalized, [System.Globalization.CultureInfo]::InvariantCulture)
}

$summaryJsonPath = Join-Path $SweepDir "sweep_summary.json"
if (Test-Path $summaryJsonPath) {
    $summary = Get-Content $summaryJsonPath -Raw | ConvertFrom-Json
}
else {
    $summary = Import-Csv -Path $summaryPath
}

$best = $summary |
    Where-Object {
        $_.status -eq "ok" -and
        -not [string]::IsNullOrWhiteSpace([string]$_.rmse_c)
    } |
    Sort-Object { Convert-ToDoubleInvariant ([string]$_.rmse_c) } |
    Select-Object -First 1

if ($null -eq $best) {
    throw "No successful sweep run found in $summaryPath"
}

$bestWeight = Convert-ToDoubleInvariant ([string]$best.physics_balance_weight)
$sourceRunDir = $best.run_dir
if ([string]::IsNullOrWhiteSpace($sourceRunDir) -or -not (Test-Path $sourceRunDir)) {
    throw "Best run directory not found: $sourceRunDir"
}

$phase1BestSource = Join-Path $sourceRunDir "forecast_gridsearch_phase1_best.json"
if (-not (Test-Path $phase1BestSource)) {
    throw "Missing phase1 best config in best run: $phase1BestSource"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$retrainDir = Join-Path $SweepDir ("retrain_from_" + $best.run_name + "_" + $timestamp)
New-Item -ItemType Directory -Path $retrainDir -Force | Out-Null

# Seed phase-2 center with the best run from the sweep winner.
Copy-Item -Path $phase1BestSource -Destination (Join-Path $retrainDir "forecast_gridsearch_phase1_best.json") -Force

$meta = [pscustomobject]@{
    selected_run_name = $best.run_name
    selected_run_dir = $sourceRunDir
    selected_physics_balance_weight = $bestWeight
    selected_rmse_c = Convert-ToDoubleInvariant ([string]$best.rmse_c)
    selected_mae_c = Convert-ToDoubleInvariant ([string]$best.mae_c)
    retrain_dir = $retrainDir
    created_at = (Get-Date).ToString("o")
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $retrainDir "retrain_selection.json") -Encoding UTF8

$args = @(
    "-m", "emhass.thermal.forecast_gridsearch",
    "--data-path", $DataPath,
    "--report-dir", $retrainDir,
    "--epochs", $Epochs,
    "--patience", $Patience,
    "--batch-size", $BatchSize,
    "--lookahead", $Lookahead,
    "--feature-level", $FeatureLevel,
    "--target-cols", $TargetCols,
    "--physics-loss-weight", $PhysicsLossWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--physics-balance-weight", $bestWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)

if ($MaxRuns -gt 0) {
    $args += @("--max-runs", $MaxRuns)
}

if ($RunAllPhases) {
    $args += "--run-all-phases"
}
else {
    # Default behavior: refine around best config from sweep winner.
    $args += "--phase2"
}

Push-Location $repoRoot
try {
    Write-Host "Selected best sweep run: $($best.run_name) (pbw=$bestWeight, rmse=$($best.rmse_c))"
    Write-Host "Retrain output dir: $retrainDir"
    & $PythonExe @args
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    throw "Retrain command failed with exit code $exitCode"
}

$retrainBestPath = Join-Path $retrainDir "forecast_gridsearch_best.json"
if (Test-Path $retrainBestPath) {
    $retrainBest = Get-Content $retrainBestPath -Raw | ConvertFrom-Json

    $baseRmse = Convert-ToDoubleInvariant ([string]$best.rmse_c)
    $baseMae = Convert-ToDoubleInvariant ([string]$best.mae_c)
    $newRmse = Convert-ToDoubleInvariant ([string]$retrainBest.rmse_c)
    $newMae = Convert-ToDoubleInvariant ([string]$retrainBest.mae_c)

    $deltaRmse = $newRmse - $baseRmse
    $deltaMae = $newMae - $baseMae
    $pctRmse = if ($baseRmse -ne 0) { 100.0 * $deltaRmse / $baseRmse } else { 0.0 }
    $pctMae = if ($baseMae -ne 0) { 100.0 * $deltaMae / $baseMae } else { 0.0 }

    $reportPath = Join-Path $retrainDir "retrain_improvement_report.md"
    $report = @(
        "# Retrain Improvement Report",
        "",
        "## Baseline (best sweep run)",
        "- run_name: $($best.run_name)",
        "- physics_balance_weight: $bestWeight",
        "- rmse_c: $baseRmse",
        "- mae_c: $baseMae",
        "",
        "## Retrain result",
        "- rmse_c: $newRmse",
        "- mae_c: $newMae",
        "",
        "## Delta (retrain - baseline)",
        "- rmse_c_delta: $deltaRmse ($pctRmse%)",
        "- mae_c_delta: $deltaMae ($pctMae%)",
        "",
        "Lower is better for RMSE/MAE. Negative delta means improvement."
    ) -join [Environment]::NewLine

    Set-Content -Path $reportPath -Value $report -Encoding UTF8
}

Write-Host "Retrain completed successfully."
Write-Host "Selection metadata: $(Join-Path $retrainDir 'retrain_selection.json')"
Write-Host "Best output: $(Join-Path $retrainDir 'forecast_gridsearch_best.json')"
if (Test-Path (Join-Path $retrainDir 'retrain_improvement_report.md')) {
    Write-Host "Improvement report: $(Join-Path $retrainDir 'retrain_improvement_report.md')"
}
