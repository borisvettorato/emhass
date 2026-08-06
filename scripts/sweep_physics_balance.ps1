param(
    [string]$PythonExe = "",
    [string]$DataPath = "tests_thermal/data/test_data.csv",
    [string]$ReportRoot = "tests_thermal/reports",
    [double[]]$PhysicsBalanceWeights = @(0.02, 0.05, 0.10),
    [double]$PhysicsLossWeight = 0.10,
    [int]$Epochs = 20,
    [int]$Patience = 5,
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

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sweepDir = Join-Path $repoRoot (Join-Path $ReportRoot ("physics_balance_sweep_" + $timestamp))
New-Item -ItemType Directory -Path $sweepDir -Force | Out-Null

$results = New-Object System.Collections.Generic.List[object]

Push-Location $repoRoot
try {
    foreach ($w in $PhysicsBalanceWeights) {
        $wLabel = $w.ToString("0.###", [System.Globalization.CultureInfo]::InvariantCulture)
        $runName = "pbw_" + $wLabel.Replace(".", "p")
        $runDir = Join-Path $sweepDir $runName
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null

        $args = @(
            "-m", "emhass.thermal.forecast_gridsearch",
            "--data-path", $DataPath,
            "--report-dir", $runDir,
            "--epochs", $Epochs,
            "--patience", $Patience,
            "--batch-size", $BatchSize,
            "--lookahead", $Lookahead,
            "--feature-level", $FeatureLevel,
            "--target-cols", $TargetCols,
            "--physics-loss-weight", $PhysicsLossWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--physics-balance-weight", $wLabel
        )

        if ($MaxRuns -gt 0) {
            $args += @("--max-runs", $MaxRuns)
        }
        if ($RunAllPhases) {
            $args += "--run-all-phases"
        }

        Write-Host "Running sweep item: physics-balance-weight=$wLabel"
        & $PythonExe @args
        $exitCode = $LASTEXITCODE

        if ($exitCode -ne 0) {
            $results.Add([pscustomobject]@{
                run_name = $runName
                physics_balance_weight = [double]$w
                status = "failed"
                exit_code = $exitCode
                rmse_c = $null
                mae_c = $null
                best_epoch = $null
                input_window = $null
                hidden_size = $null
                num_layers = $null
                run_dir = $runDir
            })
            continue
        }

        $bestPath = Join-Path $runDir "forecast_gridsearch_best.json"
        if (-not (Test-Path $bestPath)) {
            $results.Add([pscustomobject]@{
                run_name = $runName
                physics_balance_weight = [double]$w
                status = "missing_best_json"
                exit_code = 0
                rmse_c = $null
                mae_c = $null
                best_epoch = $null
                input_window = $null
                hidden_size = $null
                num_layers = $null
                run_dir = $runDir
            })
            continue
        }

        $best = Get-Content $bestPath -Raw | ConvertFrom-Json

        $results.Add([pscustomobject]@{
            run_name = $runName
            physics_balance_weight = [double]$w
            status = "ok"
            exit_code = 0
            rmse_c = [double]$best.rmse_c
            mae_c = [double]$best.mae_c
            best_epoch = [int]$best.best_epoch
            input_window = [int]$best.input_window
            hidden_size = [int]$best.hidden_size
            num_layers = [int]$best.num_layers
            run_dir = $runDir
        })
    }
}
finally {
    Pop-Location
}

$summaryCsv = Join-Path $sweepDir "sweep_summary.csv"
$summaryJson = Join-Path $sweepDir "sweep_summary.json"

$results |
    Sort-Object -Property @{ Expression = "status"; Descending = $false }, @{ Expression = "rmse_c"; Descending = $false } |
    Export-Csv -Path $summaryCsv -NoTypeInformation -Encoding UTF8

$results | ConvertTo-Json -Depth 4 | Set-Content -Path $summaryJson -Encoding UTF8

$bestOk = $results | Where-Object { $_.status -eq "ok" } | Sort-Object rmse_c | Select-Object -First 1
if ($null -ne $bestOk) {
    Write-Host "Best run: $($bestOk.run_name)  pbw=$($bestOk.physics_balance_weight)  rmse_c=$($bestOk.rmse_c)  mae_c=$($bestOk.mae_c)"
}
else {
    Write-Host "No successful runs in this sweep."
}

Write-Host "Summary CSV: $summaryCsv"
Write-Host "Summary JSON: $summaryJson"
