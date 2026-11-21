#Requires -Version 5.1
<#
    Build a clean, unzip-and-run release of Jumper Studio.

    The staged folder excludes local virtual environments, caches, logs,
    screenshots and training outputs. The recipient runs setup_windows.bat (or
    start_windows.bat) once; the script fetches the external Jumper workspace.
#>
[CmdletBinding()]
param(
    [string]$ProjectDir = "",
    [string]$OutDir = "dist",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
if (-not $ProjectDir) { $ProjectDir = $PSScriptRoot }
$stage = Join-Path (Join-Path $ProjectDir $OutDir) "JumperStudio"
$dist = Join-Path $ProjectDir $OutDir

if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null

$excludeDirs = @(".venv", "__pycache__", ".cache", ".git", "dist", "logs", $OutDir)
$excludeFiles = @("*.pyc", "*.log", "MUJOCO_LOG.TXT", "_*.npz", "_*.mp4")

$copyArgs = @($ProjectDir, $stage, "/E", "/NFL", "/NDL", "/NJH", "/NJS",
              "/XD") + $excludeDirs + @("/XF") + $excludeFiles
& robocopy @copyArgs | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }

# Test artifacts and screenshots are development-only.
Get-ChildItem (Join-Path $stage "tests") -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in @(".png", ".mp4", ".jpg") } |
    Remove-Item -Force
Get-ChildItem $stage -File -Filter "gui_*.png" | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem $stage -File -Filter "*.npz" | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem $stage -File -Filter "*.mp4" | Remove-Item -Force -ErrorAction SilentlyContinue

if (-not $NoZip) {
    $zip = Join-Path $dist "JumperStudio.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zip
    Write-Host "Release ZIP: $zip"
} else {
    Write-Host "Release folder: $stage"
}
