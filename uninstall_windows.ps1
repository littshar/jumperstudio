#Requires -Version 5.1
<#
    Jumper Studio -- uninstall selected environment parts.

    Components:
        app       project .venv (run only after closing Jumper Studio)
        jumper    D:\jumper_ws\.venv (the PPO training virtual environment)
        shortcut  desktop shortcut
        cache     %USERPROFILE%\.crab_dance_cache
        patch     (no-op; the play.py patch is left in place)
        pretrain  (no-op; the bundled weight is part of the app)

    Examples:
        powershell -NoProfile -ExecutionPolicy Bypass -File .\uninstall_windows.ps1
        powershell ... -File .\uninstall_windows.ps1 -Components jumper,shortcut -Yes
        powershell ... -File .\uninstall_windows.ps1 -DeleteJumperRepo

    -DeleteJumperRepo also removes the Jumper checkout itself (it contains
    your trained checkpoints under logs/ -- this cannot be undone).
#>
[CmdletBinding()]
param(
    [string]$ProjectDir = "",
    [string]$JumperDir = "D:\jumper_ws",
    [string]$Components = "app,jumper,shortcut,cache",
    [switch]$Force,
    [switch]$Yes,
    [switch]$FromApp,
    [switch]$DeleteJumperRepo,
    [string]$Action = "uninstall"
)

$ErrorActionPreference = "Stop"
if (-not $ProjectDir) { $ProjectDir = $PSScriptRoot }
$LogPath = Join-Path $ProjectDir "uninstall_windows.log"
try { Start-Transcript -Path $LogPath -Append | Out-Null } catch { }

function Step([string]$Text) {
    Write-Host ""
    Write-Host ("== " + $Text) -ForegroundColor Cyan
}
function Info([string]$Text) { Write-Host ("   " + $Text) }
function Warn([string]$Text) { Write-Host ("   ! " + $Text) -ForegroundColor Yellow }

$want = @($Components.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
if ($FromApp) {
    $want = @($want | Where-Object { $_ -ne "app" })
}
function Wanted([string]$Name) { return ($want -contains $Name) }

function Test-Running([string]$Pattern) {
    try {
        $procs = Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object { $_.Name -in @("python.exe", "pythonw.exe") }
        foreach ($p in $procs) {
            $cmd = [string]$p.CommandLine
            if ($cmd -and $cmd -match $Pattern) { return $p }
        }
    }
    catch { }
    return $null
}

Step "Jumper Studio uninstall"
Info ("Project: " + $ProjectDir)
Info ("Parts:   " + ($want -join ", "))

$train = Test-Running 'scripts[\\/](train|play|export)\.py'
if ($train -and -not $Force) {
    Warn ("A training/replay process is active (PID {0}); stop it first." -f $train.ProcessId)
    exit 2
}
$app = $null
if (-not $FromApp) { $app = Test-Running ([regex]::Escape((Join-Path $ProjectDir "run.py"))) }
if ($app -and -not $Force) {
    Warn "Jumper Studio is running. Close it before uninstalling the app environment."
    exit 2
}

if (-not $Yes) {
    Write-Host ""
    Write-Host "This will remove:" -ForegroundColor Yellow
    if (Wanted "app") { Write-Host ("  - " + (Join-Path $ProjectDir ".venv")) }
    if (Wanted "jumper") { Write-Host ("  - " + (Join-Path $JumperDir ".venv")) }
    if ($DeleteJumperRepo) { Write-Host ("  - the ENTIRE Jumper checkout " + $JumperDir) -ForegroundColor Red }
    if (Wanted "shortcut") { Write-Host "  - desktop shortcut" }
    if (Wanted "cache") { Write-Host ("  - " + (Join-Path $env:USERPROFILE ".crab_dance_cache")) }
    $answer = Read-Host "Continue? [y/N]"
    if ($answer -notmatch '^(y|yes)$') { Info "Cancelled."; exit 0 }
}

function Remove-Dir([string]$Path, [string]$Label) {
    if (-not (Test-Path $Path)) { Info ($Label + " already absent."); return }
    try {
        Remove-Item $Path -Recurse -Force
        Info ($Label + " removed.")
    }
    catch { Warn ("Could not remove " + $Path + ": " + $_.Exception.Message) }
}

if (Wanted "app") { Remove-Dir (Join-Path $ProjectDir ".venv") "app .venv" }
if (Wanted "jumper") { Remove-Dir (Join-Path $JumperDir ".venv") "training .venv" }
if ($DeleteJumperRepo) { Remove-Dir $JumperDir "Jumper checkout" }
if (Wanted "shortcut") {
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Jumper Studio.lnk"
    if (Test-Path $lnk) { Remove-Item $lnk -Force; Info "desktop shortcut removed." }
    else { Info "desktop shortcut already absent." }
}
if (Wanted "cache") {
    Remove-Dir (Join-Path $env:USERPROFILE ".crab_dance_cache") "app cache"
}

Step "Done"
Info "Source files, assets and the bundled pretrain weight were kept."
try { Stop-Transcript | Out-Null } catch { }
