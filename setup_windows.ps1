#Requires -Version 5.1
<#
    Jumper Studio -- one-click Windows setup.

    Sets up BOTH environments the app needs:

      1. the app itself        (.venv in this folder: PySide6/Sports2D/MuJoCo,
                                optional CUDA torch, robot assets, shortcut)
      2. PPO training          (a Jumper/mjlab checkout at -JumperDir with its
                                own .venv, CUDA torch + warp + mjlab, which the
                                training page invokes directly)

    The GPU training is Jumper's mjlab/rsl_rl stack (see SETUP_WINDOWS.md);
    the research `allmimic` SuperMimic port is G1-only and is NOT installed.

    Usage (from this folder):
        powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1

    Useful options:
        -Action install|diagnose|uninstall
        -Components app,jumper,patch,pretrain,shortcut
        -Reinstall / -FromApp
        -JumperDir D:\jumper_ws     where the training checkout lives
        -Python C:\...\python.exe   interpreter to build the venvs with
        -TorchBuild auto|cu126|cu128|cpu
        -SkipApp / -SkipJumper / -SkipTorch / -NoShortcut / -NoMirror
        -GithubMirror https://ghfast.top/
        -Force                      continue even if a training run is active

    Safe to re-run: existing venvs, packages and assets are kept and skipped.
    It refuses to run while a training/app process is alive unless -Force.
#>
[CmdletBinding()]
param(
    [string]$ProjectDir = "",
    [string]$JumperDir = "D:\jumper_ws",
    [string]$Python = "",
    [ValidateSet("auto", "cu126", "cu128", "cpu")]
    [string]$TorchBuild = "auto",
    [string]$GithubMirror = "https://ghfast.top/",
    [switch]$SkipApp,
    [switch]$SkipJumper,
    [switch]$SkipTorch,
    [switch]$NoShortcut,
    [switch]$NoMirror,
    [switch]$Force,
    [ValidateSet("install", "diagnose", "uninstall")]
    [string]$Action = "install",
    [string]$Components = "",
    [switch]$Reinstall,
    [switch]$FromApp
)

$ErrorActionPreference = "Stop"
if (-not $ProjectDir) { $ProjectDir = $PSScriptRoot }
$LogPath = Join-Path $ProjectDir "setup_windows.log"
try { Start-Transcript -Path $LogPath -Append | Out-Null } catch { }

function Step([string]$Text) {
    Write-Host ""
    Write-Host ("== " + $Text) -ForegroundColor Cyan
}
function Info([string]$Text) { Write-Host ("   " + $Text) }
function Warn([string]$Text) { Write-Host ("   ! " + $Text) -ForegroundColor Yellow }

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = ""
    )
    Info ("> " + $Exe + " " + ($Arguments -join " "))
    $old = $null
    if ($WorkingDirectory) { $old = Get-Location; Set-Location $WorkingDirectory }
    try {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw ("command failed with exit code {0}: {1} {2}" -f $LASTEXITCODE, $Exe, ($Arguments -join " "))
        }
    }
    finally {
        if ($WorkingDirectory) { Set-Location $old }
    }
}

function Test-CommandExists([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# --------------------------------------------------------------- Python ----
function Find-Python {
    $candidates = @()
    if ($Python) { $candidates += , @($Python) }
    $candidates += , @("py", "-3.11")
    $candidates += , @("py", "-3.12")
    $candidates += , @("py", "-3.13")
    $candidates += , @("python")
    foreach ($cand in $candidates) {
        $exe = $cand[0]
        $pre = @()
        if ($cand.Count -gt 1) { $pre = $cand[1..($cand.Count - 1)] }
        if (-not (Test-CommandExists $exe)) { continue }
        try {
            $ver = (& $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null)
            # sports2d 0.8.x needs >=3.11; Jumper needs <3.14. 3.11 is tested.
            if ($LASTEXITCODE -eq 0 -and $ver -match '^3\.1[1-3]$') {
                return @{ Exe = $exe; Pre = $pre; Version = $ver }
            }
        }
        catch { }
    }
    throw "No Python 3.11-3.13 found. Install Python 3.11 from python.org and tick 'Add python.exe to PATH'."
}

# ----------------------------------------------------------------- GPU ----
function Get-GpuInfo {
    if (-not (Test-CommandExists "nvidia-smi")) { return $null }
    try {
        $line = (& nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and $line) { return $line.Trim() }
    }
    catch { }
    return $null
}

function Resolve-TorchBuild([string]$gpu) {
    if ($TorchBuild -ne "auto") { return $TorchBuild }
    if (-not $gpu) { return "cpu" }
    if ($gpu -match ',\s*12\.\d') { return "cu128" }   # Blackwell
    return "cu126"                                      # Ada / Ampere
}

function Get-TorchIndex([string]$build) {
    if ($build -eq "cpu") { return "" }
    return ("https://download.pytorch.org/whl/" + $build)
}

function Test-TorchCuda([string]$Py) {
    try {
        & $Py -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 3)" 2>$null
        return ($LASTEXITCODE -eq 0)
    }
    catch { return $false }
}

function Install-Torch([string]$Py, [string]$build) {
    $index = Get-TorchIndex $build
    $args = @("-m", "pip", "install", "--upgrade", "torch", "torchvision")
    if ($index) { $args += @("--index-url", $index) }
    Invoke-Checked -Exe $Py -Arguments $args
}

# --------------------------------------------------------- safety gate ----
function Assert-NotBusy {
    $hits = @()
    try {
        $procs = Get-CimInstance Win32_Process -ErrorAction Stop |
            Where-Object { $_.Name -in @("python.exe", "pythonw.exe") }
        foreach ($p in $procs) {
            $cmd = [string]$p.CommandLine
            if (-not $cmd) { continue }
            if ($cmd -match 'scripts[\\/](train|play|export)\.py' -or
                $cmd -match [regex]::Escape((Join-Path $ProjectDir "run.py")) -or
                ($JumperDir -and $cmd -match [regex]::Escape($JumperDir))) {
                $hits += $p
            }
        }
    }
    catch { return }
    if ($hits.Count -eq 0) { return }
    Warn "A training / app process is running:"
    foreach ($p in $hits) {
        Warn ("  PID {0}: {1}" -f $p.ProcessId, $p.CommandLine)
    }
    if (-not $Force) {
        Warn "Refusing to modify environments while a run is active."
        Warn "Stop it first, or re-run with -Force (not recommended during training)."
        exit 2
    }
}

# -------------------------------------------------------------- app ----
function Setup-App($Py) {
    Step "crab-dance app environment"
    $venv = Join-Path $ProjectDir ".venv"
    $appPy = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $appPy)) {
        Invoke-Checked -Exe $Py.Exe -Arguments ($Py.Pre + @("-m", "venv", $venv))
    }
    else {
        Info "venv already exists: $venv"
    }
    Invoke-Checked -Exe $appPy -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    $req = Join-Path $ProjectDir "requirements.txt"
    $pipArgs = @("-m", "pip", "install", "-r", $req)
    if ($Reinstall) { $pipArgs = @("-m", "pip", "install", "--force-reinstall", "-r", $req) }
    Invoke-Checked -Exe $appPy -Arguments $pipArgs
    if (-not $SkipTorch) {
        if ($Force -or -not (Test-TorchCuda $appPy)) {
            Install-Torch $appPy $script:ChosenTorch
        }
        else { Info "CUDA torch already works in the app venv." }
    }

    $xml = Join-Path $ProjectDir "assets\jumper\jumper.xml"
    if (-not (Test-Path $xml)) {
        Step "Robot assets (jumper.xml + meshes)"
        if (-not $NoMirror) {
            $env:JUMPER_MIRROR = ($GithubMirror.TrimEnd("/") + "/https://raw.githubusercontent.com")
            Info "JUMPER_MIRROR=$env:JUMPER_MIRROR"
        }
        Invoke-Checked -Exe $appPy -Arguments @("run.py", "--fetch") -WorkingDirectory $ProjectDir
    }
    else { Info "Robot assets already present." }

    Step "App self-test"
    Invoke-Checked -Exe $appPy -Arguments @("run.py", "--selftest") -WorkingDirectory $ProjectDir

}

function Setup-Shortcut {
    if ($NoShortcut) { return }
    $venv = Join-Path $ProjectDir ".venv"
    $appPy = Join-Path $venv "Scripts\python.exe"
    $pythonw = Join-Path $venv "Scripts\pythonw.exe"
    if (-not (Test-Path $pythonw)) { $pythonw = $appPy }
    $lnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "Jumper Studio.lnk"
    try {
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($lnk)
        $sc.TargetPath = $pythonw
        $sc.Arguments = "run.py"
        $sc.WorkingDirectory = $ProjectDir
        $sc.Description = "Jumper Studio - Sports2D to Jumper crab"
        $sc.Save()
        Info "Desktop shortcut: $lnk"
    }
    catch { Warn ("Could not create the shortcut: " + $_.Exception.Message) }
}

function Setup-Pretrain {
    $base = Join-Path $ProjectDir "assets\pretrained\dance_base.pt"
    $meta = Join-Path $ProjectDir "assets\pretrained\dance_base.json"
    if (-not (Test-Path $base)) {
        throw "The bundled pretrain weight is missing: $base"
    }
    if (-not (Test-Path $meta)) { Warn "pretrain metadata is missing: $meta" }
    Info "Pretrain weight: $base"
}

function Setup-Patch($Py) {
    $play = Join-Path $JumperDir "scripts\play.py"
    if (-not (Test-Path $play)) { throw "play.py not found: $play" }
    $appPy = Join-Path $ProjectDir ".venv\Scripts\python.exe"
    if (-not (Test-Path $appPy)) { $appPy = $Py.Exe }
    Invoke-Checked -Exe $appPy -Arguments @(
        (Join-Path $ProjectDir "tools\patch_play_states.py"),
        "--play", $play)
}

# ------------------------------------------------------------ Jumper ----
function Setup-Jumper($Py) {
    Step "SuperMimic / Jumper training environment"
    if (-not (Test-Path (Join-Path $JumperDir "scripts\train.py"))) {
        if (Test-Path $JumperDir) {
            $entries = @(Get-ChildItem $JumperDir -Force -ErrorAction SilentlyContinue)
            if ($entries.Count -gt 0) {
                throw "$JumperDir exists but is not a Jumper checkout. Move it away or pass -JumperDir."
            }
        }
        if (-not (Test-CommandExists "git")) {
            throw "git is required to clone Jumper. Install Git for Windows first."
        }
        $urls = @("https://github.com/KingKongRobotics/jumper.git")
        if (-not $NoMirror) {
            $urls += ($GithubMirror.TrimEnd("/") + "/https://github.com/KingKongRobotics/jumper.git")
        }
        $cloned = $false
        foreach ($url in $urls) {
            Info "Cloning $url"
            try {
                git clone $url $JumperDir
                if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $JumperDir "scripts\train.py"))) {
                    $cloned = $true
                    break
                }
            }
            catch { Warn ("clone failed: " + $_.Exception.Message) }
        }
        if (-not $cloned) {
            throw "Could not clone Jumper (GitHub is blocked on this network; the mirror is $GithubMirror)."
        }
    }
    else { Info "Jumper already present: $JumperDir" }

    $jvenv = Join-Path $JumperDir ".venv"
    $jpy = Join-Path $jvenv "Scripts\python.exe"
    if ($Reinstall -and (Test-Path $jvenv)) {
        Info "Recreating the training virtual environment."
        Remove-Item $jvenv -Recurse -Force
    }
    if (-not (Test-Path $jpy)) {
        Invoke-Checked -Exe $Py.Exe -Arguments ($Py.Pre + @("-m", "venv", $jvenv))
    }
    else { Info "venv already exists: $jvenv" }
    Invoke-Checked -Exe $jpy -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    if (-not $SkipTorch) {
        if ($Force -or -not (Test-TorchCuda $jpy)) {
            Install-Torch $jpy $script:ChosenTorch
        }
        else { Info "CUDA torch already works in the training venv." }
    }

    # The vendored mjlab / rsl_rl under rl/ are only importable through this
    # editable install. Never `pip install mjlab` / `rsl-rl-lib` here.
    $installed = $false
    try {
        & $jpy -c "import mjrl, mjlab, rsl_rl" 2>$null
        $installed = ($LASTEXITCODE -eq 0)
    }
    catch { $installed = $false }
    if ($Force -or -not $installed) {
        Invoke-Checked -Exe $jpy -Arguments @("-m", "pip", "install", "-e", ".") -WorkingDirectory $JumperDir
    }
    else { Info "mjrl-lab already installed (editable)." }

    Step "Training stack check"
    $list = ""
    try {
        $list = (& $jpy "scripts/train.py" "--list" 2>&1 | Out-String)
    }
    catch {
        throw ("scripts/train.py --list failed -- see docs/AGENT_SETUP.md section 3: " + $_.Exception.Message)
    }
    if ($LASTEXITCODE -ne 0) {
        throw "scripts/train.py --list failed; see docs/AGENT_SETUP.md section 3."
    }
    if ($list -notmatch 'jumper\.dance') {
        throw "The task list has no jumper.dance; the checkout looks broken."
    }
    Info "jumper.dance is registered."
    Info "The app creates jumper.dance_crab (with your exported clip) on its first GPU run."
    Warn "For manual runs, set TORCHDYNAMO_DISABLE=1 (Windows has no Triton)."
    Warn ("Workspace for the app: set CRAB_JUMPER_WS=" + $JumperDir + " if it is not the default.")
}

# ---------------------------------------------------------------- main ----
Step "Jumper Studio environment"
Info ("Project: " + $ProjectDir)
Info ("Jumper:  " + $JumperDir)
if (-not (Test-Path (Join-Path $ProjectDir "run.py"))) {
    throw "run.py not found under $ProjectDir -- run this script from the Jumper Studio folder."
}

$script:Want = @()
if ($Components) {
    $script:Want = @($Components.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
$script:AllComponents = ($script:Want.Count -eq 0)
function Wanted([string]$Name) {
    if ($script:AllComponents) { return $true }
    return ($script:Want -contains $Name)
}

if ($Action -eq "uninstall") {
    $un = Join-Path $ProjectDir "uninstall_windows.ps1"
    if (-not (Test-Path $un)) { throw "uninstall_windows.ps1 not found: $un" }
    $uargs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $un,
               "-Components", $Components, "-JumperDir", $JumperDir, "-FromApp")
    if ($Yes) { $uargs += "-Yes" }
    & powershell @uargs
    exit $LASTEXITCODE
}

$appPy = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$jpy = Join-Path $JumperDir ".venv\Scripts\python.exe"
$play = Join-Path $JumperDir "scripts\play.py"
$shortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "Jumper Studio.lnk"
$pretrainPath = Join-Path $ProjectDir "assets\pretrained\dance_base.pt"

if ($Action -eq "diagnose") {
    Step "Environment diagnosis"
    $allOk = $true
    function Check([string]$Name, [bool]$Pass, [string]$Detail) {
        if ($Pass) { Write-Host ("   [OK]   " + $Name + "  " + $Detail) -ForegroundColor Green }
        else { Write-Host ("   [--]   " + $Name + "  " + $Detail) -ForegroundColor Yellow; $script:DiagOk = $false }
    }
    $script:DiagOk = $true
    Check "App venv" (Test-Path $appPy) $appPy
    $deps = $false
    if (Test-Path $appPy) {
        try { & $appPy -c "import PySide6, cv2, Sports2D, mujoco" 2>$null; $deps = ($LASTEXITCODE -eq 0) } catch { }
    }
    Check "App dependencies" $deps "PySide6 / cv2 / sports2d / mujoco"
    Check "Robot assets" (Test-Path (Join-Path $ProjectDir "assets\jumper\jumper.xml")) "jumper.xml"
    Check "Pretrain" (Test-Path $pretrainPath) $pretrainPath
    Check "Jumper checkout" (Test-Path (Join-Path $JumperDir "scripts\train.py")) $JumperDir
    Check "Training venv" (Test-Path $jpy) $jpy
    $cuda = $false
    if (Test-Path $jpy) { $cuda = Test-TorchCuda $jpy }
    Check "CUDA torch" $cuda ""
    $mjrl = $false
    if (Test-Path $jpy) { try { & $jpy -c "import mjrl, mjlab, rsl_rl" 2>$null; $mjrl = ($LASTEXITCODE -eq 0) } catch { } }
    Check "mjrl/mjlab" $mjrl ""
    $patched = $false
    if (Test-Path $play) { $patched = ((Get-Content $play -Raw) -match "--states-out") }
    Check "Replay patch" $patched "play.py --states-out"
    Check "Shortcut" (Test-Path $shortcutPath) $shortcutPath
    if ($script:DiagOk) { Info "Training environment is ready."; exit 0 }
    Warn "Some components are missing; run without -Action diagnose to install/repair."
    exit 1
}

Assert-NotBusy

$py = Find-Python
Info ("Python: " + $py.Exe + " " + ($py.Pre -join " ") + " -> " + $py.Version)
$gpu = Get-GpuInfo
if ($gpu) { Info ("GPU: " + $gpu) } else { Warn "No NVIDIA GPU detected (nvidia-smi missing); GPU training will be unavailable." }
$script:ChosenTorch = Resolve-TorchBuild $gpu
Info ("Torch build: " + $script:ChosenTorch)
if (-not (Test-CommandExists "cl")) {
    Warn "MSVC (cl.exe) not on PATH. Warp JIT may need 'Desktop development with C++'"
    Warn "from Visual Studio Build Tools; install it only if a warp compile error appears."
}

$doApp = (Wanted "app") -and (-not $SkipApp)
$doJumper = (Wanted "jumper") -and (-not $SkipJumper)
$doPatch = (Wanted "patch") -or $doJumper
$doPretrain = Wanted "pretrain"
$doShortcut = (Wanted "shortcut") -and (-not $NoShortcut)

if ($doApp) { Setup-App $py } else { Step "Skipping the app environment" }
if ($doJumper) { Setup-Jumper $py } else { Step "Skipping the PPO training environment" }
if ($doPatch) { Setup-Patch $py }
if ($doPretrain) { Setup-Pretrain }
if ($doShortcut) { Setup-Shortcut }

Step "Done"
Info "Launch the app with the desktop shortcut (Jumper Studio) or .venv\Scripts\pythonw.exe run.py"
Info "PPO training: training page -> method PPO -> start."
Info ("Full log: " + $LogPath)
try { Stop-Transcript | Out-Null } catch { }
