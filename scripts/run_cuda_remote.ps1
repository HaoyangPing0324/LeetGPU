param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$Problem
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$EnvFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Missing environment file: $EnvFile"
}

Get-Content -LiteralPath $EnvFile | ForEach-Object {
    $Line = $_.Trim()
    if ($Line -and -not $Line.StartsWith("#") -and $Line.Contains("=")) {
        $Parts = $Line.Split("=", 2)
        $Name = $Parts[0].Trim()
        $Value = $Parts[1].Trim()
        if (($Value.StartsWith('"') -and $Value.EndsWith('"')) -or
            ($Value.StartsWith("'") -and $Value.EndsWith("'"))) {
            $Value = $Value.Substring(1, $Value.Length - 2)
        }
        Set-Item -Path "Env:$Name" -Value $Value
    }
}

foreach ($Name in @("LOCAL_PROJECT_PATH", "REMOTE_OS", "REMOTE_HOST", "REMOTE_USER", "REMOTE_ROOT", "REMOTE_NVCC")) {
    if (-not [Environment]::GetEnvironmentVariable($Name)) {
        throw "Set $Name in scripts/.env"
    }
}

$RemoteOs = $env:REMOTE_OS.ToLowerInvariant()
if ($RemoteOs -notin @("windows", "linux")) {
    throw "REMOTE_OS must be windows or linux"
}

if ([IO.Path]::IsPathRooted($env:LOCAL_PROJECT_PATH)) {
    $ProjectDir = (Resolve-Path -LiteralPath $env:LOCAL_PROJECT_PATH).Path
} else {
    $ProjectDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot $env:LOCAL_PROJECT_PATH)).Path
}

$SourceFile = Join-Path $ProjectDir "src/cuda/$Problem.cu"
$TestFile = Join-Path $ProjectDir "test/cuda/$Problem.cu"
if (-not (Test-Path -LiteralPath $SourceFile) -or -not (Test-Path -LiteralPath $TestFile)) {
    throw "Source or test file not found for $Problem"
}

foreach ($Command in @("ssh", "scp")) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Command"
    }
}

$Port = if ($env:REMOTE_PORT) { $env:REMOTE_PORT } else { "22" }
$CommonOptions = @("-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=accept-new")
$SshOptions = @("-p", $Port) + $CommonOptions
$ScpOptions = @("-P", $Port) + $CommonOptions
if ($env:SSH_KEY_PATH) {
    $SshOptions += @("-i", $env:SSH_KEY_PATH)
    $ScpOptions += @("-i", $env:SSH_KEY_PATH)
}

$UseSshPass = $false
if ($env:REMOTE_PASSWORD) {
    if (Get-Command sshpass -ErrorAction SilentlyContinue) {
        $env:SSHPASS = $env:REMOTE_PASSWORD
        $UseSshPass = $true
    } else {
        Write-Warning "sshpass is unavailable; native OpenSSH will use key, agent, or interactive password authentication."
    }
}

$RemoteTarget = "$($env:REMOTE_USER)@$($env:REMOTE_HOST)"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ExecutionTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
$RunId = "${Timestamp}_$PID"
$RemoteDir = "$($env:REMOTE_ROOT)/${Problem}_${RunId}"
$RemoteDirWin = $RemoteDir.Replace("/", "\")
$ResultDir = Join-Path $ProjectDir "result/cuda"
$ResultFile = Join-Path $ResultDir "$Problem.txt"
$SummaryFile = Join-Path $ResultDir "history.md"
New-Item -ItemType Directory -Force -Path $ResultDir | Out-Null

if (-not (Test-Path -LiteralPath $SummaryFile) -or (Get-Item -LiteralPath $SummaryFile).Length -eq 0) {
    @(
        "# CUDA Run History",
        "",
        "| Execution Time | Problem | Platform | Status | Average | Maximum | Minimum |",
        "| --- | --- | --- | :---: | ---: | ---: | ---: |"
    ) | Set-Content -LiteralPath $SummaryFile
}

function Invoke-SshCommand([string]$Command) {
    if ($UseSshPass) {
        & sshpass -e ssh @SshOptions $RemoteTarget $Command
    } else {
        & ssh @SshOptions $RemoteTarget $Command
    }
    $script:NativeExitCode = $LASTEXITCODE
}

function Invoke-ScpCommand([string]$LocalFile, [string]$RemotePath) {
    if ($UseSshPass) {
        & sshpass -e scp @ScpOptions $LocalFile "${RemoteTarget}:$RemotePath"
    } else {
        & scp @ScpOptions $LocalFile "${RemoteTarget}:$RemotePath"
    }
    $script:NativeExitCode = $LASTEXITCODE
}

function Write-Log([string]$Message) {
    Write-Host $Message
    Add-Content -LiteralPath $ResultFile -Value $Message
}

@(
    "============================================================",
    "CUDA problem: $Problem",
    "============================================================"
) | Set-Content -LiteralPath $ResultFile
Get-Content -LiteralPath $ResultFile | ForEach-Object { Write-Host $_ }

$Status = "PASS"
$RemoteCreated = $false

Write-Log "[1/4] Creating remote directories..."
if ($RemoteOs -eq "windows") {
    $CreateCommand = "powershell.exe -NoProfile -NonInteractive -Command `"[void](New-Item -ItemType Directory -Force -Path '$RemoteDir/src/cuda','$RemoteDir/test/cuda')`""
} else {
    $CreateCommand = "mkdir -p '$RemoteDir/src/cuda' '$RemoteDir/test/cuda'"
}
Invoke-SshCommand $CreateCommand 2>&1 | Tee-Object -LiteralPath $ResultFile -Append
if ($NativeExitCode -ne 0) { $Status = "FAIL" } else { $RemoteCreated = $true }

if ($Status -eq "PASS") {
    Write-Log "[2/4] Uploading source and test files..."
    Invoke-ScpCommand $SourceFile "$RemoteDir/src/cuda/$Problem.cu" 2>&1 | Tee-Object -LiteralPath $ResultFile -Append
    if ($NativeExitCode -eq 0) {
        Invoke-ScpCommand $TestFile "$RemoteDir/test/cuda/$Problem.cu" 2>&1 | Tee-Object -LiteralPath $ResultFile -Append
    }
    if ($NativeExitCode -ne 0) { $Status = "FAIL" }
}

if ($Status -eq "PASS") {
    Write-Log "[3/4] Compiling and running..."
    if ($RemoteOs -eq "windows") {
        $RemoteCommand = "cd /d `"$RemoteDirWin`""
        if ($env:REMOTE_VCVARS) { $RemoteCommand += " && call `"$($env:REMOTE_VCVARS)`" >nul" }
        $RemoteCommand += " && echo === GPU === && nvidia-smi.exe --query-gpu=name,driver_version,memory.total --format=csv,noheader"
        $RemoteCommand += " && echo === Compile: $Problem === && `"$($env:REMOTE_NVCC)`" -O2"
        if ($env:REMOTE_CCBIN) { $RemoteCommand += " -ccbin `"$($env:REMOTE_CCBIN)`"" }
        $RemoteCommand += " -o cuda_app.exe `"src\cuda\$Problem.cu`" `"test\cuda\$Problem.cu`""
        $RemoteCommand += " && echo === Program output: $Problem === && cuda_app.exe"
    } else {
        $RemoteCommand = "cd '$RemoteDir' && echo '=== GPU ===' && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"
        $RemoteCommand += " && echo '=== Compile: $Problem ===' && `"$($env:REMOTE_NVCC)`" -O2"
        if ($env:REMOTE_CCBIN) { $RemoteCommand += " -ccbin `"$($env:REMOTE_CCBIN)`"" }
        $RemoteCommand += " -o cuda_app 'src/cuda/$Problem.cu' 'test/cuda/$Problem.cu'"
        $RemoteCommand += " && echo '=== Program output: $Problem ===' && ./cuda_app"
    }
    Invoke-SshCommand $RemoteCommand 2>&1 | Tee-Object -LiteralPath $ResultFile -Append
    if ($NativeExitCode -ne 0) { $Status = "FAIL" }
}

if ($env:KEEP_REMOTE -eq "1") {
    Write-Log "[4/4] Remote files retained at: $RemoteDir"
} elseif ($RemoteCreated) {
    Write-Log "[4/4] Cleaning remote files..."
    if ($RemoteOs -eq "windows") {
        $CleanupCommand = "powershell.exe -NoProfile -NonInteractive -Command `"Remove-Item -LiteralPath '$RemoteDir' -Recurse -Force`""
    } else {
        $CleanupCommand = "rm -rf -- '$RemoteDir'"
    }
    Invoke-SshCommand $CleanupCommand 2>&1 | Tee-Object -LiteralPath $ResultFile -Append
}

Write-Log "Result: $Status - $Problem"
$Lines = Get-Content -LiteralPath $ResultFile
$Platform = "N/A"
for ($Index = 0; $Index -lt $Lines.Count - 1; $Index++) {
    if ($Lines[$Index] -eq "=== GPU ===") {
        $Platform = ($Lines[$Index + 1] -split ",")[0].Trim()
        break
    }
}
$Average = "N/A"; $Maximum = "N/A"; $Minimum = "N/A"
$PerfLine = $Lines | Where-Object { $_ -match '^\[PERF\]' } | Select-Object -Last 1
if ($PerfLine -match 'avg=([0-9]+(?:\.[0-9]+)?)\s*ms') { $Average = "$($Matches[1]) ms" }
if ($PerfLine -match 'max=([0-9]+(?:\.[0-9]+)?)\s*ms') { $Maximum = "$($Matches[1]) ms" }
if ($PerfLine -match 'min=([0-9]+(?:\.[0-9]+)?)\s*ms') { $Minimum = "$($Matches[1]) ms" }
Add-Content -LiteralPath $SummaryFile -Value "| $ExecutionTime | ``$Problem`` | $Platform | **$Status** | $Average | $Maximum | $Minimum |"

if ($Status -ne "PASS") { exit 1 }
