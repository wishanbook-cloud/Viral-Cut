param(
    [switch]$IncludeAI
)

$ErrorActionPreference = "Stop"

# Build a single-file, windowed Windows executable from this folder.
$ToolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ToolDir

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    throw "Python 3.10+ was not found on PATH. Install it from https://www.python.org/downloads/windows/ and retry."
}
$PythonExecutable = $Python.Source
if ($IncludeAI) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    $AiPython = $null
    if ($PyLauncher) {
        $previousErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $PyLauncher.Source -3.12 -c "import sys" 2>$null
        $pyExitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorPreference
        if ($pyExitCode -eq 0) {
            $AiPython = (& $PyLauncher.Source -3.12 -c "import sys; print(sys.executable)").Trim()
        }
    }
    if (-not $AiPython) {
        $Uv = Get-Command uv -ErrorAction SilentlyContinue
        if ($Uv) {
            $candidate = (& $Uv.Source python find 3.12 2>$null).Trim()
            if ($candidate -and (Test-Path -LiteralPath $candidate)) {
                $AiPython = $candidate
            }
        }
    }
    if (-not $AiPython) {
        throw "The AI build needs Python 3.12 because current faster-whisper wheels do not support this Python version. Install Python 3.12 (or uv) and retry."
    }
    $PythonExecutable = $AiPython
}

$VenvName = if ($IncludeAI) { ".venv-ai" } else { ".venv" }
$VenvDir = Join-Path $ToolDir $VenvName
if (-not (Test-Path -LiteralPath $VenvDir)) {
    & $PythonExecutable -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install --upgrade pyinstaller
if ($LASTEXITCODE -ne 0) { throw "PyInstaller installation failed." }
& $VenvPython -m pip install -r (Join-Path $ToolDir "requirements-ui.txt")
if ($LASTEXITCODE -ne 0) { throw "Desktop UI dependency installation failed." }
if ($IncludeAI) {
    & $VenvPython -m pip install --upgrade -r (Join-Path $ToolDir "requirements-optional.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "The optional AI dependencies could not be installed. The executable was not built with AI features."
    }
}

$OutputName = if ($IncludeAI) { "VideoEditorToolProAI" } else { "VideoEditorTool" }
$DistDirName = if ($IncludeAI) { "dist-pro-ai" } else { "dist" }
$BuildDirName = if ($IncludeAI) { "build-pro-ai" } else { "build" }
$DistDir = Join-Path $ToolDir $DistDirName
$BuildDir = Join-Path $ToolDir $BuildDirName

$PyInstallerArgs = @(
    "--noconfirm",
    "--clean",
    "--onefile",
    "--windowed",
    "--name", $OutputName,
    "--distpath", $DistDir,
    "--workpath", $BuildDir,
    "--specpath", $BuildDir,
    "--collect-all", "customtkinter",
    "--hidden-import", "captions",
    "--hidden-import", "important_words",
    "app.py"
)

# Dynamic engine discovery works beside the exe at runtime. Include it in the
# build too when the developer has supplied engine.py before packaging.
if (Test-Path -LiteralPath (Join-Path $ToolDir "engine.py")) {
    $PyInstallerArgs += @("--hidden-import", "engine")
}

if ($IncludeAI) {
    # These are the runtime packages used by captions/filler removal and face
    # tracking. Silero VAD and its ONNX runtime are deliberately not included:
    # dialogue-only VAD is outside the focused product scope.
    $AiRuntimePackages = @(
        "faster_whisper",
        "ctranslate2",
        "cv2",
        "numpy",
        "tqdm",
        "av",
        "tokenizers",
        "huggingface_hub",
        "yaml",
        "fsspec",
        "filelock"
    )
    foreach ($Package in $AiRuntimePackages) {
        $PyInstallerArgs += @("--collect-all", $Package)
    }
    # The focused product does not use Silero VAD/no-dialogue removal. Exclude
    # its ONNX stack so the executable stays smaller and cannot fail while
    # loading an unrelated optional feature.
    $PyInstallerArgs += @(
        "--exclude-module", "faster_whisper.vad",
        "--exclude-module", "onnxruntime"
    )
}

& $VenvPython -m PyInstaller @PyInstallerArgs
if ($LASTEXITCODE -ne 0) { throw "Executable build failed. Close the app and retry if it is open." }
$FfmpegCommand = Get-Command ffmpeg -ErrorAction SilentlyContinue
$FfprobeCommand = Get-Command ffprobe -ErrorAction SilentlyContinue
if ($FfmpegCommand -and $FfprobeCommand) {
    $ffmpegItem = Get-Item -LiteralPath $FfmpegCommand.Source
    $ffprobeItem = Get-Item -LiteralPath $FfprobeCommand.Source
    $ffmpegSource = if ($ffmpegItem.Target) { $ffmpegItem.Target[0] } else { $ffmpegItem.FullName }
    $ffprobeSource = if ($ffprobeItem.Target) { $ffprobeItem.Target[0] } else { $ffprobeItem.FullName }
    Copy-Item -LiteralPath $ffmpegSource -Destination (Join-Path $DistDir "ffmpeg.exe") -Force
    Copy-Item -LiteralPath $ffprobeSource -Destination (Join-Path $DistDir "ffprobe.exe") -Force
    Write-Host "Bundled FFmpeg and FFprobe beside the executable."
} else {
    Write-Warning "FFmpeg/FFprobe were not found; install them or place them beside the executable."
}
Write-Host "Build complete: $(Join-Path $DistDir ($OutputName + '.exe'))"
