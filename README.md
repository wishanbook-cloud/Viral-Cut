# Video Editor Tool — Focused Local Cleanup

This is a small Windows editor for four local tasks only:

1. Face tracking / vertical reframing
2. Dead-silence removal
3. Filler-word removal
4. Burned-in captions

The UI intentionally has no provider settings, AI flow analysis, model manager,
music, watermark, transition, split-screen, timeline mockup, or decorative
workspace panels. It is a clean source → options → export flow.

## Run

Use the AI package when using captions, filler removal, or face tracking:

```text
video_editor_tool/dist-pro-ai/VideoEditorToolProAI.exe
```

Keep `ffmpeg.exe` and `ffprobe.exe` beside the executable. The lightweight
`dist/VideoEditorTool.exe` does not include Faster-Whisper or OpenCV.

## Workflow

1. Choose a source video.
2. Choose a new `.mp4` output.
3. Enable face tracking, dead-silence removal, filler-word removal, and/or
   captions.
4. If captions are enabled, choose line or word mode and a small caption style.
5. Click **Export**.

Processing is local. The source file is never overwritten. Captioned exports
also write `.srt` and `.ass` sidecars beside the MP4.

## Runtime requirements

- FFmpeg and FFprobe with H.264/AAC and libass support
- `opencv-python` for face tracking
- `faster-whisper` and its CPU runtime for filler removal/captions
- Python 3.12 for rebuilding the AI package

The focused path uses CPU `int8` transcription and the small `tiny` model by
default to keep startup and memory use down. Models download on first use and
are cached by Faster-Whisper; the application does not upload footage.

## Build

```powershell
cd "C:\Users\Silent Wolf\omnirush.ai\video_editor_tool"
powershell -NoProfile -ExecutionPolicy Bypass -File .\build_windows.ps1 -IncludeAI
```

The script writes the AI package to `dist-pro-ai` and the lightweight package to
`dist`. It copies FFmpeg and FFprobe beside the matching executable.

## Checks

```powershell
.\.venv-ai\Scripts\python.exe -m py_compile app.py engine.py test_engine.py
.\.venv-ai\Scripts\python.exe -m unittest discover -p "test_*.py"
.\.venv-ai\Scripts\python.exe app.py --smoke-test
.\.venv-ai\Scripts\python.exe .\packaging_smoke_test.py
```

The packaging smoke test checks the actual bundled imports used by the focused
features: OpenCV, NumPy, Faster-Whisper, tqdm, and CTranslate2.
