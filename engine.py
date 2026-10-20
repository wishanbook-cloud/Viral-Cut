"""A small, local FFmpeg-powered video editing engine.

The module deliberately has no cloud or third-party dependencies.  It expects
``ffmpeg`` and ``ffprobe`` to be installed and available on ``PATH`` (or at
the paths supplied in :class:`EditorConfig`).  The main flow is:

    editor = VideoEditor()
    result = editor.edit("input.mp4", "edited.mp4", filler_ranges=[(12.4, 13.1)])

``filler_ranges`` normally comes from a transcription layer.  Caption
generation is intentionally separate; pass an SRT path to
``run_caption_hook`` when an application has its own caption implementation.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


ProgressCallback = Callable[["ProgressEvent"], None]


class CaptionHook(Protocol):
    """Application-provided caption integration.

    The engine does not generate captions.  A transcription/caption layer can
    implement this callable and receive the SRT path through
    :func:`run_caption_hook`.
    """

    def __call__(self, srt_path: Path) -> Any:
        ...


class FFmpegError(RuntimeError):
    """Raised when FFmpeg or FFprobe cannot complete an operation."""


class DependencyError(FFmpegError):
    """Raised when an optional local processing dependency is unavailable."""


class MediaProbeError(FFmpegError):
    """Raised when media metadata is missing or cannot be understood."""


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """A progress update emitted by the editor."""

    phase: str
    fraction: float
    message: str = ""


@dataclass(frozen=True, slots=True)
class TimeRange:
    """A half-open media range in seconds: ``start <= t < end``."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError("TimeRange values must be finite")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("TimeRange requires 0 <= start < end")

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class Cut:
    """A source interval retained in the output timeline."""

    source_start: float
    source_end: float
    timeline_start: float

    def __post_init__(self) -> None:
        if self.source_start < 0 or self.source_end <= self.source_start:
            raise ValueError("Cut requires 0 <= source_start < source_end")
        if self.timeline_start < 0:
            raise ValueError("Cut timeline_start must be non-negative")

    @property
    def duration(self) -> float:
        return self.source_end - self.source_start


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """Useful metadata returned by FFprobe."""

    path: Path
    duration: float
    width: int
    height: int
    frame_rate: float | None
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool


@dataclass(frozen=True, slots=True)
class EditorConfig:
    """Local editing and encoding options."""

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    silence_threshold_db: float = -35.0
    silence_min_duration: float = 0.45
    remove_silence: bool = True
    silence_padding: float = 0.0
    filler_padding: float = 0.0
    speech_padding: float = 0.12
    minimum_clip_duration: float = 0.05
    zoom_factor: float | None = None
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    preset: str = "medium"
    crf: int = 20
    transition: str = "none"
    transition_duration: float = 0.16

    def __post_init__(self) -> None:
        if self.silence_min_duration <= 0:
            raise ValueError("silence_min_duration must be positive")
        if self.minimum_clip_duration <= 0:
            raise ValueError("minimum_clip_duration must be positive")
        if self.silence_padding < 0 or self.filler_padding < 0:
            raise ValueError("range padding cannot be negative")
        if self.zoom_factor is not None and not 1.0 <= self.zoom_factor <= 2.0:
            raise ValueError("zoom_factor must be between 1.0 and 2.0")
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")
        if self.transition not in {"none", "fade", "mix"}:
            raise ValueError("transition must be none, fade or mix")
        if not math.isfinite(self.transition_duration) or not 0.08 <= self.transition_duration <= 0.4:
            raise ValueError("transition_duration must be between 0.08 and 0.4 seconds")
        if not math.isfinite(self.speech_padding) or self.speech_padding < 0:
            raise ValueError("speech_padding must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class EditResult:
    """Metadata and cut plan for a completed render."""

    input_path: Path
    output_path: Path
    media: MediaInfo
    silence_ranges: tuple[TimeRange, ...]
    filler_ranges: tuple[TimeRange, ...]
    cuts: tuple[Cut, ...]


@dataclass(frozen=True, slots=True)
class TranscriptWord:
    """A word with source-media timing returned by a local STT model."""

    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """A caption-sized transcript segment with source-media timing."""

    text: str
    start: float
    end: float
    words: tuple[TranscriptWord, ...] = ()
    segment_id: str | None = None


@dataclass(frozen=True, slots=True)
class FlowAnalysis:
    """Safe, source-referenced semantic removals selected for a transcript."""

    segment_ids: tuple[str, ...] = ()
    ranges: tuple[TimeRange, ...] = ()


@dataclass(frozen=True, slots=True)
class WhisperRuntime:
    """Resolved faster-whisper device and compute settings."""

    device: str
    compute_type: str

    def __iter__(self):
        yield self.device
        yield self.compute_type


RangeLike = TimeRange | Sequence[float]


def _resolve_executable(executable: str) -> str:
    """Resolve a PATH executable while preserving explicit Windows paths."""

    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if Path(executable).exists():
        return executable
    raise FFmpegError(
        f"Could not find {executable!r}. Install FFmpeg and add its bin folder "
        "to PATH, or set the corresponding EditorConfig path."
    )


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _fraction(value: Any) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return _number(value)
    numerator, denominator = value.split("/", 1)
    top = _number(numerator)
    bottom = _number(denominator)
    if top is None or bottom in (None, 0):
        return None
    return top / bottom


def _run_capture(command: list[str], *, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            cwd=cwd,
        )
    except OSError as exc:
        raise FFmpegError(f"Could not start {' '.join(command[:1])}: {exc}") from exc


def probe_media(path: str | os.PathLike[str], ffprobe_path: str = "ffprobe") -> MediaInfo:
    """Probe a media file with FFprobe and return its primary stream metadata."""

    media_path = Path(path).expanduser()
    if not media_path.is_file():
        raise FileNotFoundError(f"Input media does not exist: {media_path}")
    executable = _resolve_executable(ffprobe_path)
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,duration",
        "-of",
        "json",
        str(media_path),
    ]
    completed = _run_capture(command)
    if completed.returncode:
        detail = completed.stderr.strip() or "FFprobe returned an error"
        raise MediaProbeError(detail)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("FFprobe did not return valid JSON") from exc

    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video is None:
        raise MediaProbeError(f"No video stream found in {media_path}")
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    format_duration = _number(payload.get("format", {}).get("duration"))
    stream_duration = _number(video.get("duration"))
    duration = format_duration or stream_duration
    if duration is None or duration <= 0:
        raise MediaProbeError(f"Could not determine a positive duration for {media_path}")

    frame_rate = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    return MediaInfo(
        path=media_path,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        frame_rate=frame_rate,
        video_codec=video.get("codec_name"),
        audio_codec=audio.get("codec_name") if audio else None,
        has_audio=audio is not None,
    )


def _coerce_range(value: RangeLike) -> TimeRange:
    if isinstance(value, TimeRange):
        return value
    if len(value) != 2:
        raise ValueError("Ranges must be TimeRange objects or (start, end) pairs")
    return TimeRange(float(value[0]), float(value[1]))


def _clamp_range(value: TimeRange, duration: float, padding: float = 0.0) -> TimeRange | None:
    start = max(0.0, value.start - padding)
    end = min(duration, value.end + padding)
    if end - start <= 0:
        return None
    return TimeRange(start, end)


def _merge_ranges(ranges: Iterable[TimeRange]) -> list[TimeRange]:
    ordered = sorted(ranges, key=lambda item: (item.start, item.end))
    merged: list[TimeRange] = []
    for current in ordered:
        if merged and current.start <= merged[-1].end:
            merged[-1] = TimeRange(merged[-1].start, max(merged[-1].end, current.end))
        else:
            merged.append(current)
    return merged


def build_cuts(
    duration: float,
    silence_ranges: Iterable[RangeLike] = (),
    filler_ranges: Iterable[RangeLike] = (),
    *,
    silence_padding: float = 0.0,
    filler_padding: float = 0.0,
    minimum_clip_duration: float = 0.05,
) -> list[Cut]:
    """Build synchronized A/V keep intervals from silence and filler ranges."""

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be positive")
    if minimum_clip_duration <= 0:
        raise ValueError("minimum_clip_duration must be positive")

    removable: list[TimeRange] = []
    for value in silence_ranges:
        item = _clamp_range(_coerce_range(value), duration, silence_padding)
        if item:
            removable.append(item)
    for value in filler_ranges:
        item = _clamp_range(_coerce_range(value), duration, filler_padding)
        if item:
            removable.append(item)

    cuts: list[Cut] = []
    cursor = 0.0
    timeline = 0.0
    for removed in _merge_ranges(removable):
        if removed.start - cursor >= minimum_clip_duration:
            cuts.append(Cut(cursor, removed.start, timeline))
            timeline += removed.start - cursor
        cursor = max(cursor, removed.end)
    if duration - cursor >= minimum_clip_duration:
        cuts.append(Cut(cursor, duration, timeline))
    if not cuts:
        raise ValueError("All media was removed by the supplied ranges")
    return cuts


def _parse_silence(stderr: str, duration: float) -> list[TimeRange]:
    ranges: list[TimeRange] = []
    open_start: float | None = None
    start_pattern = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
    end_pattern = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")
    events = sorted(
        [(match.start(), "start", float(match.group(1))) for match in start_pattern.finditer(stderr)]
        + [(match.start(), "end", float(match.group(1))) for match in end_pattern.finditer(stderr)],
        key=lambda item: item[0],
    )
    for _, kind, value in events:
        if kind == "start":
            open_start = max(0.0, value)
        elif open_start is not None:
            end = min(duration, max(open_start, value))
            if end > open_start:
                ranges.append(TimeRange(open_start, end))
            open_start = None
    if open_start is not None and duration > open_start:
        ranges.append(TimeRange(open_start, duration))
    return _merge_ranges(ranges)


def detect_silence(
    path: str | os.PathLike[str],
    *,
    ffmpeg_path: str = "ffmpeg",
    threshold_db: float = -35.0,
    minimum_duration: float = 0.45,
    media: MediaInfo | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[TimeRange]:
    """Detect silent audio spans using FFmpeg's ``silencedetect`` filter."""

    if minimum_duration <= 0:
        raise ValueError("minimum_duration must be positive")
    info = media or probe_media(path)
    if not info.has_audio:
        return []
    _notify(progress_callback, ProgressEvent("silence_detection", 0.0, "Analyzing audio silence"))
    executable = _resolve_executable(ffmpeg_path)
    command = [
        executable,
        "-hide_banner",
        "-nostats",
        "-i",
        str(info.path),
        "-map",
        "0:a:0",
        "-af",
        f"silencedetect=noise={threshold_db:g}dB:d={minimum_duration:g}",
        "-f",
        "null",
        "-",
    ]
    completed = _run_capture(command)
    if completed.returncode:
        detail = completed.stderr.strip() or "FFmpeg silence detection failed"
        raise FFmpegError(detail)
    ranges = _parse_silence(completed.stderr, info.duration)
    _notify(
        progress_callback,
        ProgressEvent("silence_detection", 1.0, f"Found {len(ranges)} silent range(s)"),
    )
    return ranges


def _notify(callback: ProgressCallback | None, event: ProgressEvent) -> None:
    if callback:
        callback(event)


_FILLER_WORDS = {
    "ah",
    "ahh",
    "erm",
    "er",
    "hmm",
    "mm",
    "mm-hm",
    "uh",
    "uhh",
    "um",
    "umm",
}


def _normalise_word(value: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", value.lower())


_WHISPER_MODEL_ALIASES = {
    "tiny": "tiny",
    "tiny-en": "tiny",
    "base": "base",
    "base-en": "base",
    "small": "small",
    "small-en": "small",
    "medium": "medium",
    "medium-en": "medium",
    "large": "large-v3",
    "large-v1": "large-v3",
    "large-v2": "large-v3",
    "large-v3": "large-v3",
    "turbo": "large-v3-turbo",
    "large-turbo": "large-v3-turbo",
    "large-v3-turbo": "large-v3-turbo",
}
_WHISPER_MODEL_CHOICES = (
    "tiny", "base", "small", "medium", "large-v3", "large-v3-turbo",
)
_WHISPER_DEVICE_ALIASES = {
    "auto": "auto",
    "cpu": "cpu",
    "host": "cpu",
    "cuda": "cuda",
    "gpu": "cuda",
}
_WHISPER_COMPUTE_ALIASES = {
    "auto": "auto",
    "default": "auto",
    "int8": "int8",
    "int8-float16": "int8_float16",
    "int8-fp16": "int8_float16",
    "float16": "float16",
    "fp16": "float16",
    "float-16": "float16",
}


def _normalise_whisper_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().casefold()).strip("-")


def normalize_whisper_model(model_size: str) -> str:
    """Normalize a faster-whisper model choice to a canonical name."""

    token = _normalise_whisper_token(model_size)
    try:
        return _WHISPER_MODEL_ALIASES[token]
    except KeyError as exc:
        choices = ", ".join(_WHISPER_MODEL_CHOICES)
        raise ValueError(f"Unknown Whisper model {model_size!r}. Choose one of: {choices}.") from exc


def normalize_whisper_device(device: str) -> str:
    """Normalize a device alias to ``auto``, ``cpu`` or ``cuda``."""

    token = _normalise_whisper_token(device)
    if token.startswith("cuda-") and token[5:].isdigit():
        token = "cuda"
    try:
        return _WHISPER_DEVICE_ALIASES[token]
    except KeyError as exc:
        raise ValueError("Whisper device must be one of: auto, cpu or cuda.") from exc


def normalize_whisper_compute_type(compute_type: str) -> str:
    """Normalize a compute-type alias to the supported public choices."""

    token = _normalise_whisper_token(compute_type)
    try:
        return _WHISPER_COMPUTE_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            "Whisper compute_type must be one of: auto, int8, int8_float16 or float16."
        ) from exc


def _cuda_device_count() -> int:
    """Return CTranslate2's CUDA count, treating missing/broken CUDA as zero."""

    try:
        import ctranslate2

        getter = getattr(ctranslate2, "get_cuda_device_count", None)
        if getter is None:
            return 0
        return max(0, int(getter()))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        return 0


def resolve_whisper_runtime(
    device: str = "auto", compute_type: str = "auto",
) -> WhisperRuntime:
    """Resolve local faster-whisper device/quantization settings."""

    requested_device = normalize_whisper_device(device)
    requested_compute = normalize_whisper_compute_type(compute_type)
    # The normal export path explicitly requests CPU. Avoid probing CUDA (and
    # importing CTranslate2) unless a caller intentionally asks for auto/CUDA.
    cuda_available = _cuda_device_count() > 0 if requested_device in {"auto", "cuda"} else False
    if requested_device == "cuda" and not cuda_available:
        raise ValueError(
            "Whisper device 'cuda' was requested, but CTranslate2 reports no CUDA device."
        )
    resolved_device = "cuda" if requested_device == "auto" and cuda_available else requested_device
    if resolved_device == "auto":
        resolved_device = "cpu"
    if requested_compute == "auto":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    else:
        resolved_compute = requested_compute
    if resolved_device == "cpu" and resolved_compute in {"float16", "int8_float16"}:
        raise ValueError(
            f"Whisper compute_type '{resolved_compute}' is not supported on CPU; use int8 or auto."
        )
    return WhisperRuntime(resolved_device, resolved_compute)


def download_whisper_model(
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    model_dir: str | os.PathLike[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Any:
    """Download/cache and initialize a local faster-whisper model.

    Construction performs the cache download and initialization. This function
    deliberately never receives audio and never calls ``transcribe``.
    """

    canonical_model = normalize_whisper_model(model_size)
    runtime = resolve_whisper_runtime(device, compute_type)
    try:
        from faster_whisper import WhisperModel
    except (ImportError, OSError) as exc:
        raise DependencyError(
            "Filler-word removal and captions require the local faster-whisper package. "
            "Install it with: python -m pip install faster-whisper"
        ) from exc

    cache_dir: Path | None = None
    if model_dir is not None and str(model_dir).strip():
        cache_dir = Path(model_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
    _notify(progress_callback, ProgressEvent("model_download", 0.0, "Preparing local Whisper model"))
    kwargs: dict[str, Any] = {"device": runtime.device, "compute_type": runtime.compute_type}
    if cache_dir is not None:
        kwargs["download_root"] = str(cache_dir)
    try:
        model = WhisperModel(canonical_model, **kwargs)
    except Exception as exc:
        # Keep URLs, local paths and authentication details out of UI logs.
        raise DependencyError(
            f"Could not initialize local Whisper model '{canonical_model}' "
            f"with {runtime.device}/{runtime.compute_type}."
        ) from exc
    if model is None:
        raise DependencyError(f"Local Whisper model '{canonical_model}' did not initialize.")
    _notify(
        progress_callback,
        ProgressEvent("model_download", 1.0, f"Local Whisper model ready: {canonical_model}"),
    )
    return model


def transcribe_media(
    path: str | os.PathLike[str],
    *,
    model_size: str = "small",
    ffmpeg_path: str = "ffmpeg",
    progress_callback: ProgressCallback | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    model_dir: str | os.PathLike[str] | None = None,
) -> tuple[tuple[TranscriptSegment, ...], tuple[TranscriptWord, ...]]:
    """Transcribe locally with faster-whisper when the optional package exists.

    The model is downloaded by faster-whisper on first use and cached by that
    package. No audio is uploaded by this function.
    """

    _notify(progress_callback, ProgressEvent("transcription", 0.0, "Loading local speech model"))
    selected_device = device if device is not None else os.environ.get("VIDEO_EDITOR_WHISPER_DEVICE", "cpu")
    selected_compute = (
        compute_type if compute_type is not None
        else os.environ.get("VIDEO_EDITOR_WHISPER_COMPUTE", "int8")
    )
    selected_model_dir = (
        model_dir if model_dir is not None
        else os.environ.get("VIDEO_EDITOR_WHISPER_MODEL_DIR")
    )
    model = download_whisper_model(
        model_size=model_size,
        device=selected_device,
        compute_type=selected_compute,
        model_dir=selected_model_dir,
        progress_callback=progress_callback,
    )
    segments, _ = model.transcribe(
        _speech_audio(path, ffmpeg_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    transcript: list[TranscriptSegment] = []
    words: list[TranscriptWord] = []
    for segment_index, segment in enumerate(segments):
        segment_words: list[TranscriptWord] = []
        for item in segment.words or ():
            start = float(item.start)
            end = float(item.end)
            if end > start:
                word = TranscriptWord(str(item.word).strip(), start, end)
                words.append(word)
                segment_words.append(word)
        start = float(segment.start)
        end = float(segment.end)
        text = str(segment.text).strip()
        if text and end > start:
            transcript.append(TranscriptSegment(
                text, start, end, tuple(segment_words), f"segment-{segment_index}"
            ))
    _notify(
        progress_callback,
        ProgressEvent("transcription", 1.0, f"Transcribed {len(words)} word(s) locally"),
    )
    return tuple(transcript), tuple(words)


def _speech_audio(path: str | os.PathLike[str], ffmpeg_path: str = "ffmpeg") -> Any:
    """Decode mono PCM using the same FFmpeg as the renderer, not PyAV APIs."""
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise DependencyError(
            "Local transcription requires NumPy for FFmpeg audio decoding. "
            "Install it with: python -m pip install numpy"
        ) from exc

    completed = subprocess.run(
        [_resolve_executable(ffmpeg_path), "-v", "error", "-nostdin", "-i", str(path),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
        capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise FFmpegError(completed.stderr.decode("utf-8", errors="replace"))
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()


def filler_word_ranges(words: Iterable[TranscriptWord]) -> list[TimeRange]:
    """Return conservative removal ranges for common spoken filler words."""

    ranges: list[TimeRange] = []
    for word in words:
        if _normalise_word(word.text) in _FILLER_WORDS and word.end > word.start:
            ranges.append(TimeRange(word.start, word.end))
    return ranges


_FLOW_LOW_VALUE_PHRASES = (
    "so anyway",
    "anyway moving on",
    "as i was saying",
    "you know what i mean",
    "where was i",
)


def _transcript_id(segment: TranscriptSegment, index: int) -> str:
    return str(segment.segment_id or f"segment-{index}")


def _flow_prompt(
    transcript: Sequence[TranscriptSegment], important_words: Sequence[str],
) -> str:
    records = [
        {
            "id": _transcript_id(segment, index),
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text,
        }
        for index, segment in enumerate(transcript)
    ]
    return (
        "Review this complete transcript for a conservative short-form edit. "
        "Return JSON only in the form {\"remove\":[{\"segment_id\":\"...\"}]}. "
        "You may instead return {\"remove\":[{\"start\":0.0,\"end\":1.0}]} "
        "when an exact source time range is necessary. Remove only clearly low-value "
        "or disruptive asides/repetitions that harm natural flow. Preserve facts, "
        "names, examples, context, and a natural conversational flow. Never delete "
        "a segment based only on filler words. Be conservative; an empty list is "
        "preferred when uncertain. Do not explain your answer or emit markdown.\n\n"
        f"Important lexical words (context only, not automatic cut targets): "
        f"{json.dumps(list(important_words), ensure_ascii=False)}\n"
        f"Transcript: {json.dumps(records, ensure_ascii=False)}"
    )


def _ai_chat_completion(
    prompt: str, *, api_key: str, base_url: str, model: str, timeout: float = 15.0,
) -> Any:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a conservative transcript editor."},
                {"role": "user", "content": prompt},
            ],
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _ai_content(payload: Any) -> Any:
    """Extract a chat content value without trusting provider-specific extras."""
    if isinstance(payload, (list, str)):
        return payload
    if not isinstance(payload, dict):
        return None
    if payload.get("refusal") or payload.get("error"):
        return None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        if isinstance(message, dict) and message.get("refusal"):
            return None
        return message.get("content") if isinstance(message, dict) else None
    return payload


def _parse_flow_candidates(payload: Any) -> list[Any] | None:
    content = _ai_content(payload)
    if content is None:
        return None
    if isinstance(content, str):
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            content = json.loads(text)
        except json.JSONDecodeError:
            return None
    if isinstance(content, list):
        return content
    if not isinstance(content, dict) or content.get("refusal"):
        return None
    candidates = content.get("remove", content.get("cuts", content.get("ranges", ())))
    return candidates if isinstance(candidates, list) else None


def _heuristic_flow_ranges(transcript: Sequence[TranscriptSegment]) -> list[tuple[str, TimeRange]]:
    """Explicit opt-in fallback for obvious discourse asides, not fillers."""
    result = []
    for index, segment in enumerate(transcript):
        normalized = " ".join(re.findall(r"[a-z0-9']+", segment.text.casefold()))
        if normalized in _FLOW_LOW_VALUE_PHRASES and len(normalized.split()) >= 2:
            result.append((_transcript_id(segment, index), TimeRange(segment.start, segment.end)))
    return result


def _safe_flow_analysis(
    candidates: Iterable[Any], transcript: Sequence[TranscriptSegment], duration: float,
) -> FlowAnalysis:
    by_id = {_transcript_id(segment, index): segment for index, segment in enumerate(transcript)}
    ranges: list[TimeRange] = []
    ids: list[str] = []
    for candidate in candidates:
        item_id: str | None = None
        item_range: TimeRange | None = None
        if isinstance(candidate, str):
            item_id = candidate
        elif isinstance(candidate, dict):
            value = candidate.get("segment_id", candidate.get("id"))
            if isinstance(value, (str, int)):
                item_id = str(value)
            start = _number(candidate.get("start"))
            end = _number(candidate.get("end"))
            if start is not None and end is not None and end > start:
                item_range = TimeRange(max(0.0, start), end)
            pair = candidate.get("time_range", candidate.get("range"))
            if item_range is None and isinstance(pair, (list, tuple)) and len(pair) == 2:
                start, end = _number(pair[0]), _number(pair[1])
                if start is not None and end is not None and end > start:
                    item_range = TimeRange(max(0.0, start), end)
        if item_id in by_id:
            segment = by_id[item_id]
            ranges.append(TimeRange(max(0.0, segment.start), min(duration, segment.end)))
            ids.append(item_id)
        elif item_range is not None:
            clamped_start = min(duration, max(0.0, item_range.start))
            clamped_end = min(duration, max(0.0, item_range.end))
            if clamped_end > clamped_start:
                ranges.append(TimeRange(clamped_start, clamped_end))
    merged = _merge_ranges(ranges)
    # Semantic selection must never turn the entire source into an empty export.
    if not merged or (merged[0].start <= 0 and merged[-1].end >= duration):
        return FlowAnalysis()
    return FlowAnalysis(tuple(dict.fromkeys(ids)), tuple(merged))


def analyze_transcript_flow(
    transcript: Sequence[TranscriptSegment], *, duration: float | None = None,
    important_words: Sequence[str] = (), options: dict[str, Any] | None = None,
) -> FlowAnalysis:
    """Analyze a complete transcript, failing closed on all AI/provider errors."""
    selected = options or {}
    if not bool(selected.get("analyze_flow", False)) or not transcript:
        return FlowAnalysis()
    if duration is None:
        duration = max((float(segment.end) for segment in transcript), default=0.0)
    if not math.isfinite(duration) or duration <= 0:
        return FlowAnalysis()
    api_key = str(selected.get("ai_api_key") or os.environ.get("ai_api_key")
                  or os.environ.get("AI_API_KEY")
                  or os.environ.get("VIDEO_EDITOR_AI_API_KEY") or "").strip()
    base_url = str(selected.get("ai_base_url") or os.environ.get("ai_base_url")
                   or os.environ.get("AI_BASE_URL") or os.environ.get("VIDEO_EDITOR_AI_BASE_URL")
                   or "https://api.openai.com/v1").strip()
    model = str(selected.get("ai_model") or os.environ.get("ai_model") or os.environ.get("AI_MODEL")
                or os.environ.get("VIDEO_EDITOR_AI_MODEL") or "gpt-4o-mini").strip()
    if api_key:
        try:
            payload = _ai_chat_completion(
                _flow_prompt(transcript, important_words), api_key=api_key,
                base_url=base_url, model=model,
            )
            candidates = _parse_flow_candidates(payload)
            if candidates is not None:
                return _safe_flow_analysis(candidates, transcript, duration)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
                urllib.error.URLError, urllib.error.HTTPError):
            pass
    if bool(selected.get("ai_fallback_heuristic", False)):
        return _safe_flow_analysis(
            [{"segment_id": item_id} for item_id, _ in _heuristic_flow_ranges(transcript)],
            transcript, duration,
        )
    return FlowAnalysis()


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds_value, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_value:02d},{millis:03d}"


def _retime_transcript(
    segments: Iterable[TranscriptSegment],
    cuts: Sequence[Cut],
) -> list[TranscriptSegment]:
    """Map source transcript segments onto the shortened output timeline."""

    result: list[TranscriptSegment] = []
    for segment in segments:
        for cut in cuts:
            start = max(segment.start, cut.source_start)
            end = min(segment.end, cut.source_end)
            if end <= start:
                continue
            offset = cut.timeline_start - cut.source_start
            retimed_words = tuple(
                TranscriptWord(word.text, max(word.start, start) + offset, min(word.end, end) + offset)
                for word in segment.words
                if cut.source_start <= (word.start + word.end) / 2 < cut.source_end
                and min(word.end, end) > max(word.start, start)
            )
            # Never fall back to the original full phrase after removing a
            # filler. Words are authoritative and carry the new timeline.
            if segment.words and not retimed_words:
                continue
            result.append(
                TranscriptSegment(
                    " ".join(word.text for word in retimed_words) if segment.words else segment.text,
                    start + offset,
                    end + offset,
                    retimed_words,
                )
            )
    return result


def write_srt(segments: Iterable[TranscriptSegment], path: str | os.PathLike[str]) -> Path:
    """Write caption segments to an SRT sidecar and return its path."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="\n") as handle:
        index = 1
        for segment in segments:
            if segment.end <= segment.start or not segment.text.strip():
                continue
            handle.write(f"{index}\n")
            handle.write(f"{_srt_timestamp(segment.start)} --> {_srt_timestamp(segment.end)}\n")
            handle.write(segment.text.strip() + "\n\n")
            index += 1
    return destination


def _run_ffmpeg(
    command: list[str],
    *,
    duration: float,
    progress_callback: ProgressCallback | None,
) -> None:
    """Run FFmpeg and stream its machine-readable progress on a callback."""

    try:
        process = subprocess.Popen(
            command[:1] + ["-progress", "pipe:1", "-nostats"] + command[1:],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        raise FFmpegError(f"Could not start FFmpeg: {exc}") from exc

    stderr_lines: list[str] = []

    def drain_stderr() -> None:
        assert process.stderr is not None
        stderr_lines.extend(process.stderr)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    last_fraction = 0.0
    assert process.stdout is not None
    for line in process.stdout:
        if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
            try:
                seconds = float(line.split("=", 1)[1].strip()) / 1_000_000
                fraction = min(0.99, max(last_fraction, seconds / duration))
                last_fraction = fraction
                _notify(progress_callback, ProgressEvent("render", fraction, "Rendering edited video"))
            except ValueError:
                continue
        elif line.strip() == "progress=end":
            _notify(progress_callback, ProgressEvent("render", 1.0, "Render complete"))
    return_code = process.wait()
    stderr_thread.join()
    process.stdout.close()
    if process.stderr:
        process.stderr.close()
    if return_code:
        detail = "".join(stderr_lines).strip() or "FFmpeg render failed"
        raise FFmpegError(detail)


def _transition_spans(cuts: Sequence[Cut], duration: float) -> list[float]:
    # Quantize overlap to whole frames. Short fragments get a shorter blend.
    return [max(1 / 15, math.floor(min(duration, a.duration / 3, b.duration / 3) * 15) / 15)
            for a, b in zip(cuts, cuts[1:])]


def _align_cuts(cuts: Sequence[Cut]) -> list[Cut]:
    """Make every segment an exact frame count at the export rate (30 fps)."""
    aligned = []
    cursor = 0.0
    for cut in cuts:
        start = math.ceil(cut.source_start * 30 - 1e-6) / 30
        end = math.floor(cut.source_end * 30 + 1e-6) / 30
        if end - start < 1 / 30 - 1e-6:
            continue
        aligned.append(Cut(start, end, cursor))
        cursor += end - start
    if not aligned:
        raise ValueError("No complete video frames remain after the selected cuts.")
    return aligned


def _build_filter_complex(
    cuts: Sequence[Cut], has_audio: bool, zoom_factor: float | None,
    transition: str = "none", transition_duration: float = 0.16,
) -> str:
    video_parts: list[str] = []
    audio_parts: list[str] = []
    spans = _transition_spans(cuts, transition_duration)
    zoom_filter = ""
    if zoom_factor is not None and zoom_factor > 1.0:
        zoom = f"{zoom_factor:.5f}"
        zoom_filter = (
            f",scale=ceil(iw*{zoom}/2)*2:ceil(ih*{zoom}/2)*2"
            f",crop=trunc(iw/{zoom}/2)*2:trunc(ih/{zoom}/2)*2:(iw-out_w)/2:(ih-out_h)/2"
        )
    for index, cut in enumerate(cuts):
        effects = ""
        if transition == "mix" and len(cuts) > 1:
            before = spans[index - 1] / 2 if index else 0.0
            after = spans[index] / 2 if index < len(cuts) - 1 else 0.0
            effects = (f",tpad=start_mode=clone:start_duration={before:.6f}"
                       f":stop_mode=clone:stop_duration={after:.6f}")
        elif transition == "fade":
            if index:
                effects += f",fade=t=in:st=0:d={spans[index - 1] / 2:.6f}"
            if index < len(cuts) - 1:
                edge = spans[index] / 2
                effects += f",fade=t=out:st={cut.duration-edge:.6f}:d={edge:.6f}"
        video_parts.append(
            f"[0:v]trim=start={cut.source_start:.6f}:end={cut.source_end:.6f}"
            f",setpts=PTS-STARTPTS,fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
            f"{zoom_filter}{effects},settb=AVTB[v{index}]"
        )
        if has_audio:
            audio_parts.append(
                f"[0:a]atrim=start={cut.source_start:.6f}:end={cut.source_end:.6f}"
                f",asetpts=PTS-STARTPTS,afade=t=in:d=0.004"
                f",afade=t=out:st={max(0, cut.duration-0.004):.6f}:d=0.004[a{index}]"
            )
    if transition == "mix" and len(cuts) > 1:
        chain = []
        previous = "v0"
        cursor = cuts[0].duration
        for index in range(1, len(cuts)):
            overlap = spans[index - 1]
            target = "vout" if index == len(cuts) - 1 else f"mix{index}"
            chain.append(
                f"[{previous}][v{index}]xfade=transition=fade:duration={overlap:.6f}"
                f":offset={cursor - overlap/2:.6f}[{target}]"
            )
            previous = target
            cursor += cuts[index].duration
        if has_audio:
            pads = "".join(f"[a{index}]" for index in range(len(cuts)))
            chain.append(f"{pads}concat=n={len(cuts)}:v=0:a=1[aout]")
        return ";".join(video_parts + audio_parts + chain)
    if has_audio:
        # concat consumes each segment as [video][audio], not all video pads
        # followed by all audio pads.
        segment_inputs = "".join(f"[v{index}][a{index}]" for index in range(len(cuts)))
        concat = f"{segment_inputs}concat=n={len(cuts)}:v=1:a=1[vout][aout]"
        return ";".join(video_parts + audio_parts + [concat])
    video_inputs = "".join(f"[v{index}]" for index in range(len(cuts)))
    concat = f"{video_inputs}concat=n={len(cuts)}:v=1:a=0[vout]"
    return ";".join(video_parts + [concat])


def render_cuts(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    media: MediaInfo,
    cuts: Sequence[Cut],
    *,
    config: EditorConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Render synchronized video/audio cuts with optional subtle zoom."""

    if not cuts:
        raise ValueError("At least one cut is required")
    settings = config or EditorConfig()
    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("Output must be different from input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    executable = _resolve_executable(settings.ffmpeg_path)
    filter_complex = _build_filter_complex(
        cuts, media.has_audio, settings.zoom_factor, settings.transition, settings.transition_duration,
    )
    command = [
        executable,
        "-hide_banner",
        "-nostdin",
        "-filter_complex_threads", "1",
        "-y",
        "-i",
        str(source),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        settings.video_codec,
        "-preset",
        settings.preset,
        "-crf",
        str(settings.crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if media.has_audio:
        command.extend(["-map", "[aout]", "-c:a", settings.audio_codec, "-b:a", settings.audio_bitrate])
    else:
        command.append("-an")
    command.extend(["-movflags", "+faststart", str(destination)])
    _notify(progress_callback, ProgressEvent("render", 0.0, "Rendering edited video"))
    # Keep long cut plans out of Windows' command-line length limit.
    with tempfile.TemporaryDirectory(prefix=".filters-", dir=destination.parent) as directory:
        graph = Path(directory) / "render.ffgraph"
        graph.write_text(filter_complex, encoding="utf-8")
        graph_index = command.index("-filter_complex")
        command[graph_index:graph_index+2] = ["-/filter_complex", str(graph)]
        _run_ffmpeg(
            command,
            duration=sum(cut.duration for cut in cuts),
            progress_callback=progress_callback,
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise FFmpegError(f"FFmpeg did not produce an output file: {destination}")
    return destination


def _run_simple_ffmpeg(command: list[str], *, cwd: Path | None = None) -> None:
    completed = _run_capture(command, cwd=cwd)
    if completed.returncode:
        detail = completed.stderr.strip() or "FFmpeg operation failed"
        raise FFmpegError(detail)


def burn_captions(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    srt_path: str | os.PathLike[str],
    *,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Burn ASS styling/motion (or legacy SRT) without changing edited audio."""

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    subtitle = Path(srt_path).expanduser().resolve()
    # A safe relative filter filename avoids FFmpeg's two-level path escaping
    # entirely (Windows drive colons, apostrophes, commas and brackets).
    alias = "captions.ass" if subtitle.suffix.lower() == ".ass" else "captions.srt"
    caption_filter = f"subtitles=filename={alias}"
    if subtitle.suffix.lower() != ".ass":
        caption_filter += ":force_style='FontName=Arial,FontSize=22,Outline=2,Shadow=1,MarginV=48'"
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _resolve_executable(ffmpeg_path),
        "-hide_banner",
        "-y",
        "-i",
        str(source),
        "-vf",
        caption_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    with tempfile.TemporaryDirectory(prefix=".caption-filter-", dir=destination.parent) as directory:
        folder = Path(directory)
        shutil.copyfile(subtitle, folder / alias)
        _run_simple_ffmpeg(command, cwd=folder)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise FFmpegError(f"FFmpeg did not produce captioned output: {destination}")
    return destination


TrackingFace = tuple[float, float, float, float]


def _normalise_tracking_mode(value: Any) -> str:
    """Return a stable tracking mode while accepting UI-friendly aliases."""

    mode = re.sub(r"[^a-z0-9]+", "_", str(value or "single_subject").strip().casefold()).strip("_")
    aliases = {
        # Explicit, explainable framing modes.  The older names below remain
        # accepted because the desktop UI has shipped them in saved projects.
        "fixed": "fixed_manual_safe",
        "manual": "fixed_manual_safe",
        "manual_safe": "fixed_manual_safe",
        "fixed_safe": "fixed_manual_safe",
        "fixed_manual": "fixed_manual_safe",
        "fixed_manual_safe": "fixed_manual_safe",
        "adaptive": "adaptive_crop",
        "adaptive_crop": "adaptive_crop",
        "content_crop": "adaptive_crop",
        "single": "single_subject",
        "single_subject": "single_subject",
        "single_face": "single_subject",
        "smart": "smart",
        "smart_group": "smart",
        "group_framing": "smart",
        "smart_multi_face": "smart",
        "multi_face": "smart",
        "group": "smart",
        "split": "two_person_split",
        "two_person": "two_person_split",
        "two_person_split": "two_person_split",
        "two_person_split_vertical": "two_person_split",
        "reaction_split": "reaction_stream_split",
        "reaction_stream": "reaction_stream_split",
        "reaction_stream_split": "reaction_stream_split",
        "stream_split": "reaction_stream_split",
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError(
            "tracking_mode must be fixed_manual_safe, single_subject, smart, "
            "adaptive_crop, two_person_split or reaction_stream_split"
        ) from exc


def _normalise_split_layout(value: Any) -> str:
    """Normalize split orientation without hiding the geometry decision."""

    token = re.sub(r"[^a-z0-9]+", "_", str(value or "auto").strip().casefold()).strip("_")
    aliases = {
        "auto": "auto",
        "automatic": "auto",
        "horizontal": "top_bottom",
        "top_bottom": "top_bottom",
        "topbottom": "top_bottom",
        "vertical": "side_by_side",
        "side_by_side": "side_by_side",
        "sidebyside": "side_by_side",
        "left_right": "side_by_side",
    }
    try:
        return aliases[token]
    except KeyError as exc:
        raise ValueError("split_layout must be auto, top_bottom or side_by_side") from exc


def _choose_split_layout(
    faces: Sequence[TrackingFace], width: int, height: int,
) -> str:
    """Choose a pane arrangement from subject centers, using a simple rule.

    A horizontal pair gets side-by-side panes; a vertically arranged pair gets
    top/bottom panes.  The fallback is deliberately conservative and stable.
    """

    selected = _select_tracking_faces(faces, max_faces=2)
    if len(selected) < 2:
        return "top_bottom"
    centers = sorted(
        ((face[0] + face[2] / 2.0, face[1] + face[3] / 2.0) for face in selected),
        key=lambda item: (item[0], item[1]),
    )
    dx = abs(centers[1][0] - centers[0][0]) / max(1.0, float(width))
    dy = abs(centers[1][1] - centers[0][1]) / max(1.0, float(height))
    return "side_by_side" if dx >= dy else "top_bottom"


def _face_area(face: Sequence[float]) -> float:
    return max(0.0, float(face[2])) * max(0.0, float(face[3]))


def _select_tracking_faces(
    faces: Iterable[Sequence[float]], *, max_faces: int = 2,
) -> tuple[TrackingFace, ...]:
    """Choose the largest detections deterministically, retaining at most two."""

    if max_faces <= 0:
        return ()
    cleaned = [
        (float(face[0]), float(face[1]), float(face[2]), float(face[3]))
        for face in faces
        if len(face) >= 4 and float(face[2]) > 0 and float(face[3]) > 0
    ]
    cleaned.sort(key=lambda item: (-_face_area(item), item[1], item[0]))
    return tuple(cleaned[:max_faces])


def _face_union(faces: Sequence[TrackingFace]) -> TrackingFace:
    """Return the bounding union of one or more face rectangles."""

    if not faces:
        raise ValueError("At least one face is required")
    left = min(face[0] for face in faces)
    top = min(face[1] for face in faces)
    right = max(face[0] + face[2] for face in faces)
    bottom = max(face[1] + face[3] for face in faces)
    return left, top, right - left, bottom - top


def _tracking_target(
    faces: Sequence[TrackingFace], fallback: tuple[float, float], mode: str,
) -> tuple[float, float]:
    """Compute a face center for a mode; ``fallback`` is the last known target."""

    if not faces:
        return fallback
    selected = _select_tracking_faces(faces, max_faces=1 if mode == "single_subject" else 2)
    if mode == "single_subject":
        face = selected[0]
        return face[0] + face[2] / 2.0, face[1] + face[3] / 2.0
    union = _face_union(selected)
    return union[0] + union[2] / 2.0, union[1] + union[3] / 2.0


def _smooth_tracking_point(
    previous: tuple[float, float], target: tuple[float, float], alpha: float = 0.12,
) -> tuple[float, float]:
    """Apply the deliberately conservative motion smoothing used by tracking."""

    if not 0 < alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    return (
        previous[0] * (1.0 - alpha) + target[0] * alpha,
        previous[1] * (1.0 - alpha) + target[1] * alpha,
    )


def _estimate_active_content_bounds(
    frame: Any, *, threshold: int = 16, min_occupancy: float = 0.025,
) -> tuple[int, int, int, int] | None:
    """Estimate non-black content bounds using inexpensive border statistics.

    This intentionally does not claim semantic scene understanding.  It removes
    common letterbox/pillarbox bars and large empty margins while returning
    ``None`` for ambiguous frames so normal center framing remains the fallback.
    """

    try:
        import cv2
        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if gray.size == 0:
            return None
        mask = gray > max(0, int(threshold))
        height, width = gray.shape[:2]
        row_counts = mask.sum(axis=1)
        col_counts = mask.sum(axis=0)
        row_limit = max(3, int(width * min_occupancy))
        col_limit = max(3, int(height * min_occupancy))
        rows = np.flatnonzero(row_counts >= row_limit)
        cols = np.flatnonzero(col_counts >= col_limit)
        if len(rows) == 0 or len(cols) == 0:
            return None
        left, right = int(cols[0]), int(cols[-1]) + 1
        top, bottom = int(rows[0]), int(rows[-1]) + 1
        # Do not chase a few dark pixels or turn a normal frame into a crop.
        if (right - left) * (bottom - top) < width * height * 0.35:
            return None
        return left, top, right - left, bottom - top
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _crop_box(
    width: int, height: int, ratio: float, zoom_factor: float,
    center: tuple[float, float], *, required_faces: Sequence[TrackingFace] = (),
    content_bounds: Sequence[float] | None = None,
) -> tuple[int, int, int, int]:
    """Return an even crop, optionally constrained to active image content.

    Content bounds are only a hint: the source frame remains the final bound,
    and required face rectangles expand the hint so detected subjects survive.
    """

    if zoom_factor <= 0 or not math.isfinite(zoom_factor):
        raise ValueError("zoom_factor must be positive and finite")
    bound_left, bound_top, bound_width, bound_height = 0.0, 0.0, float(width), float(height)
    if content_bounds is not None and len(content_bounds) >= 4:
        candidate = tuple(float(value) for value in content_bounds[:4])
        if candidate[2] > 1 and candidate[3] > 1:
            bound_left = max(0.0, min(float(width - 1), candidate[0]))
            bound_top = max(0.0, min(float(height - 1), candidate[1]))
            right = max(bound_left + 2.0, min(float(width), candidate[0] + candidate[2]))
            bottom = max(bound_top + 2.0, min(float(height), candidate[1] + candidate[3]))
            bound_width, bound_height = right - bound_left, bottom - bound_top
    union: TrackingFace | None = None
    if required_faces:
        union = _face_union(required_faces)
        left = min(bound_left, union[0])
        top = min(bound_top, union[1])
        right = max(bound_left + bound_width, union[0] + union[2])
        bottom = max(bound_top + bound_height, union[1] + union[3])
        bound_left, bound_top = max(0.0, left), max(0.0, top)
        bound_width = min(float(width) - bound_left, right - bound_left)
        bound_height = min(float(height) - bound_top, bottom - bound_top)
    fit_width = min(bound_width, bound_height * ratio)
    crop_width = fit_width / zoom_factor
    if union is not None:
        # A little headroom keeps both faces from touching a pane edge, while
        # the source bounds remain the final authority for extreme layouts.
        crop_width = max(crop_width, union[2] * 1.25, union[3] * ratio * 1.25)
    crop_width = min(fit_width, max(2.0, crop_width))
    crop_height = min(bound_height, crop_width / ratio)
    crop_width = min(bound_width, crop_height * ratio)
    crop_width = max(2, int(crop_width) // 2 * 2)
    crop_height = max(2, int(crop_height) // 2 * 2)
    left = max(int(bound_left), min(int(bound_left + bound_width - crop_width), int(center[0] - crop_width / 2)))
    top = max(int(bound_top), min(int(bound_top + bound_height - crop_height), int(center[1] - crop_height / 2)))
    if union is not None:
        if union[0] < left:
            left = max(int(bound_left), int(union[0] - crop_width * 0.08))
        if union[0] + union[2] > left + crop_width:
            left = min(int(bound_left + bound_width - crop_width), int(union[0] + union[2] - crop_width + crop_width * 0.08))
        if union[1] < top:
            top = max(int(bound_top), int(union[1] - crop_height * 0.08))
        if union[1] + union[3] > top + crop_height:
            top = min(int(bound_top + bound_height - crop_height), int(union[1] + union[3] - crop_height + crop_height * 0.08))
    return left, top, crop_width, crop_height


def _split_targets(
    faces: Sequence[TrackingFace], fallback: Sequence[tuple[float, float]],
    layout: str = "side_by_side",
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return ordered pane targets, duplicating a lone subject when needed."""

    selected = _select_tracking_faces(faces, max_faces=2)
    if layout == "top_bottom":
        selected = tuple(sorted(selected, key=lambda face: (face[1] + face[3] / 2.0, face[0])))
    else:
        selected = tuple(sorted(selected, key=lambda face: (face[0] + face[2] / 2.0, face[1])))
    targets = [(face[0] + face[2] / 2.0, face[1] + face[3] / 2.0) for face in selected]
    if len(targets) == 2:
        return targets[0], targets[1]
    if len(targets) == 1:
        return targets[0], targets[0]
    return fallback[0], fallback[1]


def apply_face_tracking(
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    ffmpeg_path: str = "ffmpeg",
    zoom_factor: float = 1.08,
    output_format: str = "original",
    tracking_mode: str = "single_subject",
    framing_mode: str | None = None,
    split_layout: str = "auto",
    manual_center: Sequence[float] | None = None,
    detect_every: int = 4,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Track faces into a reframed video while copying the original audio.

    ``fixed_manual_safe`` is a deterministic center/manual crop and is useful
    when a face detector is not appropriate. ``smart`` tracks a stable union
    of at most two faces, while ``adaptive_crop`` also trims obvious bars or
    empty margins. ``reaction_stream_split`` chooses top/bottom or side by
    side panes from the pair arrangement; all split output is 1080x1920.
    """

    mode = _normalise_tracking_mode(framing_mode if framing_mode is not None else tracking_mode)
    if mode in {"two_person_split", "reaction_stream_split"} and output_format != "vertical":
        raise ValueError(
            "tracking_mode split requires output_format='vertical' "
            "(1080x1920)"
        )
    if output_format not in {"original", "vertical", "square"}:
        raise ValueError("output_format must be original, vertical or square")
    layout_request = _normalise_split_layout(split_layout)
    try:
        interval = max(1, int(detect_every))
    except (TypeError, ValueError) as exc:
        raise ValueError("detect_every must be a positive integer") from exc

    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise DependencyError(
            "Face tracking requires the local opencv-python package. "
            "Install it with: python -m pip install opencv-python"
        ) from exc
    if not hasattr(cv2, "CascadeClassifier"):
        raise DependencyError(
            "This OpenCV build does not include the classic face detector. "
            "Install a complete opencv-python wheel."
        )

    source = Path(input_path).expanduser()
    destination = Path(output_path).expanduser()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FFmpegError(f"OpenCV could not open {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise FFmpegError("Could not read video dimensions for face tracking")
    out_width, out_height = output_dimensions(width, height, output_format)

    detector = None
    profile_detector = None
    upperbody_detector = None
    if mode not in {"fixed_manual_safe"}:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        candidate_detector = cv2.CascadeClassifier(cascade_path)
        if not candidate_detector.empty():
            detector = candidate_detector
        # Profile and upper-body cascades are inexpensive fallbacks for
        # reactors/podcasters who turn sideways or sit farther from camera.
        profile_path = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
        profile_candidate = cv2.CascadeClassifier(profile_path)
        if not profile_candidate.empty():
            profile_detector = profile_candidate
        upperbody_path = os.path.join(cv2.data.haarcascades, "haarcascade_upperbody.xml")
        upperbody_candidate = cv2.CascadeClassifier(upperbody_path)
        if not upperbody_candidate.empty():
            upperbody_detector = upperbody_candidate

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_video = destination.with_suffix(".tracking.mp4")
    pane_height = out_height // 2
    target = (width / 2.0, height / 2.0)
    if manual_center is not None:
        if len(manual_center) != 2:
            raise ValueError("manual_center must contain x and y")
        raw_x, raw_y = float(manual_center[0]), float(manual_center[1])
        # Values in [0, 1] are normalized; pixel coordinates remain accepted.
        target = (
            raw_x * width if 0 <= raw_x <= 1 else raw_x,
            raw_y * height if 0 <= raw_y <= 1 else raw_y,
        )
        target = (max(0.0, min(width, target[0])), max(0.0, min(height, target[1])))
    split_target = ((width * 0.33, height / 2.0), (width * 0.67, height / 2.0))
    selected_layout = "top_bottom"
    last_faces: tuple[TrackingFace, ...] = ()
    active_bounds: tuple[int, int, int, int] | None = None
    writer = cv2.VideoWriter(
        str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height),
    )
    if not writer.isOpened():
        cap.release()
        raise FFmpegError("OpenCV could not create the tracking intermediate")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detected = last_faces
            if mode != "fixed_manual_safe" and frame_index % interval == 0:
                if detector is not None:
                    # Resize only detection frames and cap the long edge.  The
                    # original frame is decoded once for writing and is never
                    # needlessly rescaled between detector calls.
                    detection_scale = min(1.0, 512 / max(width, height))
                    small = (cv2.resize(frame, None, fx=detection_scale, fy=detection_scale)
                             if detection_scale < 1.0 else frame)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    try:
                        found = detector.detectMultiScale(
                            gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24),
                        )
                        detected_list = [
                            (float(face[0]) / detection_scale, float(face[1]) / detection_scale,
                             float(face[2]) / detection_scale, float(face[3]) / detection_scale)
                            for face in found
                        ]
                        # A frontal cascade alone often loses a second
                        # participant or a side-facing reactor. Add profile
                        # detections only when needed, then use upper-body
                        # detection as a last resort for distant subjects.
                        if len(detected_list) < 2 and profile_detector is not None:
                            for face in profile_detector.detectMultiScale(
                                gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24),
                            ):
                                detected_list.append((
                                    float(face[0]) / detection_scale, float(face[1]) / detection_scale,
                                    float(face[2]) / detection_scale, float(face[3]) / detection_scale,
                                ))
                        if not detected_list and upperbody_detector is not None:
                            for body in upperbody_detector.detectMultiScale(
                                gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40),
                            ):
                                detected_list.append((
                                    float(body[0]) / detection_scale, float(body[1]) / detection_scale,
                                    float(body[2]) / detection_scale, float(body[3]) / detection_scale,
                                ))
                        detected = tuple(_select_tracking_faces(detected_list, max_faces=2))
                    except (cv2.error, TypeError, ValueError):
                        # A corrupt/unsupported frame should not discard the
                        # whole export; retain the last stable observation.
                        detected = last_faces
                    # Hold the last good observation through short detector
                    # misses; this is more stable than snapping to center.
                    if detected:
                        last_faces = _select_tracking_faces(detected, max_faces=2)
                    else:
                        detected = last_faces
                if mode == "adaptive_crop":
                    active_bounds = _estimate_active_content_bounds(frame)
            if mode in {"two_person_split", "reaction_stream_split"}:
                if layout_request == "auto":
                    if len(detected) >= 2:
                        selected_layout = _choose_split_layout(detected, width, height)
                else:
                    selected_layout = layout_request
                left_target, right_target = _split_targets(detected, split_target, selected_layout)
                split_target = (
                    _smooth_tracking_point(split_target[0], left_target),
                    _smooth_tracking_point(split_target[1], right_target),
                )
                side_by_side = selected_layout == "side_by_side"
                pane_width = out_width // 2 if side_by_side else out_width
                pane_height_actual = out_height if side_by_side else pane_height
                pane_ratio = pane_width / pane_height_actual
                panes = []
                for pane_target in split_target:
                    left, top, crop_width, crop_height = _crop_box(
                        width, height, pane_ratio, zoom_factor, pane_target,
                    )
                    crop = frame[top : top + crop_height, left : left + crop_width]
                    panes.append(cv2.resize(crop, (pane_width, pane_height_actual), interpolation=cv2.INTER_LINEAR))
                rendered = cv2.hconcat(panes) if side_by_side else cv2.vconcat(panes)
            else:
                if mode != "fixed_manual_safe":
                    target = _smooth_tracking_point(target, _tracking_target(detected, target, mode))
                ratio = out_width / out_height
                required = _select_tracking_faces(detected, max_faces=2) if mode in {"smart", "adaptive_crop"} else ()
                left, top, crop_width, crop_height = _crop_box(
                    width, height, ratio, zoom_factor, target, required_faces=required,
                    content_bounds=active_bounds if mode == "adaptive_crop" else None,
                )
                crop = frame[top : top + crop_height, left : left + crop_width]
                rendered = cv2.resize(crop, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
            writer.write(rendered)
            frame_index += 1
            if progress_callback and frame_count:
                _notify(
                    progress_callback,
                    ProgressEvent("face_tracking", frame_index / frame_count, "Tracking faces"),
                )
    finally:
        cap.release()
        writer.release()

    command = [
        _resolve_executable(ffmpeg_path),
        "-hide_banner",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        _run_simple_ffmpeg(command)
    finally:
        temp_video.unlink(missing_ok=True)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise FFmpegError(f"Face tracking did not produce output: {destination}")
    _notify(progress_callback, ProgressEvent("face_tracking", 1.0, "Face tracking complete"))
    return destination


def output_dimensions(width: int, height: int, output_format: str = "original") -> tuple[int, int]:
    if output_format == "vertical":
        return 1080, 1920
    if output_format == "square":
        return 1080, 1080
    if output_format != "original":
        raise ValueError("output_format must be original, vertical or square")
    return width // 2 * 2, height // 2 * 2


def reframe_video(source: Path, destination: Path, media: MediaInfo, output_format: str,
                  ffmpeg_path: str) -> Path:
    """Center crop without distortion when face following is disabled."""
    width, height = output_dimensions(media.width, media.height, output_format)
    _run_simple_ffmpeg([
        _resolve_executable(ffmpeg_path), "-v", "error", "-nostdin", "-y", "-i", str(source),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-c:a", "copy", "-movflags", "+faststart", str(destination),
    ])
    return destination


def _validated_watermark_options(options: dict[str, Any]) -> tuple[Path | None, float, float, str]:
    raw_path = options.get("watermark_path")
    if raw_path in (None, ""):
        return None, 0.25, 0.2, "bottom_right"
    path = Path(str(raw_path)).expanduser()
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        raise ValueError("watermark_path must be an existing PNG, JPG, JPEG, WebP, BMP, GIF, or TIFF file")
    try:
        opacity = float(options.get("watermark_opacity", 1.0))
        scale = float(options.get("watermark_scale", 0.2))
    except (TypeError, ValueError) as exc:
        raise ValueError("watermark_opacity and watermark_scale must be numbers") from exc
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("watermark_opacity must be between 0 and 1")
    if not math.isfinite(scale) or not 0.05 <= scale <= 0.5:
        raise ValueError("watermark_scale must be between 0.05 and 0.5")
    position = str(options.get("watermark_position", "bottom_right"))
    if position not in {"top_left", "top_right", "bottom_left", "bottom_right", "center"}:
        raise ValueError("watermark_position is invalid")
    return path.resolve(), opacity, scale, position


def _validated_music_options(options: dict[str, Any]) -> tuple[Path | None, float, bool]:
    raw_path = options.get("music_path")
    if raw_path in (None, ""):
        return None, 0.25, False
    path = Path(str(raw_path)).expanduser()
    if not path.is_file():
        raise ValueError("music_path must be an existing audio file")
    try:
        volume = float(options.get("music_volume", 0.25))
    except (TypeError, ValueError) as exc:
        raise ValueError("music_volume must be a number") from exc
    if not math.isfinite(volume) or not 0 <= volume <= 1:
        raise ValueError("music_volume must be between 0 and 1")
    return path.resolve(), volume, bool(options.get("music_loop", False))


def apply_media_overlays(
    input_path: str | os.PathLike[str], output_path: str | os.PathLike[str],
    *, ffmpeg_path: str = "ffmpeg", watermark_path: str | os.PathLike[str] | None = None,
    watermark_opacity: float = 1.0, watermark_scale: float = 0.2,
    watermark_position: str = "bottom_right", music_path: str | os.PathLike[str] | None = None,
    music_volume: float = 0.25, music_loop: bool = False, has_audio: bool = True,
) -> Path:
    """Apply optional watermark and music in one staged FFmpeg operation."""
    options = {
        "watermark_path": watermark_path,
        "watermark_opacity": watermark_opacity,
        "watermark_scale": watermark_scale,
        "watermark_position": watermark_position,
        "music_path": music_path,
        "music_volume": music_volume,
        "music_loop": music_loop,
    }
    watermark, opacity, scale, position = _validated_watermark_options(options)
    music, volume, loop = _validated_music_options(options)
    if watermark is None and music is None:
        raise ValueError("At least one media overlay option is required")
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("Output must be different from input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [_resolve_executable(ffmpeg_path), "-hide_banner", "-nostdin", "-y", "-i", str(source)]
    if watermark:
        command.extend(["-loop", "1", "-i", str(watermark)])
    if music:
        if loop:
            command.extend(["-stream_loop", "-1"])
        command.extend(["-i", str(music)])
    filters: list[str] = []
    video_label = "0:v"
    if watermark:
        wm_index = 1
        filters.append(
            f"[{wm_index}:v]format=rgba,colorchannelmixer=aa={opacity:.6f}[wm0];"
            f"[wm0][0:v]scale2ref=w=main_w*{scale:.6f}:h=-1[wm][base];"
            f"[base][wm]overlay="
            f"{ {'top_left': '10:10', 'top_right': 'main_w-overlay_w-10:10', 'bottom_left': '10:main_h-overlay_h-10', 'bottom_right': 'main_w-overlay_w-10:main_h-overlay_h-10', 'center': '(main_w-overlay_w)/2:(main_h-overlay_h)/2'}[position] }"
            f":format=auto[vout]"
        )
        video_label = "vout"
    audio_label: str | None = None
    if music:
        music_index = 2 if watermark else 1
        if has_audio:
            filters.append(
                f"[0:a]volume=1[a_original];[{music_index}:a]volume={volume:.6f}[a_music];"
                "[a_original][a_music]amix=inputs=2:duration=first:dropout_transition=0[aout]"
            )
        else:
            filters.append(f"[{music_index}:a]volume={volume:.6f}[aout]")
        audio_label = "aout"
    if filters:
        command.extend(["-filter_complex", ";".join(filters)])
    command.extend(["-map", f"[{video_label}]" if video_label != "0:v" else "0:v:0"])
    if audio_label:
        command.extend(["-map", f"[{audio_label}]", "-c:a", "aac", "-b:a", "192k"])
    elif has_audio:
        command.extend(["-map", "0:a:0", "-c:a", "copy"])
    else:
        command.append("-an")
    command.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                    "-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart",
                    str(destination)])
    _run_simple_ffmpeg(command)
    if not destination.is_file() or destination.stat().st_size == 0:
        raise FFmpegError(f"FFmpeg did not produce media-overlaid output: {destination}")
    return destination


def process_video(
    input_path: str,
    output_path: str,
    options: dict[str, Any] | None = None,
    progress_callback: Callable[[float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Run the dependable four-feature local media pipeline.

    The options dictionary remains intentionally permissive because the desktop
    UI has older saved projects.  Only silence removal, filler-word removal,
    captions, and single-subject face tracking affect this flow.  Legacy AI,
    overlays, transitions, split/adaptive framing, and centered reframing are
    ignored rather than allowed to change an export unexpectedly.
    """

    selected = options or {}
    remove_silence = bool(selected.get("remove_silence", True))
    remove_fillers = bool(selected.get("remove_filler_words", False))
    face_tracking = bool(selected.get("face_tracking", False))
    captions = bool(selected.get("captions", False))
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination or (source.is_file() and destination.exists() and os.path.samefile(source, destination)):
        raise ValueError("Output must be different from input. Your original is never overwritten.")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Choose an .mp4 output file.")

    bundled_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    bundled_ffmpeg = bundled_dir / "ffmpeg.exe"
    ffmpeg_path = os.environ.get("FFMPEG_PATH") or (str(bundled_ffmpeg) if bundled_ffmpeg.is_file() else "ffmpeg")
    ffmpeg_file = Path(ffmpeg_path)
    ffprobe_path = os.environ.get("FFPROBE_PATH") or (
        str(ffmpeg_file.with_name("ffprobe.exe")) if ffmpeg_file.is_file() else "ffprobe"
    )

    last_progress = 0.0
    last_message = ""

    def report(fraction: float) -> None:
        nonlocal last_progress
        last_progress = max(last_progress, min(1.0, fraction))
        if progress_callback:
            progress_callback(last_progress)

    def log(message: str) -> None:
        if log_callback:
            log_callback(message)

    def notify(event: ProgressEvent) -> None:
        nonlocal last_message
        bounds = {
            "transcription": (0.02, 0.20),
            "model_download": (0.02, 0.20),
            "probe": (0.20, 0.24),
            "silence_detection": (0.24, 0.34),
            "cut_planning": (0.34, 0.40),
            "render": (0.40, 0.72),
            "face_tracking": (0.72, 0.90),
        }
        start, end = bounds.get(event.phase, (0.20, 0.24))
        report(start + (end - start) * event.fraction)
        if event.message and event.message != last_message:
            log(event.message)
            last_message = event.message

    media = probe_media(source, ffprobe_path)
    if (remove_fillers or captions) and not media.has_audio:
        raise FFmpegError("Filler-word removal and captions require an audio stream.")

    transcript: tuple[TranscriptSegment, ...] = ()
    words: tuple[TranscriptWord, ...] = ()
    if remove_fillers or captions:
        # Keep the active path deterministic and CPU-only.  Explicit model/cache
        # settings remain accepted for compatibility, but GPU/runtime switches
        # from older projects are deliberately not consulted.
        model_size = selected.get("model_size", selected.get("whisper_model_size", selected.get("whisper_model", "tiny")))
        model_dir = selected.get("model_dir", selected.get("whisper_model_dir"))
        log(f"Loading local CPU Whisper model: {model_size}")
        transcript, words = transcribe_media(
            source,
            model_size=model_size,
            ffmpeg_path=ffmpeg_path,
            progress_callback=notify,
            device="cpu",
            compute_type="int8",
            model_dir=model_dir,
        )

    filler_ranges = filler_word_ranges(words) if remove_fillers else []
    if remove_fillers:
        log(f"Detected {len(filler_ranges)} filler-word range(s)")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".video-editor-", dir=destination.parent) as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_base = temp_dir_path / "cuts.mp4"
        temp_face = temp_dir_path / "face-tracked.mp4"
        temp_caption = temp_dir_path / "captioned.mp4"
        temp_srt = temp_dir_path / "captions.srt"
        temp_ass = temp_dir_path / "captions.ass"
        config = EditorConfig(
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            remove_silence=remove_silence,
            zoom_factor=None,
            preset="veryfast",
            crf=22,
            transition="none",
        )
        result = VideoEditor(config, progress_callback=notify).edit(
            source, temp_base, filler_ranges=filler_ranges, extra_removals=()
        )
        current = temp_base
        if face_tracking:
            # The old UI can still send split/adaptive mode names; the core
            # pipeline intentionally provides one stable single-subject mode.
            current = apply_face_tracking(
                current,
                temp_face,
                ffmpeg_path=ffmpeg_path,
                output_format="original",
                tracking_mode="single_subject",
                detect_every=selected.get("tracking_detect_every", selected.get("detect_every", 4)),
                progress_callback=notify,
            )

        if captions:
            from captions import write_ass

            retimed = _retime_transcript(transcript, result.cuts)
            if not retimed:
                raise FFmpegError("No captionable speech remained after the selected edits.")
            write_srt(retimed, temp_srt)
            write_ass(
                retimed,
                temp_ass,
                width=result.media.width,
                height=result.media.height,
                preset=str(selected.get("caption_preset", "clean")),
                mode=str(selected.get("caption_mode", "line")),
            )
            log(f"Writing {len(retimed)} caption segment(s)")
            current = burn_captions(current, temp_caption, temp_ass, ffmpeg_path=ffmpeg_path)

        os.replace(current, destination)
        if captions:
            os.replace(temp_srt, destination.with_suffix(".srt"))
            os.replace(temp_ass, destination.with_suffix(".ass"))

    report(1.0)
    if log_callback:
        log_callback(f"Saved edited video: {destination}")
    return {
        "output_path": str(destination),
        "removed_silence_ranges": len(result.silence_ranges),
        "removed_filler_ranges": len(result.filler_ranges),
        "removed_analyze_flow_ranges": 0,
        "analyze_flow_segment_ids": (),
        "analyze_flow_ranges": (),
        "captions": captions,
        "face_tracking": face_tracking,
        "tracking_mode": "single_subject" if face_tracking else None,
        "split_layout": None,
        "removed_no_dialogue_ranges": 0,
        "output_format": "original",
        "transition": "none",
        "important_words": (),
        "watermark": False,
        "music": False,
    }


def run_caption_hook(srt_path: str | os.PathLike[str], hook: CaptionHook) -> Any:
    """Pass an existing SRT file to an application-owned caption hook."""

    subtitle_path = Path(srt_path).expanduser()
    if not subtitle_path.is_file():
        raise FileNotFoundError(f"SRT file does not exist: {subtitle_path}")
    return hook(subtitle_path)


class VideoEditor:
    """Orchestrate probing, silence detection, cut planning, and rendering."""

    def __init__(
        self,
        config: EditorConfig | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config or EditorConfig()
        self.progress_callback = progress_callback

    def probe(self, input_path: str | os.PathLike[str]) -> MediaInfo:
        return probe_media(input_path, self.config.ffprobe_path)

    def detect_silence(self, media: MediaInfo) -> list[TimeRange]:
        return detect_silence(
            media.path,
            ffmpeg_path=self.config.ffmpeg_path,
            threshold_db=self.config.silence_threshold_db,
            minimum_duration=self.config.silence_min_duration,
            media=media,
            progress_callback=self.progress_callback,
        )

    def edit(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        *,
        filler_ranges: Iterable[RangeLike] = (),
        extra_removals: Iterable[RangeLike] = (),
    ) -> EditResult:
        """Edit a video, removing detected silence and transcription ranges."""

        _notify(self.progress_callback, ProgressEvent("probe", 0.0, "Probing media"))
        media = self.probe(input_path)
        _notify(self.progress_callback, ProgressEvent("probe", 1.0, "Media probe complete"))
        detected = self.detect_silence(media) if self.config.remove_silence else []
        # Leave a short breath on either side instead of expanding cuts into
        # consonants, which made the original edits sound abrupt.
        padding = self.config.speech_padding
        silence = [TimeRange(item.start + padding, item.end - padding)
                   for item in detected if item.duration > 2 * padding]
        filler = [_coerce_range(item) for item in filler_ranges]
        _notify(self.progress_callback, ProgressEvent("cut_planning", 0.0, "Planning synchronized cuts"))
        cuts = build_cuts(
            media.duration,
            [*silence, *extra_removals],
            filler,
            silence_padding=self.config.silence_padding,
            filler_padding=self.config.filler_padding,
            minimum_clip_duration=self.config.minimum_clip_duration,
        )
        cuts = _align_cuts(cuts)
        _notify(
            self.progress_callback,
            ProgressEvent("cut_planning", 1.0, f"Planned {len(cuts)} retained clip(s)"),
        )
        rendered = render_cuts(
            input_path,
            output_path,
            media,
            cuts,
            config=self.config,
            progress_callback=self.progress_callback,
        )
        _notify(self.progress_callback, ProgressEvent("complete", 1.0, "Video edit complete"))
        return EditResult(
            input_path=Path(input_path).expanduser(),
            output_path=rendered,
            media=media,
            silence_ranges=tuple(silence),
            filler_ranges=tuple(filler),
            cuts=tuple(cuts),
        )


__all__ = [
    "CaptionHook",
    "Cut",
    "DependencyError",
    "EditResult",
    "EditorConfig",
    "FFmpegError",
    "FlowAnalysis",
    "MediaInfo",
    "MediaProbeError",
    "ProgressEvent",
    "TranscriptSegment",
    "TranscriptWord",
    "TimeRange",
    "VideoEditor",
    "WhisperRuntime",
    "apply_face_tracking",
    "apply_media_overlays",
    "analyze_transcript_flow",
    "burn_captions",
    "build_cuts",
    "detect_silence",
    "filler_word_ranges",
    "process_video",
    "probe_media",
    "render_cuts",
    "download_whisper_model",
    "normalize_whisper_compute_type",
    "normalize_whisper_device",
    "normalize_whisper_model",
    "resolve_whisper_runtime",
    "run_caption_hook",
    "transcribe_media",
    "write_srt",
]
